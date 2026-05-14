"""
utils.py
========
Signal processing utilities for ECG preprocessing and inference helpers.
"""

import math
import traceback

import numpy as np
import scipy.signal as spsig
import torch

from config import INPUT_SIZE, SAMPLE_FS, N_ARRHY, N_MI, DEVICE
from model import ECGResNet, ECGResNetTransfer


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def resample_signal(sig: np.ndarray, orig_fs: float, target_fs: float = SAMPLE_FS) -> np.ndarray:
    if orig_fs == target_fs:
        return sig.astype(np.float32)
    g = math.gcd(int(orig_fs), int(target_fs))
    return spsig.resample_poly(sig.astype(np.float32), target_fs // g, orig_fs // g).astype(np.float32)


def normalize_window(win: np.ndarray):
    w_min, w_max = float(win.min()), float(win.max())
    if w_max == w_min:
        return None
    return (win - w_min) / (w_max - w_min)


def find_r_peaks(win_norm: np.ndarray) -> np.ndarray:
    dx  = np.diff(win_norm)
    idx = np.where((dx[:-1] > 0) & (dx[1:] <= 0))[0] + 1
    if len(idx) == 0:
        return np.array([], dtype=int)
    r = idx[win_norm[idx] >= 0.9]
    min_d = int(0.25 * SAMPLE_FS)
    if len(r) > 1:
        f = [int(r[0])]
        for pk in r[1:]:
            if int(pk) - f[-1] >= min_d:
                f.append(int(pk))
        r = np.array(f, dtype=int)
    return r


def extract_beat(win_norm: np.ndarray, center: int, T: float) -> np.ndarray:
    seg_len = int(round(1.2 * T))
    half    = seg_len // 2
    p_start = max(0, center - half)
    p_end   = p_start + seg_len
    if p_end > len(win_norm):
        p_end   = len(win_norm)
        p_start = max(0, p_end - seg_len)
    beat = win_norm[p_start:p_end].copy()
    if len(beat) >= INPUT_SIZE:
        beat = beat[:INPUT_SIZE]
    else:
        beat = np.pad(beat, (0, INPUT_SIZE - len(beat)))
    return beat.astype(np.float32)


def preprocess_raw_signal(sig_raw: np.ndarray, orig_fs: float = 360):
    """Full pipeline: raw signal → list of (beat_array, r_peak_center)."""
    resampled  = resample_signal(sig_raw, orig_fs)
    window_len = 10 * SAMPLE_FS   # 1250 samples
    beats = []
    for start in range(0, len(resampled) - window_len + 1, window_len):
        win      = resampled[start:start + window_len].copy()
        win_norm = normalize_window(win)
        if win_norm is None:
            continue
        r_peaks = find_r_peaks(win_norm)
        if len(r_peaks) < 2:
            continue
        T = float(np.median(np.diff(r_peaks)))
        T = np.clip(T, SAMPLE_FS * 60 / 200, SAMPLE_FS * 60 / 30)
        for center in r_peaks:
            beat = extract_beat(win_norm, int(center), T)
            beats.append((beat, start + int(center)))
    return beats, resampled


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING & INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def try_load_model(ckpt_path: str, n_res_blocks: int = 5, task: str = "arrhy"):
    """
    Returns (model, info_str) or (None, error_str).
    task = 'arrhy' | 'mi'
    """
    import os
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None, f"Checkpoint not found: {ckpt_path}"
    try:
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        if task == "arrhy":
            model = ECGResNet(num_classes=N_ARRHY, n_res_blocks=n_res_blocks)
            model.load_state_dict(ckpt["model_state"])
            acc  = ckpt.get("best_val_acc", "?")
            info = f"Loaded arrhythmia model (val_acc={acc:.4f} @ epoch {ckpt.get('epoch', '?')})"
        else:
            base  = ECGResNet(num_classes=N_ARRHY, n_res_blocks=n_res_blocks)
            model = ECGResNetTransfer(base, num_classes=N_MI)
            model.load_state_dict(ckpt["model_state"])
            acc  = ckpt.get("best_acc", "?")
            info = f"Loaded MI transfer model (best_acc={acc:.4f})"
        model.eval()
        return model, info
    except Exception as e:
        return None, f"Load error: {e}\n{traceback.format_exc()}"


def run_inference(model, beat_array: np.ndarray):
    """beat_array shape: (300,) → returns (probs np.ndarray, pred_idx int)"""
    x = torch.tensor(beat_array, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,300)
    with torch.no_grad():
        logits = model(x)
        probs  = torch.softmax(logits, dim=1).squeeze().numpy()
    pred = int(np.argmax(probs))
    return probs, pred


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHETIC DATA FOR DEMOS
# ─────────────────────────────────────────────────────────────────────────────

def synthetic_ecg(beat_class: str = "N", length: int = 300, fs: int = 125) -> np.ndarray:
    """Return a plausible synthetic ECG beat for demonstration."""
    t = np.linspace(0, 1, length)
    if beat_class == "N":
        sig  = 0.05 * np.sin(2 * np.pi * 1 * t)
        sig += np.exp(-((t - 0.40) ** 2) / (2 * 0.006 ** 2)) * (-0.12)   # Q
        sig += np.exp(-((t - 0.45) ** 2) / (2 * 0.003 ** 2)) * 1.00       # R
        sig += np.exp(-((t - 0.50) ** 2) / (2 * 0.006 ** 2)) * (-0.08)   # S
        sig += np.exp(-((t - 0.60) ** 2) / (2 * 0.020 ** 2)) * 0.22       # T
    elif beat_class == "V":
        sig  = np.exp(-((t - 0.45) ** 2) / (2 * 0.020 ** 2)) * 1.00
        sig += np.exp(-((t - 0.55) ** 2) / (2 * 0.025 ** 2)) * (-0.50)
        sig += 0.03 * np.sin(2 * np.pi * 2 * t)
    elif beat_class == "S":
        sig  = 0.08 * np.sin(2 * np.pi * 1.5 * t)
        sig += np.exp(-((t - 0.44) ** 2) / (2 * 0.004 ** 2)) * 0.85
        sig += np.exp(-((t - 0.58) ** 2) / (2 * 0.018 ** 2)) * 0.18
    elif beat_class == "F":
        sig  = np.exp(-((t - 0.45) ** 2) / (2 * 0.010 ** 2)) * 0.70
        sig += np.exp(-((t - 0.50) ** 2) / (2 * 0.015 ** 2)) * 0.40
        sig += 0.06 * np.sin(2 * np.pi * 3 * t)
    else:  # Q
        sig  = 0.10 * np.sin(2 * np.pi * 0.5 * t) + np.random.randn(length) * 0.05
        sig += np.exp(-((t - 0.45) ** 2) / (2 * 0.015 ** 2)) * 0.55
    noise = np.random.randn(length) * 0.012
    sig   = sig + noise
    mn, mx = sig.min(), sig.max()
    if mx != mn:
        sig = (sig - mn) / (mx - mn)
    return sig.astype(np.float32)


def synthetic_training_curves(final_tr: float = 0.945, final_val: float = 0.934,
                               epochs: int = 50, seed: int = 0):
    rng = np.random.RandomState(seed)
    ep  = np.arange(1, epochs + 1)
    tr  = final_tr  * (1 - np.exp(-5 * ep / epochs)) + rng.randn(epochs) * 0.012
    val = final_val * (1 - np.exp(-4.5 * ep / epochs)) + rng.randn(epochs) * 0.018
    return np.clip(tr, 0, 1), np.clip(val, 0, 1)
