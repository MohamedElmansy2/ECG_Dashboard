"""
callbacks.py
============
All Dash callbacks for the ECG Deep Learning Dashboard.
"""

import io
import base64
import json

import numpy as np
import plotly.graph_objects as go
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, ctx, html

from config import ARRHYTHMIA_CLASSES, MI_CLASSES, CLASS_COLORS, N_ARRHY, N_MI
from model import ECGResNet
from utils import (
    synthetic_ecg, find_r_peaks, try_load_model, run_inference,
)
from figures import (
    build_waveform_fig, build_confidence_fig,
    build_training_fig, build_arch_fig, fig_layout,
)

# In-process model cache (reset on server restart)
_model_cache: dict = {}


def register_callbacks(app):
    """Attach all callbacks to the Dash app instance."""

    # ── ROUTING ───────────────────────────────────────────────────────────────
    from pages.inference import inference_layout
    from pages.overview  import make_overview_layout
    from pages.training  import training_layout
    from pages.ablation  import make_ablation_layout
    from pages.pipeline  import make_pipeline_layout

    @app.callback(Output("page-content", "children"), Input("url", "pathname"))
    def render_page(pathname):
        if pathname == "/model":    return make_overview_layout()
        if pathname == "/training": return training_layout
        if pathname == "/ablation": return make_ablation_layout()
        if pathname == "/pipeline": return make_pipeline_layout()
        return inference_layout   # default "/"

    # ── MODEL OVERVIEW ────────────────────────────────────────────────────────
    @app.callback(Output("arch-graph", "figure"), Input("arch-blocks-sel", "value"))
    def update_arch(n):
        return build_arch_fig(int(n))

    # ── TRAINING ──────────────────────────────────────────────────────────────
    @app.callback(
        Output("training-graph", "figure"),
        Input("curve-task",     "value"),
        Input("history-upload", "contents"),
        State("history-upload", "filename"),
        prevent_initial_call=False,
    )
    def update_training_graph(task, contents, filename):
        tr_arr = val_arr = None
        if contents and filename and filename.endswith(".npy"):
            try:
                raw     = base64.b64decode(contents.split(",")[1])
                hist    = np.load(io.BytesIO(raw), allow_pickle=True).item()
                tr_arr  = np.array(hist.get("tr_acc",  []))
                val_arr = np.array(hist.get("val_acc", hist.get("te_acc", [])))
            except Exception:
                pass
        return build_training_fig(tr_arr, val_arr, task)

    # ── INFERENCE: load model ─────────────────────────────────────────────────
    @app.callback(
        Output("model-status",       "children"),
        Output("model-loaded-store", "data"),
        Input("load-model-btn",      "n_clicks"),
        State("arrhy-ckpt",          "value"),
        State("mi-ckpt",             "value"),
        State("n-blocks",            "value"),
        State("inf-task",            "value"),
        prevent_initial_call=True,
    )
    def load_model(_, arrhy_path, mi_path, n_blocks, task):
        n_blocks = int(n_blocks)
        ckpt     = arrhy_path if task == "arrhy" else mi_path
        model, info = try_load_model(ckpt, n_blocks, task)
        if model:
            _model_cache["model"]   = model
            _model_cache["task"]    = task
            _model_cache["n_blocks"] = n_blocks
            return (dbc.Alert(info, color="success", dismissable=True),
                    {"loaded": True, "task": task})
        else:
            _model_cache.clear()
            return (dbc.Alert(info, color="warning", dismissable=True),
                    {"loaded": False})

    # ── INFERENCE: prepare beat ───────────────────────────────────────────────
    @app.callback(
        Output("current-beat-store", "data"),
        Output("upload-status",      "children"),
        Input("gen-demo-btn",  "n_clicks"),
        Input("upload-npy",    "contents"),
        State("upload-npy",    "filename"),
        State("demo-class",    "value"),
        State("noise-slider",  "value"),
        prevent_initial_call=True,
    )
    def prepare_beat(n_demo, contents, filename, demo_class, noise):
        triggered = ctx.triggered_id
        if triggered == "gen-demo-btn" or (triggered is None and n_demo):
            beat = synthetic_ecg(demo_class)
            beat = beat + np.random.randn(len(beat)).astype(np.float32) * float(noise)
            beat = np.clip(beat, 0, 1)
            return json.dumps(beat.tolist()), f"Synthetic {demo_class} beat generated."

        if contents:
            from config import INPUT_SIZE
            try:
                raw = base64.b64decode(contents.split(",")[1])
                if filename.endswith(".npy"):
                    arr = np.load(io.BytesIO(raw)).astype(np.float32).flatten()
                    if len(arr) >= INPUT_SIZE:
                        arr = arr[:INPUT_SIZE]
                    else:
                        arr = np.pad(arr, (0, INPUT_SIZE - len(arr)))
                    mn, mx = arr.min(), arr.max()
                    if mx != mn: arr = (arr - mn) / (mx - mn)
                    return json.dumps(arr.tolist()), f"Loaded {filename} ({len(arr)} samples)."
                elif filename.endswith(".csv"):
                    text   = raw.decode("utf-8")
                    values = [float(v) for line in text.strip().split("\n")
                              for v in line.replace(",", " ").split() if v.strip()]
                    arr    = np.array(values, dtype=np.float32).flatten()
                    if len(arr) >= INPUT_SIZE:
                        arr = arr[:INPUT_SIZE]
                    else:
                        arr = np.pad(arr, (0, INPUT_SIZE - len(arr)))
                    mn, mx = arr.min(), arr.max()
                    if mx != mn: arr = (arr - mn) / (mx - mn)
                    return json.dumps(arr.tolist()), f"Loaded {filename} → first {INPUT_SIZE} samples."
            except Exception as e:
                return dash.no_update, dbc.Alert(f"Error reading file: {e}", color="danger")

        return dash.no_update, dash.no_update

    # ── INFERENCE: run ────────────────────────────────────────────────────────
    @app.callback(
        Output("waveform-graph",    "figure"),
        Output("confidence-graph",  "figure"),
        Output("prediction-badge",  "children"),
        Input("run-inf-btn",         "n_clicks"),
        State("current-beat-store",  "data"),
        State("model-loaded-store",  "data"),
        State("inf-task",            "value"),
        State("demo-class",          "value"),
        prevent_initial_call=True,
    )
    def run_inference_cb(_, beat_json, model_info, task, demo_class):
        if not beat_json:
            empty = go.Figure()
            fig_layout(empty, height=260)
            return empty, empty, dbc.Alert("No beat loaded. Generate or upload one first.", color="warning")

        beat    = np.array(json.loads(beat_json), dtype=np.float32)
        r_peaks = find_r_peaks(beat)
        classes = ARRHYTHMIA_CLASSES if task == "arrhy" else MI_CLASSES

        model     = _model_cache.get("model")
        using_real = model is not None and _model_cache.get("task") == task

        if using_real:
            probs, pred_idx = run_inference(model, beat)
        else:
            demo_model = ECGResNet(N_ARRHY if task == "arrhy" else N_MI, 5)
            demo_model.eval()
            probs, pred_idx = run_inference(demo_model, beat)

        pred_class = classes[pred_idx]
        color      = CLASS_COLORS.get(pred_class, "#888")

        wave_fig = build_waveform_fig(beat, r_peaks,
                                      title="Input beat waveform",
                                      highlight_class=pred_class)
        conf_fig = build_confidence_fig(probs, classes)

        badge = html.Div([
            dbc.Badge(
                [html.I(className="bi bi-activity me-1"),
                 f"Prediction: {pred_class}  ({probs[pred_idx]*100:.1f}%)"],
                style={"fontSize": "1rem", "padding": "10px 20px", "background": color},
            ),
            html.Span(
                " (demo — load real checkpoint for meaningful results)" if not using_real else "",
                className="text-muted ms-2 small",
            ),
        ])
        return wave_fig, conf_fig, badge

    # ── INFERENCE: reset ──────────────────────────────────────────────────────
    @app.callback(
        Output("current-beat-store", "data",     allow_duplicate=True),
        Output("waveform-graph",     "figure",   allow_duplicate=True),
        Output("confidence-graph",   "figure",   allow_duplicate=True),
        Output("prediction-badge",   "children", allow_duplicate=True),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def reset(_):
        empty = go.Figure()
        fig_layout(empty, height=260)
        return None, empty, empty, ""
