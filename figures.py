"""
figures.py
==========
Plotly figure builders for all dashboard charts.
"""

import numpy as np
import plotly.graph_objects as go

from config import (
    SAMPLE_FS, CLASS_COLORS, ARRHYTHMIA_CLASSES, ABLATION_CONFIGS,
    PLOT_TEMPLATE, FONT_FAMILY,
)
from utils import synthetic_training_curves


# ─────────────────────────────────────────────────────────────────────────────
# SHARED LAYOUT HELPER
# ─────────────────────────────────────────────────────────────────────────────

def fig_layout(fig: go.Figure, title=None, height=None) -> go.Figure:
    upd = dict(
        template      = PLOT_TEMPLATE,
        font          = dict(family=FONT_FAMILY, size=12, color="#444"),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        margin        = dict(l=40, r=20, t=40 if title else 20, b=40),
        showlegend    = True,
        legend        = dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
    )
    if title:  upd["title"]  = dict(text=title, font=dict(size=14, color="#222"))
    if height: upd["height"] = height
    fig.update_layout(**upd)
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", zeroline=False)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_waveform_fig(beat: np.ndarray, r_peaks=None,
                       title: str = "ECG Beat (300 samples @ 125 Hz)",
                       highlight_class: str = None) -> go.Figure:
    t     = np.arange(len(beat)) / SAMPLE_FS * 1000  # ms
    color = CLASS_COLORS.get(highlight_class, "#E24B4A") if highlight_class else "#E24B4A"
    fig   = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=beat, mode="lines", name="Signal",
        line=dict(color=color, width=1.8),
        hovertemplate="t=%{x:.1f} ms<br>amp=%{y:.3f}<extra></extra>",
    ))
    if r_peaks is not None and len(r_peaks):
        fig.add_trace(go.Scatter(
            x=np.array(r_peaks) / SAMPLE_FS * 1000, y=beat[r_peaks],
            mode="markers", name="R-peak",
            marker=dict(size=8, color="#FF9800", symbol="circle",
                        line=dict(width=1.5, color="#333")),
        ))
    fig.update_xaxes(title="Time (ms)")
    fig.update_yaxes(title="Normalized amplitude", range=[-0.1, 1.15])
    return fig_layout(fig, title=title, height=260)


def build_confidence_fig(probs: np.ndarray, class_names: list) -> go.Figure:
    colors = [CLASS_COLORS.get(c, "#888") for c in class_names]
    pred   = int(np.argmax(probs))
    border = ["rgba(0,0,0,0.5)" if i == pred else "rgba(0,0,0,0)" for i in range(len(class_names))]
    fig = go.Figure(go.Bar(
        x=class_names,
        y=[float(p) * 100 for p in probs],
        marker_color=colors,
        marker_line_color=border,
        marker_line_width=2,
        text=[f"{p * 100:.1f}%" for p in probs],
        textposition="outside",
        hovertemplate="%{x}: %{y:.2f}%<extra></extra>",
        name="Confidence",
    ))
    fig.update_yaxes(range=[0, 115], title="Confidence (%)")
    fig.update_xaxes(title="Class")
    return fig_layout(fig, title="Model Confidence", height=260)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_training_fig(tr_arr=None, val_arr=None, task: str = "arrhy") -> go.Figure:
    if tr_arr is None:
        if task == "arrhy":
            tr_arr, val_arr = synthetic_training_curves(0.945, 0.934, seed=1)
        else:
            tr_arr, val_arr = synthetic_training_curves(0.970, 0.959, seed=2)
    ep  = list(range(1, len(tr_arr) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ep, y=tr_arr * 100, mode="lines", name="Train acc",
                             line=dict(color="#378ADD", width=2)))
    fig.add_trace(go.Scatter(x=ep, y=val_arr * 100, mode="lines", name="Val / Test acc",
                             line=dict(color="#E24B4A", width=2, dash="dash")))
    fig.add_hline(y=93.4 if task == "arrhy" else 95.9,
                  line_dash="dot", line_color="#aaa",
                  annotation_text=f"Paper target: {'93.4' if task == 'arrhy' else '95.9'}%",
                  annotation_position="bottom right")
    fig.update_xaxes(title="Epoch")
    fig.update_yaxes(title="Accuracy (%)", range=[40, 102])
    title = ("Training Curves — Arrhythmia (MIT-BIH)" if task == "arrhy"
             else "Training Curves — MI Transfer (PTB)")
    return fig_layout(fig, title=title, height=320)


def build_class_dist_fig() -> go.Figure:
    classes = ARRHYTHMIA_CLASSES
    approx  = [75, 5, 12, 2, 6]   # % of raw MIT-BIH beats
    colors  = [CLASS_COLORS[c] for c in classes]
    fig = go.Figure(go.Bar(
        x=classes, y=approx,
        marker_color=colors,
        text=[f"{v}%" for v in approx],
        textposition="outside",
        hovertemplate="%{x}: ~%{y}% of raw beats<extra></extra>",
    ))
    fig.update_yaxes(range=[0, 90], title="Approx. % of raw beats")
    fig.update_xaxes(title="AAMI class")
    return fig_layout(fig, title="MIT-BIH Class Distribution (raw, before balancing)", height=280)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL OVERVIEW TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_arch_fig(n_res_blocks: int = 5) -> go.Figure:
    """Sankey-style architecture diagram."""
    layers = (
        ["Input\n(1×300)", "Conv1d\nk=5, 32ch"]
        + [f"ResBlock {i + 1}\n(÷2)" for i in range(n_res_blocks)]
        + ["GAP\n→32-d", "FC×2\n+ReLU", "Output\n5 cls"]
    )
    colors = (
        ["#E6F1FB", "#B5D4F4"]
        + ["#EEEDFE"] * n_res_blocks
        + ["#9FE1CB", "#5DCAA5", "#FAC775"]
    )
    n   = len(layers)
    fig = go.Figure()
    for i, (lbl, col) in enumerate(zip(layers, colors)):
        fig.add_shape(type="rect", x0=i * 1.1, x1=i * 1.1 + 0.9,
                      y0=0.2, y1=0.8, fillcolor=col,
                      line=dict(color="#ccc", width=1))
        fig.add_annotation(x=i * 1.1 + 0.45, y=0.5, text=lbl.replace("\n", "<br>"),
                           showarrow=False, font=dict(size=9.5, color="#333"),
                           align="center")
        if i < n - 1:
            fig.add_annotation(
                x=i * 1.1 + 0.9, y=0.5, ax=i * 1.1 + 0.9 + 0.001, ay=0.5,
                xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=1.5,
                arrowcolor="#999",
            )
    fig.update_xaxes(visible=False, range=[-0.1, n * 1.1])
    fig.update_yaxes(visible=False, range=[0, 1])
    fig.update_layout(
        height=180,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY),
        title=dict(text=f"ECGResNet  ({n_res_blocks} residual blocks)", font=dict(size=13, color="#222")),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# ABLATION TAB
# ─────────────────────────────────────────────────────────────────────────────

def build_ablation_fig() -> go.Figure:
    configs   = list(ABLATION_CONFIGS.keys())
    arrhy_acc = [ABLATION_CONFIGS[c]["arrhy_acc"] for c in configs]
    mi_acc    = [ABLATION_CONFIGS[c]["mi_acc"]    for c in configs]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Arrhythmia acc (%)", x=configs, y=arrhy_acc,
                         marker_color="#378ADD", text=[f"{v}%" for v in arrhy_acc],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="MI acc (%)", x=configs, y=mi_acc,
                         marker_color="#E24B4A", text=[f"{v}%" for v in mi_acc],
                         textposition="outside"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[84, 100], title="Accuracy (%)")
    fig.update_xaxes(title="Config")
    return fig_layout(fig, title="Ablation Study — Config Comparison", height=340)


def build_per_class_acc_fig() -> go.Figure:
    classes = ARRHYTHMIA_CLASSES
    accs    = [96.0, 82.0, 97.0, 79.0, 99.0]   # representative paper values
    colors  = [CLASS_COLORS[c] for c in classes]
    fig = go.Figure(go.Bar(
        x=classes, y=accs,
        marker_color=colors,
        text=[f"{v}%" for v in accs],
        textposition="outside",
        hovertemplate="%{x}: %{y}%<extra></extra>",
    ))
    fig.update_yaxes(range=[70, 106], title="Per-class accuracy (%)")
    fig.update_xaxes(title="AAMI class")
    return fig_layout(fig, title="Per-Class Accuracy (balanced test set, 819/class)", height=280)
