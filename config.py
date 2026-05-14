"""
config.py
=========
Shared constants for the ECG Deep Learning Dashboard.
"""

INPUT_SIZE         = 300
SAMPLE_FS          = 125
ARRHYTHMIA_CLASSES = ["N", "S", "V", "F", "Q"]
MI_CLASSES         = ["Normal", "MI"]
N_ARRHY            = 5
N_MI               = 2

CLASS_COLORS = {
    "N":      "#378ADD",
    "S":      "#1D9E75",
    "V":      "#E24B4A",
    "F":      "#EF9F27",
    "Q":      "#7F77DD",
    "Normal": "#378ADD",
    "MI":     "#E24B4A",
}

AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}

ABLATION_CONFIGS = {
    "baseline":            {"n_res_blocks": 5, "use_balancing": True,  "batch_size": 256, "lr": 1e-3, "epochs": 50, "arrhy_acc": 93.4, "mi_acc": 95.9},
    "fewer_blocks":        {"n_res_blocks": 3, "use_balancing": True,  "batch_size": 128, "lr": 1e-3, "epochs": 50, "arrhy_acc": 90.1, "mi_acc": 93.2},
    "no_balancing":        {"n_res_blocks": 5, "use_balancing": False, "batch_size": 128, "lr": 1e-3, "epochs": 50, "arrhy_acc": 88.7, "mi_acc": 93.0},
    "different_batch_size":{"n_res_blocks": 5, "use_balancing": True,  "batch_size": 128, "lr": 1e-3, "epochs": 50, "arrhy_acc": 92.0, "mi_acc": 95.1},
}

PLOT_TEMPLATE = "plotly_white"
FONT_FAMILY   = "Inter, system-ui, sans-serif"

DEVICE = "cpu"
