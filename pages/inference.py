"""
pages/inference.py
==================
Layout for the Live Inference tab.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import ARRHYTHMIA_CLASSES
from layout_helpers import card, section_header


inference_layout = dbc.Container([

    # ── Model checkpoints ────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            card([
                section_header("Model checkpoints"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Arrhythmia checkpoint (.pt)", html_for="arrhy-ckpt"),
                        dbc.Input(id="arrhy-ckpt", type="text",
                                  placeholder="e.g. output/arrhy_best.pt",
                                  debounce=True),
                    ], md=6),
                    dbc.Col([
                        dbc.Label("MI transfer checkpoint (.pt)", html_for="mi-ckpt"),
                        dbc.Input(id="mi-ckpt", type="text",
                                  placeholder="e.g. output/transfer_best.pt",
                                  debounce=True),
                    ], md=6),
                ], className="mb-2"),
                dbc.Row([
                    dbc.Col([
                        dbc.Label("Res blocks (must match checkpoint)", html_for="n-blocks"),
                        dbc.Select(id="n-blocks",
                                   options=[{"label": f"{n} blocks", "value": n} for n in [3, 4, 5, 6]],
                                   value=5),
                    ], md=4),
                    dbc.Col([
                        dbc.Label("Task", html_for="inf-task"),
                        dbc.Select(id="inf-task",
                                   options=[
                                       {"label": "Arrhythmia (5-class)", "value": "arrhy"},
                                       {"label": "MI detection (2-class)", "value": "mi"},
                                   ],
                                   value="arrhy"),
                    ], md=4),
                    dbc.Col([
                        dbc.Button("Load model", id="load-model-btn", color="primary",
                                   className="w-100 mt-4"),
                    ], md=4),
                ]),
                html.Div(id="model-status", className="mt-2"),
            ])
        ])
    ]),

    # ── Input signal + Run inference ─────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            card([
                section_header("Input signal"),
                dbc.Tabs([

                    dbc.Tab([
                        html.Div([
                            html.P(
                                "Upload a .npy file containing a single beat (shape: 300,) "
                                "or a raw signal array (will be preprocessed).",
                                className="text-muted small mt-3",
                            ),
                            dcc.Upload(
                                id="upload-npy",
                                children=html.Div([
                                    html.I(className="bi bi-upload me-2"),
                                    "Drag & drop or ", html.A("browse", href="#"),
                                    " (.npy or .csv)",
                                ]),
                                style={
                                    "width": "100%", "height": "80px", "lineHeight": "80px",
                                    "borderWidth": "1.5px", "borderStyle": "dashed",
                                    "borderRadius": "8px", "borderColor": "#ccc",
                                    "textAlign": "center", "cursor": "pointer",
                                },
                                multiple=False,
                            ),
                            html.Div(id="upload-status", className="mt-2 small text-muted"),
                        ])
                    ], label="Upload file", tab_id="tab-upload"),

                    dbc.Tab([
                        html.P("Pick a synthetic beat to see how the model responds.",
                               className="text-muted small mt-3"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Beat type"),
                                dbc.Select(
                                    id="demo-class",
                                    options=[
                                        {"label": f"{c} – {n}", "value": c}
                                        for c, n in zip(
                                            ARRHYTHMIA_CLASSES,
                                            ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"],
                                        )
                                    ],
                                    value="N",
                                ),
                            ], md=5),
                            dbc.Col([
                                dbc.Label("Noise level"),
                                dcc.Slider(
                                    id="noise-slider", min=0, max=0.15, step=0.01, value=0.02,
                                    marks={0: "0", 0.05: "0.05", 0.1: "0.1", 0.15: "0.15"},
                                    tooltip={"placement": "bottom", "always_visible": False},
                                ),
                            ], md=5),
                            dbc.Col([
                                dbc.Button("Generate", id="gen-demo-btn", color="success",
                                           className="w-100 mt-4"),
                            ], md=2),
                        ]),
                    ], label="Synthetic demo", tab_id="tab-demo"),

                ], id="input-tabs", active_tab="tab-demo"),
            ])
        ], md=6),

        dbc.Col([
            card([
                section_header("Run inference"),
                dbc.Row([
                    dbc.Col([
                        dbc.Button("▶  Run Inference", id="run-inf-btn",
                                   color="danger", size="lg", className="w-100"),
                    ], md=8),
                    dbc.Col([
                        dbc.Button("↺  Reset", id="reset-btn",
                                   color="secondary", outline=True, className="w-100"),
                    ], md=4),
                ], className="mb-3"),
                html.Div(id="prediction-badge", className="mb-2"),
                html.Hr(),
                section_header("Note: no checkpoint?"),
                html.P(
                    "If no .pt checkpoint is loaded, the dashboard runs with randomly-initialised "
                    "weights to demonstrate the interface. Load real checkpoints for meaningful predictions.",
                    className="text-muted small",
                ),
            ])
        ], md=6),
    ]),

    # ── Waveform + Confidence ─────────────────────────────────────────────────
    dbc.Row([
        dbc.Col([
            card([
                section_header("Waveform"),
                dcc.Graph(id="waveform-graph", config={"displayModeBar": False}),
            ])
        ], md=7),
        dbc.Col([
            card([
                section_header("Confidence"),
                dcc.Graph(id="confidence-graph", config={"displayModeBar": False}),
            ])
        ], md=5),
    ]),

    dcc.Store(id="current-beat-store"),
    dcc.Store(id="model-loaded-store", data={"loaded": False}),

], fluid=True)
