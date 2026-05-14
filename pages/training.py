"""
pages/training.py
=================
Layout for the Training tab.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from figures import build_class_dist_fig
from layout_helpers import card, section_header, metric_card


training_layout = dbc.Container([

    # ── Metric cards ──────────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(metric_card("Optimizer", "Adam",          "β=(0.9, 0.999)"),             md=3),
        dbc.Col(metric_card("LR",        "1e-3",          "StepLR ×0.75 / 10k iters"),   md=3),
        dbc.Col(metric_card("Epochs",    "50",            "both tasks"),                  md=3),
        dbc.Col(metric_card("Loss",      "CrossEntropy",  "no label smoothing"),          md=3),
    ], className="mb-3"),

    # ── Training curves ───────────────────────────────────────────────────────
    dbc.Row([
        dbc.Col(card([
            section_header("Training curves"),
            dbc.Row([
                dbc.Col([
                    dbc.Label("Task"),
                    dbc.Select(
                        id="curve-task",
                        options=[
                            {"label": "Arrhythmia (MIT-BIH)", "value": "arrhy"},
                            {"label": "MI Transfer (PTB)",    "value": "mi"},
                        ],
                        value="arrhy",
                    ),
                ], md=4),
                dbc.Col([
                    dbc.Label("History file (.npy, optional)"),
                    dcc.Upload(
                        id="history-upload",
                        children=html.Div(["Drop history .npy"]),
                        style={
                            "borderWidth": "1px", "borderStyle": "dashed",
                            "borderRadius": "6px", "padding": "6px 12px",
                            "textAlign": "center", "cursor": "pointer",
                            "borderColor": "#ccc", "fontSize": "0.82rem",
                        },
                    ),
                ], md=8),
            ], className="mb-2"),
            dcc.Graph(id="training-graph", config={"displayModeBar": False}),
        ]))
    ]),

    # ── Dataset splits + class distribution ───────────────────────────────────
    dbc.Row([
        dbc.Col(card([
            section_header("Dataset splits (MIT-BIH)"),
            dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Split"), html.Th("Size"), html.Th("Method"), html.Th("Notes"),
                ])),
                html.Tbody([
                    html.Tr([html.Td("Train"),      html.Td("~72%"),
                             html.Td("Stratified by class"),
                             html.Td("WeightedRandomSampler for balancing")]),
                    html.Tr([html.Td("Validation"), html.Td("~8%"),
                             html.Td("10% of train"),
                             html.Td("Used for checkpoint saving")]),
                    html.Tr([html.Td("Test"),       html.Td("~20% → balanced"),
                             html.Td("819/class (4,095 total)"),
                             html.Td("make_balanced_test() with seed=42")],
                            className="table-success"),
                ]),
            ], bordered=True, hover=True, size="sm"),
        ]), md=6),

        dbc.Col(card([
            section_header("Class balancing: WeightedRandomSampler"),
            html.P(
                "Weight = 1 / class_count. Rare classes (S, F, Q) get higher sampling probability.",
                className="text-muted small mb-2",
            ),
            dcc.Graph(
                id="class-dist-graph",
                figure=build_class_dist_fig(),
                config={"displayModeBar": False},
            ),
        ]), md=6),
    ]),

], fluid=True)
