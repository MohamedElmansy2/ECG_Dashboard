"""
pages/ablation.py
=================
Layout for the Ablation Study tab.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from config import ABLATION_CONFIGS
from figures import build_ablation_fig, build_per_class_acc_fig
from layout_helpers import card, section_header


def make_ablation_layout():
    rows = []
    for name, cfg in ABLATION_CONFIGS.items():
        best = (name == "baseline")
        rows.append(html.Tr([
            html.Td([
                dbc.Badge(name, color="success" if best else "secondary"),
                " ★" if best else "",
            ]),
            html.Td(str(cfg["n_res_blocks"])),
            html.Td(dbc.Badge("Yes", color="success") if cfg["use_balancing"]
                    else dbc.Badge("No", color="danger")),
            html.Td(str(cfg["batch_size"])),
            html.Td(f"{cfg['arrhy_acc']}%",
                    style={"fontWeight": 700 if best else 400,
                           "color": "#1D9E75" if best else "inherit"}),
            html.Td(f"{cfg['mi_acc']}%",
                    style={"fontWeight": 700 if best else 400,
                           "color": "#1D9E75" if best else "inherit"}),
        ], className="table-success" if best else ""))

    return dbc.Container([

        # ── Config table ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(card([
                section_header("Ablation configurations"),
                dbc.Table(
                    [
                        html.Thead(html.Tr([
                            html.Th(c) for c in
                            ["Config", "Res blocks", "Balancing", "Batch", "Arrhy acc", "MI acc"]
                        ])),
                        html.Tbody(rows),
                    ],
                    bordered=True, hover=True, size="sm",
                ),
                dbc.Alert(
                    "★ Baseline is the paper reproduction target (93.4% arrhy, 95.9% MI). "
                    "Other rows show estimated performance from each ablation.",
                    color="light", className="mt-2 small",
                ),
            ]))
        ]),

        # ── Accuracy comparison chart + key findings ──────────────────────────
        dbc.Row([
            dbc.Col(card([
                section_header("Accuracy comparison"),
                dcc.Graph(figure=build_ablation_fig(), config={"displayModeBar": False}),
            ]), md=8),

            dbc.Col([
                card([
                    section_header("Key finding: class balancing"),
                    html.P(
                        "Removing WeightedRandomSampler causes bias toward N (~75% of raw data). "
                        "Rare classes S and F suffer the most — per-class recall drops significantly "
                        "even if overall accuracy looks acceptable.",
                        className="small",
                    ),
                    dbc.Badge("~3–5% accuracy drop", color="danger"),
                ]),
                card([
                    section_header("Key finding: block depth"),
                    html.P(
                        "3 blocks → seq length 38 before GAP (vs 10 with 5 blocks). "
                        "Smaller receptive field reduces multi-scale pattern capacity.",
                        className="small",
                    ),
                    dbc.Badge("~2–4% accuracy drop", color="warning"),
                ]),
            ], md=4),
        ]),

        # ── Per-class accuracy ────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(card([
                section_header("Per-class accuracy (baseline, balanced test set)"),
                dcc.Graph(figure=build_per_class_acc_fig(), config={"displayModeBar": False}),
            ]))
        ]),

    ], fluid=True)
