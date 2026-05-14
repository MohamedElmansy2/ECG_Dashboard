"""
pages/overview.py
=================
Layout for the Model Overview tab.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

from model import ECGResNet
from figures import build_arch_fig
from layout_helpers import card, section_header, metric_card


def make_overview_layout():
    m5     = ECGResNet(5, 5)
    total  = sum(p.numel() for p in m5.parameters())
    conv_p = sum(p.numel() for n, p in m5.named_parameters() if "fc" not in n and "out" not in n)
    fc_p   = total - conv_p

    dim_table = dbc.Table([
        html.Thead(html.Tr([html.Th(c) for c in ["Stage", "Channels", "Seq length", "Notes"]])),
        html.Tbody([
            html.Tr([html.Td("Input"),         html.Td("1"),  html.Td("300"), html.Td("Normalised beat")]),
            html.Tr([html.Td("First Conv1d"),  html.Td("32"), html.Td("300"), html.Td("k=5, pad=2 (same)")]),
            html.Tr([html.Td("After Block 1"), html.Td("32"), html.Td("150"), html.Td("MaxPool stride=2")]),
            html.Tr([html.Td("After Block 2"), html.Td("32"), html.Td("75"),  html.Td("MaxPool stride=2")]),
            html.Tr([html.Td("After Block 3"), html.Td("32"), html.Td("38"),  html.Td("⌈75/2⌉")]),
            html.Tr([html.Td("After Block 4"), html.Td("32"), html.Td("19"),  html.Td("⌈38/2⌉")]),
            html.Tr([html.Td("After Block 5"), html.Td("32"), html.Td("10"),  html.Td("⌈19/2⌉")]),
            html.Tr([html.Td("GAP"),           html.Td("32"), html.Td("1"),   html.Td("→ 32-d vector")],
                    className="table-info"),
            html.Tr([html.Td("FC head out"),   html.Td("N"),  html.Td("—"),   html.Td("5 (arrhy) or 2 (MI)")]),
        ]),
    ], bordered=True, hover=True, size="sm", className="mb-0")

    return dbc.Container([
        dbc.Row([
            dbc.Col(metric_card("Total parameters",    f"{total:,}",   "baseline (5 blocks)", "#378ADD"), md=3),
            dbc.Col(metric_card("Conv/pool params",    f"{conv_p:,}",  "backbone (frozen for MI)", "#7F77DD"), md=3),
            dbc.Col(metric_card("FC head params",      f"{fc_p:,}",    "fine-tuned for MI", "#1D9E75"), md=3),
            dbc.Col(metric_card("GAP feature dim",     "32-d",         "transferred representation", "#EF9F27"), md=3),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col(card([
                section_header("Architecture diagram"),
                dbc.Label("Residual blocks", html_for="arch-blocks-sel"),
                dbc.Select(id="arch-blocks-sel",
                           options=[{"label": f"{n} blocks", "value": n} for n in [3, 4, 5, 6]],
                           value=5, style={"maxWidth": "160px"}),
                dcc.Graph(id="arch-graph", config={"displayModeBar": False}),
            ]))
        ]),

        dbc.Row([
            dbc.Col(card([
                section_header("Signal dimensions through network (5 blocks)"),
                dim_table,
            ]), md=6),
            dbc.Col(card([
                section_header("Transfer learning setup"),
                dbc.Table([
                    html.Thead(html.Tr([html.Th("Component"), html.Th("Status"), html.Th("Details")])),
                    html.Tbody([
                        html.Tr([html.Td("First Conv1d"),   html.Td(dbc.Badge("Frozen",    color="danger")),
                                 html.Td("Pretrained on MIT-BIH")]),
                        html.Tr([html.Td("ResidualBlocks"), html.Td(dbc.Badge("Frozen",    color="danger")),
                                 html.Td("Feature extractor")]),
                        html.Tr([html.Td("GAP"),            html.Td(dbc.Badge("Frozen",    color="danger")),
                                 html.Td("32-d representation")]),
                        html.Tr([html.Td("FC layer 1"),     html.Td(dbc.Badge("Trainable", color="success")),
                                 html.Td("32→32 + ReLU, new for PTB")]),
                        html.Tr([html.Td("FC layer 2"),     html.Td(dbc.Badge("Trainable", color="success")),
                                 html.Td("32→32 + ReLU")]),
                        html.Tr([html.Td("Output FC"),      html.Td(dbc.Badge("Trainable", color="success")),
                                 html.Td("32→2, MI vs Normal")]),
                    ]),
                ], bordered=True, hover=True, size="sm"),
                dbc.Alert(
                    "Backbone uses torch.no_grad() during forward pass in transfer mode. "
                    "Only the new FC head receives gradients.",
                    color="info", className="mt-2 small",
                ),
            ]), md=6),
        ]),
    ], fluid=True)
