"""
pages/pipeline.py
=================
Layout for the Data Pipeline tab.
"""

import dash_bootstrap_components as dbc
from dash import html

from layout_helpers import card, section_header


def make_pipeline_layout():
    steps = [
        ("bi-file-earmark-binary", "#E6F1FB", "#185FA5", "Raw WFDB record + annotations"),
        ("bi-sliders",             "#EEEDFE", "#534AB7", "Lead selection\n(MLII preferred → Lead II → ch 0)"),
        ("bi-graph-up",            "#E1F5EE", "#0F6E56", "Resample to 125 Hz\n(scipy.signal.resample_poly)"),
        ("bi-scissors",            "#FAF0E0", "#854F0B", "10-second windows\n(1250 samples, non-overlapping)"),
        ("bi-bar-chart-line",      "#FAECE7", "#993C1D", "Min-max normalize window\n[0, 1]"),
        ("bi-activity",            "#FAEEDA", "#854F0B", "Detect R-peaks\n(threshold=0.9, min dist=31)"),
        ("bi-box-arrow-in-right",  "#EAF3DE", "#3B6D11", "Extract 1.2T beat segment\npad/crop → 300 samples"),
        ("bi-tag",                 "#EEEDFE", "#534AB7", "AAMI label mapping\n→ one-hot (5-class)"),
    ]

    icons = []
    for ic, bg, fg, lbl in steps:
        icons.append(
            html.Div([
                html.Div(
                    html.I(className=f"bi {ic}", style={"fontSize": "1.4rem", "color": fg}),
                    style={
                        "width": "48px", "height": "48px", "borderRadius": "50%",
                        "background": bg, "display": "flex", "alignItems": "center",
                        "justifyContent": "center", "margin": "0 auto",
                    },
                ),
                html.P(
                    lbl,
                    className="text-muted mt-2 mb-0",
                    style={"fontSize": "0.72rem", "textAlign": "center",
                           "whiteSpace": "pre-line", "lineHeight": "1.35"},
                ),
            ], style={"minWidth": "88px", "maxWidth": "96px"})
        )
        if (ic, bg, fg, lbl) != steps[-1]:
            icons.append(html.Span(
                "→",
                style={"alignSelf": "flex-start", "paddingTop": "12px",
                       "color": "#bbb", "fontSize": "1.2rem"},
            ))

    aami_table = dbc.Table([
        html.Thead(html.Tr([html.Th("AAMI class"), html.Th("Symbols"), html.Th("Description")])),
        html.Tbody([
            html.Tr([html.Td(dbc.Badge("N", color="primary")),                       html.Td("N, L, R, e, j"), html.Td("Normal")]),
            html.Tr([html.Td(dbc.Badge("S", color="success")),                       html.Td("A, a, J, S"),   html.Td("Supraventricular ectopic")]),
            html.Tr([html.Td(dbc.Badge("V", color="danger")),                        html.Td("V, E"),         html.Td("Ventricular ectopic")]),
            html.Tr([html.Td(dbc.Badge("F", color="warning", text_color="dark")),    html.Td("F"),            html.Td("Fusion beat")]),
            html.Tr([html.Td(dbc.Badge("Q", color="secondary")),                     html.Td("/, f, Q"),      html.Td("Unknown / Paced")]),
        ]),
    ], bordered=True, hover=True, size="sm")

    param_table = dbc.Table([
        html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value"), html.Th("Notes")])),
        html.Tbody([
            html.Tr([html.Td("Sample rate"),   html.Td("125 Hz"),       html.Td("Resampled from original FS")]),
            html.Tr([html.Td("Input size"),    html.Td("300 samples"),  html.Td("~2.4 s at 125 Hz")]),
            html.Tr([html.Td("Window length"), html.Td("1250 samples"), html.Td("10 s @ 125 Hz")]),
            html.Tr([html.Td("Beat segment"),  html.Td("1.2 × T"),      html.Td("T = median RR interval")]),
            html.Tr([html.Td("Min RR"),        html.Td("37.5 samples"), html.Td("125 × 60/200 (200 bpm)")]),
            html.Tr([html.Td("Max RR"),        html.Td("250 samples"),  html.Td("125 × 60/30  (30 bpm)")]),
            html.Tr([html.Td("R-peak thresh"), html.Td("≥ 0.9"),        html.Td("After normalisation")]),
            html.Tr([html.Td("Min peak dist"), html.Td("31 samples"),   html.Td("0.25 s @ 125 Hz")]),
        ]),
    ], bordered=True, hover=True, size="sm")

    return dbc.Container([

        # ── Pipeline flow ─────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(card([
                section_header("Preprocessing pipeline"),
                html.Div(
                    icons,
                    style={
                        "display": "flex", "alignItems": "flex-start",
                        "gap": "4px", "flexWrap": "wrap", "rowGap": "20px",
                    },
                ),
            ]))
        ]),

        # ── AAMI map + signal parameters ─────────────────────────────────────
        dbc.Row([
            dbc.Col(card([section_header("AAMI class mapping"),           aami_table]),  md=6),
            dbc.Col(card([section_header("Signal processing parameters"), param_table]), md=6),
        ]),

        # ── Output files ──────────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(card([
                section_header("Output files"),
                dbc.Table([
                    html.Thead(html.Tr([html.Th("File"), html.Th("Description")])),
                    html.Tbody([
                        html.Tr([html.Td(html.Code("X_test_arrhy.npy / Y_test_arrhy.npy")),
                                 html.Td("Balanced arrhythmia test set (4095 beats)")]),
                        html.Tr([html.Td(html.Code("X_ptb_te.npy / Y_ptb_te_oh.npy")),
                                 html.Td("PTB MI test set (one-hot labels)")]),
                        html.Tr([html.Td(html.Code("arrhy_best.pt")),
                                 html.Td("Best arrhythmia checkpoint (by val acc)")]),
                        html.Tr([html.Td(html.Code("transfer_best.pt")),
                                 html.Td("Best MI transfer checkpoint (by test acc)")]),
                        html.Tr([html.Td(html.Code("arrhy_history_*.npy")),
                                 html.Td("Training curve dict {tr_acc, val_acc}")]),
                        html.Tr([html.Td(html.Code("transfer_history_*.npy")),
                                 html.Td("Transfer curve dict {tr_acc, te_acc}")]),
                    ]),
                ], bordered=True, hover=True, size="sm"),
                dbc.Alert([
                    html.Strong("Training: "),  html.Code("python train.py --config baseline"),    html.Br(),
                    html.Strong("Evaluate: "),  html.Code("python evaluate.py --config baseline"), html.Br(),
                    html.Strong("Dashboard: "), html.Code("python app.py"),
                ], color="light", className="mt-2 small font-monospace"),
            ]))
        ]),

    ], fluid=True)
