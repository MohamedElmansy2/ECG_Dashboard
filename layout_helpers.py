"""
layout_helpers.py
=================
Reusable Dash/DBC UI component factories shared across page layouts.
"""

import dash_bootstrap_components as dbc
from dash import html


def card(children, className: str = "", style: dict = None) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(children),
        className=f"shadow-sm mb-3 {className}",
        style=style or {},
    )


def section_header(text: str) -> html.H6:
    return html.H6(
        text,
        className="text-uppercase text-muted fw-bold mb-3",
        style={"letterSpacing": "0.07em", "fontSize": "0.72rem"},
    )


def metric_card(label: str, value: str, sub: str = None, color: str = "#378ADD") -> dbc.Card:
    return dbc.Card(
        dbc.CardBody([
            html.P(label, className="text-muted mb-1", style={"fontSize": "0.78rem"}),
            html.H4(value, style={"fontWeight": 700, "color": color}),
            html.P(sub, className="text-muted mb-0", style={"fontSize": "0.72rem"}) if sub else None,
        ]),
        className="shadow-sm text-center",
    )
