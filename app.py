"""
app.py
======
Entry point for the ECG Deep Learning Dashboard.

Run:
    python app.py

Then open  http://127.0.0.1:8050  in your browser.

Tabs
----
/           — Live Inference  (upload beat / synthetic demo)
/model      — Architecture overview
/training   — Training curves & dataset stats
/ablation   — Config comparison
/pipeline   — Data preprocessing reference
"""

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

from callbacks import register_callbacks

# ─────────────────────────────────────────────────────────────────────────────
# APP INIT
# ─────────────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.FLATLY,
        "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.min.css",
    ],
    title="ECG DL Dashboard",
    suppress_callback_exceptions=True,
)
server = app.server   # expose Flask server for deployment (gunicorn, etc.)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
SIDEBAR_STYLE = {
    "position": "fixed", "top": 0, "left": 0, "bottom": 0,
    "width": "230px", "padding": "2rem 1rem",
    "background": "#1a2236", "zIndex": 1000,
}
CONTENT_STYLE = {"marginLeft": "230px", "padding": "1.5rem 1.5rem"}

sidebar = html.Div([
    html.Div([
        html.I(className="bi bi-heart-pulse-fill me-2",
               style={"color": "#E24B4A", "fontSize": "1.3rem"}),
        html.Span("ECG Dashboard",
                  style={"color": "#fff", "fontWeight": 700, "fontSize": "1.05rem"}),
    ], className="d-flex align-items-center mb-1"),
    html.P("MIT-BIH · PTB · ResNet", style={"color": "#8895a7", "fontSize": "0.72rem"}),
    html.Hr(style={"borderColor": "#2e3f5c"}),
    dbc.Nav([
        dbc.NavLink([html.I(className="bi bi-cpu-fill me-2"),          "Live Inference"],
                    href="/",         active="exact", id="nav-inf"),
        dbc.NavLink([html.I(className="bi bi-diagram-3 me-2"),         "Model Overview"],
                    href="/model",    active="exact"),
        dbc.NavLink([html.I(className="bi bi-graph-up me-2"),          "Training"],
                    href="/training", active="exact"),
        dbc.NavLink([html.I(className="bi bi-bar-chart-steps me-2"),   "Ablation Study"],
                    href="/ablation", active="exact"),
        dbc.NavLink([html.I(className="bi bi-arrow-right-circle me-2"),"Data Pipeline"],
                    href="/pipeline", active="exact"),
    ], vertical=True, pills=True,
       style={"--bs-nav-link-color": "#8895a7",
              "--bs-nav-pills-link-active-bg": "#2563eb"}),
    html.Hr(style={"borderColor": "#2e3f5c"}),
    html.P([
        "Paper target ", html.Br(),
        html.Strong("Arrhy: 93.4%", style={"color": "#60a5fa"}), html.Br(),
        html.Strong("MI: 95.9%",    style={"color": "#34d399"}),
    ], style={"color": "#8895a7", "fontSize": "0.75rem"}),
], style=SIDEBAR_STYLE)

# ─────────────────────────────────────────────────────────────────────────────
# ROOT LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
app.layout = html.Div([
    dcc.Location(id="url"),
    sidebar,
    html.Div(id="page-content", style=CONTENT_STYLE),
])

# ─────────────────────────────────────────────────────────────────────────────
# REGISTER CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
register_callbacks(app)

# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  ECG Deep Learning Dashboard")
    print("=" * 60)
    print("  Open:  http://127.0.0.1:8050")
    print("  Stop:  Ctrl+C")
    print()
    print("  Tabs:")
    print("    /           — Live Inference (upload beat / synthetic demo)")
    print("    /model      — Architecture overview")
    print("    /training   — Training curves & dataset stats")
    print("    /ablation   — Config comparison")
    print("    /pipeline   — Data preprocessing reference")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=8050)
