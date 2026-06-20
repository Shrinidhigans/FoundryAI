"""Industrial dark theme — CSS tokens, Plotly layout, and color palettes."""

from __future__ import annotations

# ── Base palette ─────────────────────────────────────────────────────────────
BG_PRIMARY = "#0d1117"
BG_SECONDARY = "#161b22"
BG_CARD = "#1c2128"
BG_ELEVATED = "#21262d"
BORDER = "#30363d"

TEXT_PRIMARY = "#e6edf3"
TEXT_MUTED = "#8b949e"
TEXT_DIM = "#6e7681"

ACCENT = "#58a6ff"
SUCCESS = "#3fb950"
WARNING = "#d29922"
DANGER = "#f85149"
CRITICAL = "#ff6b6b"

# ── Semantic maps ────────────────────────────────────────────────────────────
RISK_COLORS = {
    "HEALTHY": SUCCESS,
    "NORMAL": SUCCESS,
    "LOW": "#79c0ff",
    "MEDIUM": WARNING,
    "HIGH": DANGER,
    "CRITICAL": CRITICAL,
}

SEVERITY_COLORS = {
    "NORMAL": SUCCESS,
    "LOW": "#79c0ff",
    "MEDIUM": WARNING,
    "HIGH": DANGER,
    "CRITICAL": CRITICAL,
}

REC_COLORS = {
    "PROCEED": SUCCESS,
    "MONITOR": "#f2cc60",
    "HOLD": "#f0883e",
    "STOP": DANGER,
}

FEATURE_GROUP_COLORS = {
    "chemistry": "#58a6ff",
    "thermal": "#f0883e",
    "additive": "#a371f7",
    "engineered": "#3fb950",
    "other": "#8b949e",
}

# ── Plotly defaults ──────────────────────────────────────────────────────────
PLOTLY_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": BG_CARD,
    "font": {"family": "Segoe UI, Roboto, Helvetica, Arial, sans-serif", "color": TEXT_PRIMARY, "size": 12},
    "margin": {"l": 48, "r": 24, "t": 48, "b": 40},
    "xaxis": {
        "gridcolor": BORDER,
        "linecolor": BORDER,
        "tickcolor": TEXT_MUTED,
        "zerolinecolor": BORDER,
    },
    "yaxis": {
        "gridcolor": BORDER,
        "linecolor": BORDER,
        "tickcolor": TEXT_MUTED,
        "zerolinecolor": BORDER,
    },
    "legend": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": BORDER, "font": {"color": TEXT_MUTED}},
    "colorway": [ACCENT, SUCCESS, WARNING, DANGER, "#a371f7", "#f0883e"],
    "hoverlabel": {"bgcolor": BG_ELEVATED, "bordercolor": BORDER, "font": {"color": TEXT_PRIMARY}},
}


def plotly_config() -> dict:
    return {
        "displayModeBar": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
        "responsive": True,
    }


def inject_global_css() -> None:
    import streamlit as st

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {BG_PRIMARY} 0%, #0a0e14 100%);
            color: {TEXT_PRIMARY};
        }}
        [data-testid="stSidebar"] {{
            background: {BG_SECONDARY};
            border-right: 1px solid {BORDER};
            transition: width 180ms ease, transform 180ms ease;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            display: flex;
            flex-direction: column;
            min-height: 100vh;
            padding-bottom: 1rem;
            scrollbar-gutter: stable;
        }}
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
            gap: 0.72rem;
            min-height: calc(100vh - 6rem);
        }}
        [data-testid="stSidebar"] [data-testid="stElementContainer"]:has(.sidebar-footer) {{
            margin-top: auto;
            position: sticky;
            bottom: 0;
            z-index: 5;
            background: linear-gradient(180deg, rgba(22,27,34,0.94) 0%, {BG_SECONDARY} 42%);
        }}
        [data-testid="stSidebar"] .stRadio label {{
            padding: 0.48rem 0.6rem;
            border-radius: 8px;
            border: 1px solid transparent;
            transition: background 140ms ease, border-color 140ms ease, transform 140ms ease;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            background: {BG_ELEVATED};
            border-color: {BORDER};
            transform: translateX(2px);
        }}
        [data-testid="stSidebar"] [aria-checked="true"] {{
            background: {ACCENT}18 !important;
            border-color: {ACCENT}55 !important;
        }}
        .sidebar-footer {{
            position: relative;
            margin-top: 0;
            margin-bottom: 0;
            width: 100%;
            color: {TEXT_DIM};
            font-size: 0.7rem;
            letter-spacing: 0.02em;
            line-height: 1.35;
            opacity: 0.78;
            padding: 0.85rem 0 0.9rem 0;
            border-top: 1px solid {BORDER};
            background: transparent;
            box-shadow: 0 -10px 18px rgba(22,27,34,0.42);
        }}
        @media (max-width: 640px) {{
            .sidebar-footer {{
                font-size: 0.68rem;
                padding-bottom: 0.75rem;
            }}
        }}
        .ind-section {{
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: {TEXT_MUTED};
            margin: 1.65rem 0 0.75rem 0;
            padding-bottom: 0.48rem;
            border-bottom: 1px solid {BORDER};
        }}
        .ind-kpi {{
            background: linear-gradient(145deg, {BG_CARD} 0%, {BG_ELEVATED} 100%);
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            min-height: 104px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.25);
            transition: transform 140ms ease, border-color 140ms ease, box-shadow 140ms ease;
        }}
        .ind-kpi:hover {{
            transform: translateY(-1px);
            border-color: {ACCENT}55;
            box-shadow: 0 8px 20px rgba(0,0,0,0.28);
        }}
        .ind-kpi-value {{
            font-size: 1.65rem;
            font-weight: 700;
            line-height: 1.2;
        }}
        .ind-kpi-label {{
            font-size: 0.78rem;
            color: {TEXT_MUTED};
            margin-top: 0.25rem;
        }}
        .ind-kpi-sub {{
            font-size: 0.72rem;
            color: {TEXT_DIM};
            margin-top: 0.15rem;
        }}
        .ind-card {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem 1.25rem;
            box-shadow: 0 8px 18px rgba(0,0,0,0.18);
        }}
        div[data-testid="stVerticalBlock"] {{
            gap: 0.8rem;
        }}
        .badge {{
            display: inline-block;
            padding: 0.2rem 0.65rem;
            border-radius: 6px;
            font-size: 0.75rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }}
        .banner-critical {{
            background: {CRITICAL}18;
            border: 1px solid {CRITICAL}55;
            color: {CRITICAL};
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            font-weight: 600;
        }}
        .banner-high {{
            background: {DANGER}18;
            border: 1px solid {DANGER}55;
            color: {DANGER};
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }}
        .banner-ok {{
            background: {SUCCESS}15;
            border: 1px solid {SUCCESS}44;
            color: {SUCCESS};
            padding: 0.75rem 1rem;
            border-radius: 8px;
            margin: 0.5rem 0;
        }}
        div[data-testid="stMetric"] {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.75rem 1rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
