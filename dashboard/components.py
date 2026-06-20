"""Reusable Streamlit UI components."""

import streamlit as st

from dashboard.theme import REC_COLORS, TEXT_MUTED, TEXT_PRIMARY
from dashboard.risk_scoring import normalize_recommendation, normalize_risk_level

_DIV = "div"


def page_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def section(title: str):
    st.markdown(
        f"<{_DIV} class=\"ind-section\">{title}</{_DIV}>",
        unsafe_allow_html=True,
    )


def help_block(title: str, body: str):
    with st.expander(f"What does this mean? — {title}", expanded=False):
        st.markdown(body)


def render_kpi(label: str, value: str, sub: str = "", color: str = TEXT_PRIMARY):
    sub_line = (
        f"<{_DIV} class=\"ind-kpi-sub\">{sub}</{_DIV}>" if sub else ""
    )
    st.markdown(
        f"<{_DIV} class=\"ind-kpi\">"
        f"<{_DIV} class=\"ind-kpi-value\" style=\"color:{color}\">{value}</{_DIV}>"
        f"<{_DIV} class=\"ind-kpi-label\">{label}</{_DIV}>"
        f"{sub_line}"
        f"</{_DIV}>",
        unsafe_allow_html=True,
    )


def rec_badge(rec: str):
    rec = normalize_recommendation(rec)
    c = REC_COLORS.get(rec, TEXT_MUTED)
    st.markdown(
        f"<span class=\"badge\" style=\"background:{c}22;color:{c};"
        f"border:1px solid {c}66\">{rec}</span>",
        unsafe_allow_html=True,
    )


def warning_banner(message: str, level: str = "high"):
    lvl = str(level).upper()
    if lvl == "CRITICAL":
        cls = "banner-critical"
    elif lvl in ("HIGH", "HIGH RISK"):
        cls = "banner-high"
    else:
        cls = "banner-ok"
    st.markdown(f"<{_DIV} class=\"{cls}\">{message}</{_DIV}>", unsafe_allow_html=True)


def show_chart(fig, label: str = "Chart"):
    """Render Plotly figure; never crash the page."""
    from dashboard.charts import PLOTLY_AVAILABLE
    from dashboard.theme import plotly_config

    if fig is None:
        if not PLOTLY_AVAILABLE:
            st.caption(f"{label}: install Plotly with `pip install plotly` for interactive charts.")
        return
    try:
        st.plotly_chart(fig, width="stretch", config=plotly_config())
    except Exception as exc:
        st.warning(f"{label} could not be displayed ({exc}).")


def safe_render(label: str, builder, *args, **kwargs):
    """Run a chart builder with isolated error handling."""
    try:
        show_chart(builder(*args, **kwargs), label=label)
    except Exception as exc:
        st.warning(f"{label}: {exc}")


def empty_state(title: str, detail: str, icon: str = "📊"):
    st.markdown(
        f"<{_DIV} class=\"ind-card\" style=\"text-align:center;padding:2.5rem;\">"
        f"<{_DIV} style=\"font-size:2rem;\">{icon}</{_DIV}>"
        f"<{_DIV} style=\"color:{TEXT_PRIMARY};font-weight:600;margin-top:0.5rem;\">"
        f"{title}</{_DIV}>"
        f"<{_DIV} style=\"color:{TEXT_MUTED};font-size:0.9rem;margin-top:0.5rem;\">"
        f"{detail}</{_DIV}>"
        f"</{_DIV}>",
        unsafe_allow_html=True,
    )


def risk_panel(risk_level, recommendation, final_score, confidence, factors=""):
    risk_level = normalize_risk_level(risk_level)
    recommendation = normalize_recommendation(recommendation)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Final Risk Score", f"{final_score:.0f}/100")
    c2.metric("Recommendation", recommendation)
    c3.metric("Risk Level", risk_level)
    c4.metric("Confidence", f"{confidence:.0%}")
    if factors:
        warning_banner(f"Active risk factors: {factors}", level=risk_level)


def status_indicator(label: str, ok: bool, detail: str = ""):
    dot = "status-ok" if ok else "status-err"
    cap = detail or ("Ready" if ok else "Unavailable")
    st.markdown(
        f"<{_DIV} style=\"margin:0.25rem 0;font-size:0.8rem;color:{TEXT_MUTED}\">"
        f"<span class=\"status-dot {dot}\"></span>{label}: {cap}"
        f"</{_DIV}>",
        unsafe_allow_html=True,
    )
