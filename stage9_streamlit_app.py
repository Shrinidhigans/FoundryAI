"""
=============================================================================
STAGE 9 — INDUSTRIAL CASTING QUALITY MONITORING DASHBOARD
=============================================================================
Run with:  streamlit run stage9_streamlit_app.py

Primary UX: unified Main Analytics + ML Performance.
=============================================================================
"""

import warnings

import streamlit as st
from streamlit import session_state as ss

warnings.filterwarnings("ignore")

from dashboard.pipeline import models_ready, prepare_dashboard_dataframe
from dashboard.risk_scoring import DECISION_LOGIC_VERSION, dataframe_needs_decision_refresh, ensure_unified_decisions
from dashboard.theme import inject_global_css
from dashboard.pages import main_analytics, ml_performance
from dashboard.components import status_indicator

PAGES = {
    "main": ("Main Analytics", main_analytics.render),
    "ml_perf": ("ML Performance", ml_performance.render),
}

st.set_page_config(
    page_title="Casting AI | Quality Monitor",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_global_css()


def _ensure_prepared_data():
    """Session state always holds a prepared dataframe."""
    if "current_data" not in ss:
        return
    if (
        ss.get("_dashboard_prepared")
        and ss.get("_decision_logic_version") == DECISION_LOGIC_VERSION
        and not dataframe_needs_decision_refresh(ss["current_data"])
    ):
        return
    ss["current_data"] = ensure_unified_decisions(prepare_dashboard_dataframe(ss["current_data"]))
    ss["_dashboard_prepared"] = True
    ss["_decision_logic_version"] = DECISION_LOGIC_VERSION


def _sidebar() -> str:
    with st.sidebar:
        st.markdown("# Casting AI")
        st.caption("Industrial quality monitoring")
        st.markdown("---")

        labels = {k: v[0] for k, v in PAGES.items()}
        choice = st.radio(
            "Navigation",
            list(labels.keys()),
            format_func=lambda k: labels[k],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown("**System status**")
        status_indicator("ML models", models_ready())
        status_indicator(
            "Dataset",
            "current_data" in ss,
            f"{len(ss['current_data']):,} batches" if "current_data" in ss else "",
        )
        status_indicator("Pipeline", True, "Stages 1–5 ready")

        if "current_data" in ss:
            df = ss["current_data"]
            st.markdown("---")
            st.caption(f"Batches: {len(df):,}")
            fname = ss.get("upload_filename")
            if fname:
                st.caption(f"File: {fname}")
            if "defect" in df.columns:
                st.caption(f"Defect rate: {df['defect'].mean():.1%}")
            if "recommendation" in df.columns:
                hold = (df["recommendation"].isin(["HOLD", "STOP"])).sum()
                st.caption(f"HOLD/STOP: {hold:,}")

        st.markdown('<div class="sidebar-footer">Developed by Shrinidhi Ganesan</div>', unsafe_allow_html=True)

    return choice


def main():
    page_key = _sidebar()

    if page_key == "ml_perf":
        ml_performance.render()
        return

    _ensure_prepared_data()
    main_analytics.render()


if __name__ == "__main__":
    main()
