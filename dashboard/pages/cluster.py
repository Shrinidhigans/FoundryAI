"""Cluster analysis page."""

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.components import page_header, render_kpi, section
from dashboard.exports import render_page_exports
from dashboard.theme import ACCENT, CRITICAL, REC_COLORS, SUCCESS, WARNING, plotly_config
from dashboard.utils.data_validation import safe_dataframe, safe_for_plotting
from dashboard.utils.debug import log_df_info


def render(df: pd.DataFrame):
    df = safe_for_plotting(safe_dataframe(df))
    log_df_info("cluster", df)
    if len(df) == 0:
        st.warning("No data for cluster analysis.")
        return
    _render_cluster(df)


def _render_cluster(df: pd.DataFrame):
    page_header("Cluster Analysis", "Interactive casting map colored by business decision")

    section("Cluster analysis")
    st.plotly_chart(
        charts.fig_cluster_analysis(df),
        width="stretch",
        config={**plotly_config(), "scrollZoom": True},
        key="cluster_page_scatter",
    )

    rec_counts = df["recommendation"].astype(str).str.upper().value_counts() if "recommendation" in df.columns else pd.Series(dtype=int)
    risk_counts = df["risk_level"].astype(str).str.upper().value_counts() if "risk_level" in df.columns else pd.Series(dtype=int)
    total_clusters = int(pd.to_numeric(df["cluster"], errors="coerce").nunique()) if "cluster" in df.columns else 0
    largest_cluster = int(df["cluster"].value_counts().max()) if "cluster" in df.columns and len(df) else len(df)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi("Total Castings", f"{len(df):,}", "plotted points", ACCENT)
    with c2:
        render_kpi("Total Clusters", f"{total_clusters:,}", "process groups", ACCENT)
    with c3:
        critical = int(risk_counts.get("CRITICAL", rec_counts.get("STOP", 0)))
        render_kpi("Critical Castings", f"{critical:,}", "STOP / critical", REC_COLORS.get("STOP", CRITICAL))
    with c4:
        high = int(risk_counts.get("HIGH", rec_counts.get("HOLD", 0)))
        render_kpi("High Risk Castings", f"{high:,}", "HOLD / high", REC_COLORS.get("HOLD", WARNING))
    with c5:
        render_kpi("Largest Cluster Size", f"{largest_cluster:,}", "batches", SUCCESS)

    try:
        render_page_exports(
            df,
            "cluster_analysis",
            "Cluster Analysis",
            pdf_sections=[
                {"heading": "Summary", "body": f"Total castings: {len(df):,}\nTotal clusters: {total_clusters:,}"},
                {"heading": "Sample batches", "table": df.head(25)},
            ],
        )
    except Exception as exc:
        st.caption(f"Exports: {exc}")
