"""Anomaly detection report page."""

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.components import help_block, page_header, safe_render, section, warning_banner
from dashboard.exports import render_page_exports
from dashboard.theme import SEVERITY_COLORS
from dashboard.utils.data_validation import safe_dataframe, safe_for_plotting
from dashboard.utils.debug import log_df_info


def render(df: pd.DataFrame):
    df = safe_for_plotting(safe_dataframe(df))
    log_df_info("anomaly", df)
    if len(df) == 0:
        st.warning("No anomaly data available.")
        return
    _render_anomaly(df)


def _render_anomaly(df: pd.DataFrame):
    page_header("Anomaly Report", "Severity filtering, dangerous batches, and root-cause indicators")

    help_block(
        "Anomaly score",
        "Isolation Forest compares each batch to normal historical variation. "
        "High scores mean the entire process fingerprint is unusual — not just one sensor.",
    )

    if "anomaly_severity" in df.columns:
        crit_count = int((df["anomaly_severity"] == "CRITICAL").sum())
        if crit_count > 0:
            proceed_crit = int(
                ((df["anomaly_severity"] == "CRITICAL") & (df["recommendation"] == "PROCEED")).sum()
            ) if "recommendation" in df.columns else 0
            if proceed_crit > 0:
                warning_banner(
                    f"{proceed_crit} CRITICAL anomaly batch(es) still marked PROCEED — review unified risk scoring.",
                    "CRITICAL",
                )
            else:
                warning_banner(f"{crit_count} CRITICAL anomaly batches require engineering review.", "CRITICAL")

    sev_options = ["ALL"] + [
        s
        for s in ["CRITICAL", "HIGH RISK", "MEDIUM RISK", "LOW RISK", "NORMAL"]
        if "anomaly_severity" in df.columns and s in df["anomaly_severity"].astype(str).unique()
    ]
    sel_sev = st.selectbox("Filter by severity", sev_options)
    filtered = df if sel_sev == "ALL" else df[df["anomaly_severity"].astype(str) == sel_sev]
    st.caption(f"Showing {len(filtered):,} batches")

    try:
        render_page_exports(
            filtered,
            "anomaly",
            "Anomaly Audit Report",
            pdf_sections=[
                {"heading": "Summary", "body": f"Filtered batches: {len(filtered):,}"},
                {
                    "heading": "Top anomalies",
                    "table": filtered.nlargest(25, "anomaly_score")
                    if "anomaly_score" in filtered.columns
                    else filtered.head(25),
                },
            ],
        )
    except Exception as exc:
        st.caption(f"Exports: {exc}")

    c1, c2 = st.columns([1, 1])
    with c1:
        if "anomaly_severity" in df.columns:
            sev_c = df["anomaly_severity"].value_counts()
            order = [s for s in ["NORMAL", "LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL"] if s in sev_c.index]
            if not order:
                order = list(sev_c.index.astype(str))
            safe_render(
                "Severity distribution",
                charts.fig_horizontal_bar,
                order,
                [int(sev_c[o]) for o in order],
                [SEVERITY_COLORS.get(o, "#888") for o in order],
                "Severity distribution",
            )
    with c2:
        safe_render("Anomaly scatter", charts.fig_anomaly_scatter, df)

    section("Top dangerous batches")
    show_cols = [
        c
        for c in [
            "anomaly_score",
            "anomaly_severity",
            "defect_prob",
            "final_risk_score",
            "risk_level",
            "recommendation",
            "risk_confidence",
            "cluster",
            "defect",
        ]
        if c in filtered.columns
    ]
    sort_col = "anomaly_score" if "anomaly_score" in filtered.columns else (show_cols[0] if show_cols else None)
    if sort_col:
        top = filtered.sort_values(sort_col, ascending=False).head(30)
        st.dataframe(top[show_cols].reset_index(drop=True) if show_cols else top.reset_index(drop=True), width="stretch", hide_index=True)
    else:
        st.dataframe(filtered.head(30), width="stretch", hide_index=True)
