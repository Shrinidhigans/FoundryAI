"""Overview dashboard page."""

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.components import help_block, page_header, render_kpi, safe_render, section
from dashboard.exports import render_page_exports
from dashboard.pipeline import correlation_feature_importance, get_feature_importance
from dashboard.theme import ACCENT, CRITICAL, DANGER, SUCCESS, WARNING
from dashboard.utils.data_validation import safe_dataframe, safe_for_plotting
from dashboard.utils.debug import log_df_info


def _model_accuracy(df) -> str:
    if "defect" not in df.columns or "defect_pred" not in df.columns:
        return "N/A"
    return f"{(df['defect'] == df['defect_pred']).mean():.1%}"


def _fleet_quality_score(df) -> float:
    parts = []
    if "defect_prob" in df.columns:
        parts.append(1.0 - float(pd.to_numeric(df["defect_prob"], errors="coerce").mean()))
    if "anomaly_score" in df.columns:
        parts.append(1.0 - float(pd.to_numeric(df["anomaly_score"], errors="coerce").mean()))
    if "defect" in df.columns:
        parts.append(1.0 - float(pd.to_numeric(df["defect"], errors="coerce").mean()))
    return float(sum(parts) / len(parts)) if parts else 0.5


def _stability_scores(df) -> dict:
    scores = {}
    if "feat_chemistry_instability" in df.columns:
        inv = 1.0 / (1.0 + float(pd.to_numeric(df["feat_chemistry_instability"], errors="coerce").mean()))
        scores["Chemistry stability"] = min(1.0, inv)
    if "feat_temp_loss" in df.columns:
        scores["Temperature consistency"] = float((pd.to_numeric(df["feat_temp_loss"], errors="coerce") < 60).mean())
    if "anomaly_score" in df.columns:
        scores["Process stability"] = float(1.0 - pd.to_numeric(df["anomaly_score"], errors="coerce").mean())
    if "feat_shrinkage_risk_index" in df.columns:
        scores["Shrinkage control"] = float(
            1.0 / (1.0 + pd.to_numeric(df["feat_shrinkage_risk_index"], errors="coerce").mean())
        )
    return scores or {"Process stability": 0.5}


def render(df: pd.DataFrame):
    df = safe_for_plotting(safe_dataframe(df))
    log_df_info("overview", df)
    _render_overview(df)


def _render_overview(df: pd.DataFrame):
    page_header(
        "Overview Dashboard",
        "Executive quality intelligence across all melting batches",
    )

    total = len(df)
    def_rate = float(pd.to_numeric(df["defect"], errors="coerce").mean()) if "defect" in df.columns else None
    crit = int((df["risk_level"] == "CRITICAL").sum()) if "risk_level" in df.columns else 0
    avg_prob = float(pd.to_numeric(df["defect_prob"], errors="coerce").mean()) if "defect_prob" in df.columns else 0
    anomalies = int(pd.to_numeric(df["anomaly_flag"], errors="coerce").sum()) if "anomaly_flag" in df.columns else 0
    acc = _model_accuracy(df)
    health = _fleet_quality_score(df)

    help_block(
        "Executive summary",
        "KPIs summarize fleet-wide quality. **Critical risk batches** combine high defect "
        "probability, anomaly score, or historically risky clusters — these should not proceed to pour.",
    )

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    kpis = [
        (c1, "Total Batches", f"{total:,}", "", ACCENT),
        (c2, "Defect Rate", f"{def_rate:.1%}" if def_rate is not None else "N/A", "", DANGER if def_rate and def_rate > 0.15 else SUCCESS),
        (c3, "Critical Risk", f"{crit:,}", "HOLD / STOP", CRITICAL),
        (c4, "Avg Defect Prob", f"{avg_prob:.1%}", "ML classifier", WARNING if avg_prob > 0.3 else SUCCESS),
        (c5, "Model Accuracy", acc, "vs actual defect", ACCENT),
        (c6, "Anomalies", f"{anomalies:,}", "Isolation forest", CRITICAL if anomalies > total * 0.15 else ACCENT),
    ]
    for col, label, val, sub, color in kpis:
        with col:
            render_kpi(label, val, sub, color)

    try:
        render_page_exports(
            df,
            "overview",
            "Executive Quality Summary",
            pdf_sections=[
                {"heading": "Fleet KPIs", "body": f"Total batches: {total:,}\nAvg defect prob: {avg_prob:.1%}"},
                {
                    "heading": "Risk distribution",
                    "table": df["risk_level"].value_counts().reset_index(name="Count")
                    if "risk_level" in df.columns
                    else df.head(5),
                },
            ],
        )
    except Exception as exc:
        st.caption(f"Exports unavailable: {exc}")

    st.markdown("---")
    section("Quality Health")
    q1, q2, q3 = st.columns([1, 1, 1])
    with q1:
        safe_render("Fleet quality gauge", charts.fig_fleet_quality_gauge, health)
    with q2:
        safe_render("Risk distribution", charts.fig_risk_distribution, df)
    with q3:
        safe_render("Anomaly severity", charts.fig_severity_donut, df)

    q4, q5 = st.columns(2)
    with q4:
        safe_render("Defect trend", charts.fig_defect_trend, df)
    with q5:
        safe_render("Stacked risk", charts.fig_stacked_risk, df)

    section("Defect Insights")
    help_block(
        "Defect-driving parameters",
        "Parameters with highest deviation between healthy and defective batches "
        "highlight where process engineers should focus.",
    )
    safe_render("Defect drivers", charts.fig_defect_driving, df)
    fi = get_feature_importance() or correlation_feature_importance(df)
    if fi:
        names, imp = fi
        safe_render("Feature importance", charts.fig_feature_importance, names, imp)
    else:
        safe_render("Feature importance", charts.fig_feature_importance_from_df, df)

    section("Batch Distribution")
    b1, b2, b3 = st.columns(3)
    with b1:
        if "defect_prob" in df.columns:
            safe_render(
                "Defect probability histogram",
                charts.fig_histogram,
                pd.to_numeric(df["defect_prob"], errors="coerce"),
                "Defect Probability Distribution",
                0.5,
            )
    with b2:
        if "defect" in df.columns:
            d = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int)
            safe_render(
                "Quality split",
                charts.fig_horizontal_bar,
                ["Healthy", "Defective"],
                [int((d == 0).sum()), int((d == 1).sum())],
                [SUCCESS, DANGER],
                "Batch Quality Split",
            )
    with b3:
        safe_render("Cluster distribution", charts.fig_cluster_donut, df)

    b4, b5 = st.columns(2)
    with b4:
        ce_col = "feat_ce_calculated" if "feat_ce_calculated" in df.columns else "ce"
        if ce_col in df.columns:
            safe_render(f"{ce_col} distribution", charts.fig_histogram, pd.to_numeric(df[ce_col], errors="coerce"), f"{ce_col} Distribution")
    with b5:
        if "feat_temp_loss" in df.columns:
            safe_render("Temp loss", charts.fig_histogram, pd.to_numeric(df["feat_temp_loss"], errors="coerce"), "Pouring Temperature Loss (°C)")

    section("Process Intelligence")
    scores = _stability_scores(df)
    p1, p2 = st.columns([1, 1])
    with p1:
        safe_render("Stability scores", charts.fig_process_stability_bars, scores)
    with p2:
        safe_render("Chemistry radar", charts.fig_chemistry_radar, df)
