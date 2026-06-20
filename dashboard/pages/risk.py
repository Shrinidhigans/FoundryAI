"""Engineering guidance and recommendations."""

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.column_mapping import process_feature_columns
from dashboard.components import help_block, page_header, render_kpi, safe_render, section
from dashboard.exports import render_page_exports
from dashboard.pipeline import correlation_feature_importance, get_feature_importance
from dashboard.risk_scoring import ensure_unified_decisions
from dashboard.theme import REC_COLORS
from dashboard.utils.data_validation import safe_dataframe, safe_for_plotting
from dashboard.utils.debug import log_df_info


def _engineering_hints(df: pd.DataFrame) -> list:
    hints = []
    if "feat_temp_loss" in df.columns and pd.to_numeric(df["feat_temp_loss"], errors="coerce").mean() > 70:
        hints.append("Reduce pouring temperature by ~20°C or pre-heat ladle to cut heat loss and shrinkage risk.")
    s_col = "s_" if "s_" in df.columns else ("s" if "s" in df.columns else None)
    if s_col and pd.to_numeric(df[s_col], errors="coerce").mean() > 0.015:
        hints.append("Elevated sulfur — increase desulfurisation and verify Mn/S ≥ 3 before Mg treatment.")
    ce_col = "feat_ce_calculated" if "feat_ce_calculated" in df.columns else ("ce" if "ce" in df.columns else None)
    if ce_col and (pd.to_numeric(df[ce_col], errors="coerce") < 4.2).mean() > 0.2:
        hints.append("Hypoeutectic CE trend — increase carbon/silicon toward eutectic (CE ≥ 4.20).")
    mg_col = "mg_recovery_" if "mg_recovery_" in df.columns else ("mg_recovery" if "mg_recovery" in df.columns else None)
    if mg_col and (pd.to_numeric(df[mg_col], errors="coerce") < 0.40).mean() > 0.1:
        hints.append("Low Mg recovery — check FSM wire speed, treatment timing, and pouring temperature.")
    if not hints:
        hints.append("No fleet-wide instability flags — continue standard SPC monitoring.")
    return hints


def render(df: pd.DataFrame):
    df = safe_for_plotting(safe_dataframe(ensure_unified_decisions(df)))
    log_df_info("risk", df)
    if len(df) == 0:
        st.warning("No risk data available.")
        return
    _render_risk(df)


def _render_risk(df: pd.DataFrame):
    page_header("Engineering Guidance", "Process corrections, chemistry adjustments, and instability indicators")

    help_block(
        "Recommendations",
        "**STOP** = do not pour; **HOLD** = quarantine for metallurgist review; "
        "**MONITOR** = pour with extra checks; **PROCEED** = within normal limits.",
    )

    if "recommendation" in df.columns:
        rec_c = df["recommendation"].value_counts()
        cols = st.columns(4)
        for col, rec in zip(cols, ["PROCEED", "MONITOR", "HOLD", "STOP"]):
            with col:
                render_kpi(rec, str(rec_c.get(rec, 0)), "batches", REC_COLORS.get(rec, "#888"))

    section("Fleet engineering guidance")
    for h in _engineering_hints(df):
        st.markdown(f"- {h}")

    risk_options = ["ALL", "CRITICAL", "HIGH", "MEDIUM", "LOW"]
    sel = st.selectbox("Filter by risk level", risk_options)
    filtered = df if sel == "ALL" else df[df["risk_level"] == sel] if "risk_level" in df.columns else df

    try:
        render_page_exports(
            filtered,
            "risk",
            "Engineering Guidance Report",
            pdf_sections=[
                {"heading": "Fleet guidance", "body": "\n".join(_engineering_hints(df))},
                {"heading": "Filtered batches", "table": filtered.head(25)},
            ],
        )
    except Exception as exc:
        st.caption(f"Exports: {exc}")

    section("Feature importance")
    fi = get_feature_importance() or correlation_feature_importance(df)
    if fi:
        names, imp = fi
        safe_render("Feature importance", charts.fig_feature_importance, names, imp)
    else:
        safe_render("Feature importance fallback", charts.fig_feature_importance_from_df, df)
    st.caption("Grouped colors: chemistry (blue), thermal (orange), engineered (green), other (gray).")

    section("Chemistry correlation heatmap")
    chem_cols = process_feature_columns(df)
    safe_render(
        "Correlation heatmap",
        charts.fig_heatmap_from_df,
        df,
        chem_cols if len(chem_cols) >= 2 else None,
        "Parameter correlations (chemistry & process)",
    )

    section("High / critical QA reports")
    if "qa_summary" in df.columns and "risk_level" in filtered.columns:
        hr = filtered[filtered["risk_level"].isin(["HIGH", "CRITICAL"])].head(10)
        if len(hr) == 0:
            st.success("No high-risk batches in current filter.")
        else:
            for _, row in hr.iterrows():
                with st.expander(
                    f"Batch — {row.get('risk_level')} | Defect {float(row.get('defect_prob', 0)):.1%} | {row.get('recommendation')}"
                ):
                    st.code(str(row.get("qa_summary", "")), language=None)
    else:
        st.caption("QA narratives appear when risk enrichment completes on load.")
