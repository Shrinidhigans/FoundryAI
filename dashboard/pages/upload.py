"""Batch upload and live prediction page."""

import pandas as pd
import streamlit as st
from streamlit import session_state as ss

from dashboard.components import page_header, section
from dashboard.exports import render_page_exports
from dashboard.pipeline import (
    InferenceSchemaError,
    inspect_upload_file,
    inspect_upload_for_template,
    master_template_csv_bytes,
    master_template_dataframe,
    master_template_excel_bytes,
    run_full_pipeline,
)
from dashboard.risk_scoring import DECISION_LOGIC_VERSION, ensure_unified_decisions


def _render_validation(report) -> None:
    section("Column validation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Uploaded columns", report.uploaded_columns)
    c2.metric("Expected ML features", report.expected_columns)
    c3.metric("Matched", len(report.matched_columns))

    if not report.will_auto_fill:
        st.success("All required ML feature columns are present in your file.")
    elif report.missing_columns:
        st.warning(
            "Some ML features were missing and will be **auto-filled** from training medians:\n\n"
            + ", ".join(f"`{c}`" for c in report.missing_columns[:30])
            + (" …" if len(report.missing_columns) > 30 else "")
        )
    else:
        st.info("Upload accepted. Missing values will be imputed during prediction.")

    if report.extra_columns:
        with st.expander(f"Extra columns ignored ({len(report.extra_columns)})", expanded=False):
            st.caption(", ".join(report.extra_columns[:60]))


def render():
    try:
        _render_upload()
    except Exception as exc:
        st.error(f"Upload page error: {exc}")


def _render_upload():
    page_header("Batch Upload & Predict", "Upload Excel/CSV — full AI pipeline runs automatically")

    section("Sample templates")
    tpl = master_template_dataframe()
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download Sample Excel Template",
            master_template_excel_bytes(),
            "melting_cleaned_template.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with c2:
        st.download_button(
            "Download Sample CSV Template",
            master_template_csv_bytes(),
            "melting_cleaned_template.csv",
            "text/csv",
        )
    st.caption("Use this template format for successful AI analysis.")

    uploaded = st.file_uploader("Melting batch file", type=["xlsx", "xls", "csv"], key="batch_uploader")
    if uploaded is None:
        st.info("Upload a melting log file (Excel or CSV) to run classification, clustering, and anomaly detection.")
        return

    file_bytes = uploaded.read()
    tpl_check, _ = inspect_upload_for_template(file_bytes, uploaded.name)
    pre_report, df_norm = inspect_upload_file(file_bytes, uploaded.name)
    with st.expander("Template validation debug (temporary)", expanded=False):
        dbg = tpl_check.get("debug", {})
        st.markdown("**Normalized template columns**")
        st.code(", ".join(dbg.get("normalized_template_columns", [])[:40]))
        st.markdown("**Normalized uploaded columns**")
        st.code(", ".join(dbg.get("normalized_uploaded_columns", [])[:40]))
        st.markdown("**Matched columns**")
        st.code(", ".join(dbg.get("matched_columns", [])[:40]))
        st.markdown("**True missing columns**")
        st.code(", ".join(dbg.get("true_missing_columns", [])[:40]) or "(none)")
        if tpl_check.get("extra_columns"):
            st.markdown("**Extra columns (allowed)**")
            st.code(", ".join(tpl_check["extra_columns"][:40]))
    if not tpl_check["matches_template"]:
        st.error("Uploaded file format does not match required casting template.")
        if tpl_check["missing_columns"]:
            st.markdown("**Missing required columns**")
            st.code(", ".join(tpl_check["missing_columns"][:30]))
        return
    _render_validation(pre_report)

    with st.spinner("Running industrial AI pipeline…"):
        try:
            df = ensure_unified_decisions(run_full_pipeline(file_bytes, uploaded.name))
        except InferenceSchemaError as e:
            st.error("Uploaded data could not be aligned with the trained ML feature schema.")
            report = getattr(e, "report", None)
            st.markdown(f"**Preprocessing stage failed:** `{getattr(e, 'stage', 'inference')}`")
            if report is not None and (report.auto_filled_columns or report.missing_columns):
                st.markdown("**Missing features**")
                missing = report.auto_filled_columns or report.missing_columns
                st.code(", ".join(missing[:60]))
            st.info("Use the sample template and keep all required process and engineered feature inputs available.")
            with st.expander("Technical details"):
                st.exception(e)
            return
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            with st.expander("Technical details"):
                st.exception(e)
            return

    ss["current_data"] = df
    ss["_dashboard_prepared"] = True
    ss["_decision_logic_version"] = DECISION_LOGIC_VERSION
    st.success(f"Pipeline complete — {len(df):,} batches processed.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Batches", len(df))
    if "defect" in df.columns:
        c2.metric("Actual defective", int(pd.to_numeric(df["defect"], errors="coerce").sum()))
    if "defect_pred" in df.columns:
        c3.metric("Predicted defective", int(pd.to_numeric(df["defect_pred"], errors="coerce").sum()))
    if "anomaly_flag" in df.columns:
        c4.metric("Anomalies", int(pd.to_numeric(df["anomaly_flag"], errors="coerce").sum()))

    section("Risk summary")
    if "risk_level" in df.columns and "recommendation" in df.columns:
        breakdown = df.groupby(["risk_level", "recommendation"]).size().reset_index(name="Batches")
        st.dataframe(breakdown, width="stretch", hide_index=True)

    out_cols = [
        c
        for c in [
            "defect",
            "defect_pred",
            "defect_prob",
            "anomaly_score",
            "anomaly_severity",
            "cluster",
            "pca_pc1",
            "pca_pc2",
            "risk_level",
            "recommendation",
            "final_risk_score",
            "risk_confidence",
            "qa_summary",
        ]
        if c in df.columns
    ]

    section("Exports")
    render_page_exports(
        df[out_cols] if out_cols else df,
        "upload",
        "AI Inference Report",
        pdf_sections=[
            {
                "heading": "Summary",
                "body": (
                    f"Batches: {len(df)} · "
                    f"Avg defect prob: {pd.to_numeric(df.get('defect_prob', 0), errors='coerce').mean():.1%}"
                ),
            },
            {"heading": "Predictions sample", "table": df[out_cols].head(30) if out_cols else df.head(30)},
        ],
    )

    st.dataframe(df[out_cols].head(50) if out_cols else df.head(50), width="stretch")
