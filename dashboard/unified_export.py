"""Unified full-report export for Main Analytics."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Optional

import pandas as pd

from dashboard.exports import REPORTLAB_OK, build_pdf_report, dataframe_csv_bytes, dataframe_excel_bytes
from dashboard.risk_scoring import ensure_unified_decisions


def _summary_rows(df: pd.DataFrame, selected_idx: Optional[int] = None) -> List[Dict[str, Any]]:
    df = ensure_unified_decisions(df)
    rows = [{"Metric": "Total batches", "Value": len(df)}]
    if "defect_prob" in df.columns:
        rows.append({"Metric": "Avg defect probability", "Value": f"{df['defect_prob'].mean():.1%}"})
    if "risk_level" in df.columns:
        rows.append({"Metric": "Critical batches", "Value": int((df["risk_level"] == "CRITICAL").sum())})
    if "recommendation" in df.columns:
        for rec in ["PROCEED", "MONITOR", "HOLD", "STOP"]:
            rows.append({"Metric": f"Recommendation {rec}", "Value": int((df["recommendation"] == rec).sum())})
    if selected_idx is not None and 0 <= selected_idx < len(df):
        row = df.iloc[selected_idx]
        rows.append({"Metric": "Selected batch index", "Value": selected_idx})
        rows.append({"Metric": "Selected risk level", "Value": str(row.get("risk_level", ""))})
        rows.append({"Metric": "Selected recommendation", "Value": str(row.get("recommendation", ""))})
    return rows


def export_csv_bytes(df: pd.DataFrame) -> bytes:
    df = ensure_unified_decisions(df)
    return dataframe_csv_bytes(df)


def export_excel_bytes(df: pd.DataFrame, selected_idx: Optional[int] = None) -> bytes:
    df = ensure_unified_decisions(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Processed Results")
        pd.DataFrame(_summary_rows(df, selected_idx)).to_excel(writer, index=False, sheet_name="Fleet Summary")
        out_cols = [
            c
            for c in [
                "defect_prob",
                "anomaly_score",
                "anomaly_severity",
                "risk_level",
                "recommendation",
                "final_risk_score",
                "qa_summary",
                "cluster",
            ]
            if c in df.columns
        ]
        if out_cols:
            df[out_cols].to_excel(writer, index=False, sheet_name="Risk & Anomaly")
        if selected_idx is not None and 0 <= selected_idx < len(df):
            df.iloc[[selected_idx]].T.reset_index().to_excel(writer, index=False, sheet_name="Selected Batch")
        if "anomaly_score" in df.columns:
            df.nlargest(50, "anomaly_score")[out_cols].to_excel(writer, index=False, sheet_name="Top Anomalies")
    return buf.getvalue()


def export_pdf_bytes(df: pd.DataFrame, selected_idx: Optional[int] = None) -> bytes:
    df = ensure_unified_decisions(df)
    if not REPORTLAB_OK:
        raise ImportError("reportlab required for PDF export")
    sections: List[Dict[str, Any]] = [
        {"heading": "Fleet summary", "body": "\n".join(f"{r['Metric']}: {r['Value']}" for r in _summary_rows(df, selected_idx))},
    ]
    out_cols = [c for c in ["defect_prob", "anomaly_score", "risk_level", "recommendation"] if c in df.columns]
    if out_cols:
        sections.append({"heading": "Processed results (sample)", "table": df[out_cols].head(40)})
    if "anomaly_score" in df.columns and out_cols:
        sections.append(
            {
                "heading": "Top anomalies",
                "table": df.nlargest(25, "anomaly_score")[out_cols].reset_index(drop=True),
            }
        )
    if selected_idx is not None and 0 <= selected_idx < len(df):
        row = df.iloc[selected_idx]
        sections.append(
            {
                "heading": f"Selected batch {selected_idx}",
                "body": str(row.get("qa_summary", "No engineering report.")),
            }
        )
        if "risk_factors" in row.index:
            sections.append({"heading": "Risk factors", "body": str(row.get("risk_factors", ""))})
    return build_pdf_report("Casting AI — Full Quality Report", sections)
