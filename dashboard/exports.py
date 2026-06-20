"""PDF / CSV / Excel export utilities for dashboard pages."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from dashboard.risk_scoring import ensure_unified_decisions

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _export_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out


def dataframe_csv_bytes(df: pd.DataFrame) -> bytes:
    df = ensure_unified_decisions(df)
    return _export_safe_df(df).to_csv(index=False).encode("utf-8")


def dataframe_excel_bytes(df: pd.DataFrame, sheet_name: str = "Data") -> bytes:
    df = ensure_unified_decisions(df)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        _export_safe_df(df).to_excel(writer, index=False, sheet_name=sheet_name)
    return buf.getvalue()


def build_pdf_report(
    title: str,
    sections: List[Dict[str, Any]],
    subtitle: str = "Casting AI Quality Monitor",
) -> bytes:
    """
    Build a simple branded PDF from text sections and optional tables.

    sections: [{"heading": str, "body": str}, {"heading": str, "table": pd.DataFrame}, ...]
    """
    if not REPORTLAB_OK:
        raise ImportError("reportlab is required for PDF export. Install with: pip install reportlab")

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter, topMargin=0.6 * inch, bottomMargin=0.6 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CastTitle",
        parent=styles["Heading1"],
        fontSize=16,
        textColor=colors.HexColor("#00a3e0"),
        spaceAfter=8,
    )
    h_style = ParagraphStyle("CastH", parent=styles["Heading2"], fontSize=12, spaceBefore=10, spaceAfter=4)
    body_style = ParagraphStyle("CastBody", parent=styles["Normal"], fontSize=9, leading=12)

    story = [
        Paragraph(title, title_style),
        Paragraph(f"{subtitle} · Generated {_timestamp()}", body_style),
        Spacer(1, 0.15 * inch),
    ]

    for sec in sections:
        if sec.get("heading"):
            story.append(Paragraph(str(sec["heading"]), h_style))
        if sec.get("body"):
            for line in str(sec["body"]).split("\n"):
                if line.strip():
                    story.append(Paragraph(line.strip(), body_style))
        if sec.get("table") is not None and len(sec["table"]) > 0:
            tbl_df = sec["table"].head(40)
            data = [list(tbl_df.columns)] + tbl_df.astype(str).values.tolist()
            t = Table(data, repeatRows=1)
            t.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1f2b")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                        ("FONTSIZE", (0, 0), (-1, -1), 7),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
                    ]
                )
            )
            story.append(t)
        story.append(Spacer(1, 0.08 * inch))

    doc.build(story)
    return buf.getvalue()


def overview_executive_pdf(df: pd.DataFrame) -> bytes:
    lines = [
        f"Total batches: {len(df):,}",
        f"Defect rate: {df['defect'].mean():.1%}" if "defect" in df.columns else "Defect rate: N/A",
        f"Avg defect probability: {df['defect_prob'].mean():.1%}" if "defect_prob" in df.columns else "",
    ]
    if "risk_level" in df.columns:
        lines.append("Risk breakdown: " + ", ".join(f"{k}={v}" for k, v in df["risk_level"].value_counts().items()))
    kpi = pd.DataFrame({"Metric": lines})
    return build_pdf_report(
        "Executive Quality Summary",
        [
            {"heading": "Fleet KPIs", "body": "\n".join(lines)},
            {"heading": "Risk distribution", "table": df["risk_level"].value_counts().reset_index(name="Count")
             if "risk_level" in df.columns else kpi},
        ],
    )


def single_batch_qa_pdf(row: pd.Series, batch_idx: int) -> bytes:
    body = str(row.get("qa_summary", "No QA summary available."))
    params = pd.DataFrame(
        {
            "Parameter": [k for k in row.index if not str(k).startswith("Unnamed")][:25],
            "Value": [row[k] for k in row.index if not str(k).startswith("Unnamed")][:25],
        }
    )
    header = (
        f"Batch index: {batch_idx}\n"
        f"Risk: {row.get('risk_level', 'N/A')} · Recommendation: {row.get('recommendation', 'N/A')}\n"
        f"Defect probability: {float(row.get('defect_prob', 0)):.1%} · Anomaly: {float(row.get('anomaly_score', 0)):.2f}"
    )
    return build_pdf_report(
        "QA Inspection Report",
        [{"heading": "Assessment", "body": header}, {"heading": "Engineering narrative", "body": body}, {"heading": "Parameters", "table": params}],
    )


def render_page_exports(
    df: pd.DataFrame,
    page_slug: str,
    pdf_title: str,
    pdf_sections: Optional[List[Dict[str, Any]]] = None,
    sample_rows: int = 500,
) -> None:
    """Standard CSV / Excel / PDF download row for dashboard pages."""
    import streamlit as st

    sections = pdf_sections or [
        {"heading": "Summary", "body": f"Total batches: {len(df):,}"},
        {"heading": "Sample data", "table": df.head(30)},
    ]
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button(
            "Export CSV",
            dataframe_csv_bytes(df.head(sample_rows)),
            f"{page_slug}_export.csv",
            "text/csv",
            key=f"csv_{page_slug}",
        )
    with c2:
        st.download_button(
            "Export Excel",
            dataframe_excel_bytes(df.head(sample_rows), page_slug.title()),
            f"{page_slug}_export.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"xlsx_{page_slug}",
        )
    with c3:
        if REPORTLAB_OK:
            try:
                pdf = build_pdf_report(pdf_title, sections)
                st.download_button(
                    "Export PDF",
                    pdf,
                    f"{page_slug}_report.pdf",
                    "application/pdf",
                    key=f"pdf_{page_slug}",
                )
            except Exception:
                st.caption("PDF export temporarily unavailable.")
        else:
            st.caption("Install reportlab for PDF: pip install reportlab")


def anomaly_audit_pdf(df: pd.DataFrame, top_n: int = 25) -> bytes:
    cols = [c for c in ["anomaly_score", "anomaly_severity", "defect_prob", "risk_level", "recommendation", "cluster"] if c in df.columns]
    top = df.nlargest(top_n, "anomaly_score")[cols] if "anomaly_score" in df.columns else df[cols].head(top_n)
    return build_pdf_report(
        "Anomaly Audit Report",
        [
            {"heading": "Summary", "body": f"Critical anomalies: {(df['anomaly_severity'] == 'CRITICAL').sum() if 'anomaly_severity' in df.columns else 'N/A'}"},
            {"heading": f"Top {len(top)} batches by anomaly score", "table": top.reset_index(drop=True)},
        ],
    )
