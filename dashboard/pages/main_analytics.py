"""
Unified Main Analytics — single-page industrial casting quality workflow.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st
from streamlit import session_state as ss

from dashboard import charts
from dashboard.column_mapping import chemistry_columns
from dashboard.components import (
    empty_state,
    help_block,
    page_header,
    rec_badge,
    render_kpi,
    risk_panel,
    safe_render,
    section,
    show_chart,
    warning_banner,
)
from dashboard.exports import REPORTLAB_OK
from dashboard.main_glossary import MAIN_TERMS
from dashboard.pages.ml_performance import _fig_correlation_heatmap, _fig_top_defect_parameters
from dashboard.pages.overview import _fleet_quality_score, _stability_scores
from dashboard.pipeline import (
    InferenceSchemaError,
    correlation_feature_importance,
    get_feature_importance,
    inspect_upload_file,
    inspect_upload_for_template,
    load_default_data,
    master_template_csv_bytes,
    master_template_excel_bytes,
    run_full_pipeline,
)
from dashboard.risk_scoring import DECISION_LOGIC_VERSION, ensure_unified_decisions
from dashboard.theme import (
    ACCENT,
    BG_CARD,
    BG_ELEVATED,
    BORDER,
    CRITICAL,
    RISK_COLORS,
    REC_COLORS,
    SEVERITY_COLORS,
    SUCCESS,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
    plotly_config,
)
from dashboard.unified_export import export_csv_bytes, export_excel_bytes, export_pdf_bytes
from dashboard.utils.data_validation import safe_dataframe, safe_for_plotting, safe_probability

BATCH_ID_CANDIDATES = [
    "heat",
    "heat_no",
    "heat_number",
    "batch",
    "batch_id",
    "casting",
    "casting_no",
    "serial",
    "aa",
]

CHART_HELP = {
    "Fleet quality score": (
        "Combines fleet defect probability, risk, and anomaly signals into one quality score. "
        "Metallurgy and QA teams use it to spot whether the loaded casting set is broadly stable. "
        "Higher is better."
    ),
    "Process stability score": (
        "Shows how consistent process signals are across the loaded castings. "
        "QA teams use it to identify drift in melting, chemistry, and thermal behavior. "
        "Higher is better because it means less variation."
    ),
    "Anomaly severity": (
        "Counts castings by how unusual their process pattern looks. "
        "Metallurgy teams use it to find batches that may need review even when defect probability is not high. "
        "Lower severity is better."
    ),
    "Risk level distribution": (
        "Splits all castings into HEALTHY, LOW, MEDIUM, HIGH, and CRITICAL risk buckets. "
        "QA teams use it to confirm the full fleet is accounted for and prioritize reviews. "
        "More HEALTHY/LOW is better; HIGH/CRITICAL need attention."
    ),
    "Cluster Analysis": (
        "Maps castings by process similarity and decision status. "
        "Metallurgy teams use it to detect groups of similar batches and repeated process patterns. "
        "Clusters with fewer high-risk points are better."
    ),
    "Parameter Correlation": (
        "Shows how chemistry, thermal, and engineered parameters move together. "
        "QA teams use it to identify linked process variables and possible defect drivers. "
        "Strong correlations are not automatically good or bad; they indicate relationships to investigate."
    ),
    "Casting Comparison": (
        "Compares the selected casting against healthy fleet behavior. "
        "Metallurgy teams use it to see which process or chemistry signals differ from normal production. "
        "Closer to the healthy fleet profile is generally better."
    ),
}


def _inject_main_layout_css():
    st.markdown(
        f"""
        <style>
        .block-container {{
            max-width: 1500px;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        h2.report-results-title {{
            font-size: 1.85rem;
            font-weight: 800;
            letter-spacing: 0.07em;
            color: {TEXT_PRIMARY};
            margin: 2rem 0 0.5rem 0;
            padding-bottom: 0.65rem;
            border-bottom: 2px solid {BORDER};
        }}
        .report-results-spacer {{
            margin-bottom: 0.85rem;
        }}
        .analytics-section {{
            margin: 2.5rem 0 1.25rem 0;
            padding-bottom: 0.7rem;
            border-bottom: 1px solid {BORDER};
        }}
        .analytics-section.compact {{
            margin-top: 2rem;
        }}
        .analytics-section h2 {{
            font-size: 1.58rem;
            line-height: 1.18;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            color: {TEXT_PRIMARY};
            margin: 0 0 0.5rem 0;
        }}
        .analytics-section p {{
            color: {TEXT_MUTED};
            font-size: 0.9rem;
            margin: 0;
        }}
        div.fleet-kpi-row [data-testid="column"] > div {{
            min-height: 108px;
        }}
        div.fleet-kpi-row [data-testid="column"] .ind-kpi {{
            height: 100%;
            min-height: 108px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 1rem 1.05rem;
            border-radius: 10px;
        }}
        div.fleet-recommendations-block {{
            margin-top: 1.75rem;
            margin-bottom: 2.25rem;
            padding-top: 1.4rem;
            border-top: 1px solid {BORDER};
        }}
        div.fleet-recommendations-block [data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 1.2rem 1.35rem 1rem 1.35rem;
            min-height: 0;
        }}
        div.fleet-recommendations-block [data-testid="stCaptionContainer"] {{
            margin-bottom: 0.75rem;
        }}
        div.fleet-rec-summary-row {{
            margin-bottom: 1.05rem;
        }}
        div.fleet-rec-summary-row [data-testid="column"] > div {{
            min-height: 104px;
        }}
        div.fleet-rec-summary-row [data-testid="column"] .ind-kpi {{
            height: 100%;
            min-height: 104px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            padding: 0.92rem 1rem;
            border-radius: 10px;
        }}
        .fleet-rec-title {{
            color: {TEXT_PRIMARY};
            font-size: 1.2rem;
            font-weight: 800;
            letter-spacing: 0;
            margin-bottom: 0.9rem;
        }}
        .fleet-rec-item {{
            border-top: 1px solid {BORDER};
            padding: 1rem 0 0.85rem 0;
        }}
        .fleet-rec-badge {{
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 0.18rem 0.58rem;
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.04em;
            color: {TEXT_PRIMARY};
        }}
        .fleet-rec-text {{
            color: {TEXT_PRIMARY};
            font-size: 1.16rem;
            line-height: 1.32;
            font-weight: 600;
            margin-top: 0.45rem;
        }}
        .fleet-rec-affected {{
            color: {TEXT_MUTED};
            font-size: 0.82rem;
            margin-top: 0.32rem;
        }}
        div.fleet-chart-grid {{
            margin-top: 0.35rem;
            margin-bottom: 2.5rem;
        }}
        div.fleet-chart-grid [data-testid="column"] {{
            min-width: 0;
        }}
        div.fleet-chart-grid [data-testid="stVerticalBlockBorderWrapper"] {{
            min-height: 470px;
            height: 470px;
            padding: 0.9rem 1rem 0.35rem 1rem;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
        }}
        div.fleet-chart-grid [data-testid="stCaptionContainer"] {{
            min-height: 1.4rem;
            margin-bottom: 0.7rem;
        }}
        div.fleet-chart-grid [data-testid="stPlotlyChart"] {{
            height: 430px !important;
            min-height: 430px !important;
        }}
        div.fleet-chart-grid .js-plotly-plot .plotly .modebar,
        div.cluster-analysis-block .js-plotly-plot .plotly .modebar {{
            top: 10px !important;
            right: 10px !important;
            opacity: 0.62;
        }}
        div.fleet-chart-grid .js-plotly-plot .plotly .modebar:hover,
        div.cluster-analysis-block .js-plotly-plot .plotly .modebar:hover {{
            opacity: 0.95;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background: {BG_CARD};
            border-color: {BORDER} !important;
            border-radius: 10px;
            padding: 0.72rem 0.92rem 0.42rem 0.92rem;
            margin-bottom: 1rem;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stPlotlyChart"] {{
            min-height: 380px;
        }}
        div.comparison-block {{
            margin-top: 0.15rem;
        }}
        div.comparison-block [data-testid="stVerticalBlockBorderWrapper"] {{
            min-height: 480px;
        }}
        div.comparison-block [data-testid="stPlotlyChart"] {{
            min-height: 440px;
        }}
        div.comparison-heatmap [data-testid="stVerticalBlockBorderWrapper"] {{
            min-height: 500px;
        }}
        div.comparison-heatmap [data-testid="stPlotlyChart"] {{
            min-height: 460px;
        }}
        div.cluster-analysis-block {{
            margin-top: 0.15rem;
            margin-bottom: 1.1rem;
        }}
        div.cluster-section-wrap .analytics-section {{
            margin-top: 2rem;
            margin-bottom: 1.25rem;
        }}
        div.cluster-section-wrap .analytics-section h2 {{
            font-size: 1.62rem;
        }}
        div.cluster-analysis-block [data-testid="stVerticalBlockBorderWrapper"] {{
            padding: 1.05rem 1.15rem 0.75rem 1.15rem;
            min-height: 520px;
        }}
        div.cluster-analysis-block [data-testid="stPlotlyChart"] {{
            margin-bottom: 0 !important;
            min-height: 500px;
        }}
        div.cluster-analysis-block [data-testid="stElementContainer"]:has([data-testid="stPlotlyChart"]) {{
            margin-bottom: 0 !important;
        }}
        div.cluster-summary-row {{
            margin-top: 0.4rem;
            margin-bottom: 2.5rem;
        }}
        div.cluster-summary-row [data-testid="column"] > div {{
            min-height: 88px;
        }}
        div.cluster-summary-row .ind-kpi {{
            min-height: 88px;
            padding: 0.72rem 0.85rem;
        }}
        div.cluster-summary-row .ind-kpi-value {{
            font-size: 1.38rem;
        }}
        div.cluster-summary-row .ind-kpi-label {{
            font-size: 0.72rem;
        }}
        div.cluster-summary-row .ind-kpi-sub {{
            font-size: 0.68rem;
        }}
        div.casting-comparison-select {{
            max-width: 520px;
            margin-bottom: 0.25rem;
        }}
        .section-help {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 1rem;
            height: 1rem;
            margin-left: 0.45rem;
            border: 1px solid {BORDER};
            border-radius: 50%;
            color: {TEXT_MUTED};
            font-size: 0.72rem;
            vertical-align: middle;
            cursor: help;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _tune_donut(fig):
    if fig is None:
        return None
    fig.update_layout(
        height=440,
        margin=dict(l=24, r=140, t=64, b=48),
        legend=dict(orientation="v", yanchor="middle", y=0.5, x=1.02, font=dict(size=11)),
        showlegend=True,
    )
    fig.update_traces(textinfo="percent", textposition="outside", textfont_size=11, pull=0.02)
    return fig


def _tune_fleet_donut(fig):
    if fig is None:
        return None
    fig.update_layout(
        height=450,
        margin=dict(l=18, r=18, t=56, b=68),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=10)),
        showlegend=True,
    )
    fig.update_traces(textinfo="label+percent", textposition="inside", textfont_size=11, pull=0)
    return fig


def _tune_horizontal(fig, min_height: int = 380):
    if fig is None:
        return None
    fig.update_layout(
        height=min_height,
        margin=dict(l=110, r=48, t=64, b=48),
    )
    return fig


def _tune_risk_distribution(fig):
    if fig is None:
        return None
    fig.update_layout(
        height=450,
        margin=dict(l=118, r=26, t=58, b=64),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5, font=dict(size=10)),
        yaxis=dict(automargin=True, tickfont=dict(size=12)),
        xaxis=dict(automargin=True, tickfont=dict(size=11), title="Castings"),
    )
    fig.update_traces(width=0.62, textposition="outside", cliponaxis=False, textfont=dict(size=12))
    return fig


def _tune_gauge(fig):
    if fig is None:
        return None
    fig.update_layout(height=450, margin=dict(l=28, r=28, t=64, b=34))
    return fig


def _tune_radar(fig):
    if fig is None:
        return None
    fig.update_layout(
        height=500,
        margin=dict(l=72, r=72, t=72, b=88),
        legend=dict(orientation="h", yanchor="top", y=-0.08, x=0.5, xanchor="center", font=dict(size=11)),
    )
    fig.update_layout(
        polar=dict(
            radialaxis=dict(tickfont=dict(size=10)),
            angularaxis=dict(tickfont=dict(size=11), rotation=0),
        )
    )
    return fig


def _tune_heatmap(fig):
    if fig is None:
        return None
    current_h = fig.layout.height if fig.layout.height else 480
    fig.update_layout(
        height=max(480, int(current_h)),
        margin=dict(l=72, r=32, t=56, b=110),
    )
    return fig


def _show_panel_chart(fig, label: str):
    show_chart(fig, label=label)


def _chart_caption(caption: str, help_key: str | None = None):
    key = help_key or caption
    help_text = CHART_HELP.get(key) or CHART_HELP.get(str(key).capitalize()) or CHART_HELP.get(str(key).title())
    st.caption(caption, help=help_text)


def _border_container():
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def _fleet_chart_slot(caption: str, builder, *args, tune_fn=None, **kwargs):
    with _border_container():
        _chart_caption(caption)
        try:
            fig = builder(*args, **kwargs)
            if tune_fn and fig is not None:
                fig = tune_fn(fig)
            _show_panel_chart(fig, caption)
        except Exception as exc:
            st.warning(f"{caption}: {exc}")


def major_section(title: str, description: str = "", compact: bool = False, help_text: str | None = None):
    cls = "analytics-section compact" if compact else "analytics-section"
    help_icon = ""
    if help_text:
        help_icon = f' <span class="section-help" title="{html.escape(help_text, quote=True)}">i</span>'
    st.markdown(
        f"""
        <div class="{cls}">
          <h2>{title}{help_icon}</h2>
          <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _export_download_buttons(df: pd.DataFrame, idx: int):
    st.download_button(
        "CSV",
        export_csv_bytes(df),
        "casting_ai_full_report.csv",
        "text/csv",
        key="unified_csv",
        width="stretch",
    )
    st.download_button(
        "Excel",
        export_excel_bytes(df, idx),
        "casting_ai_full_report.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="unified_xlsx",
        width="stretch",
    )
    if REPORTLAB_OK:
        try:
            st.download_button(
                "PDF",
                export_pdf_bytes(df, idx),
                "casting_ai_full_report.pdf",
                "application/pdf",
                key="unified_pdf",
                width="stretch",
            )
        except Exception as exc:
            st.caption(f"PDF unavailable: {exc}")
    else:
        st.caption("Install reportlab for PDF export.")


def _report_results_header(df: pd.DataFrame):
    idx = int(ss.get("selected_batch_idx", 0))
    title_col, export_col = st.columns([5, 1.35])
    with title_col:
        st.markdown('<h2 class="report-results-title">REPORT RESULTS</h2>', unsafe_allow_html=True)
    with export_col:
        st.markdown("<div style='margin-top:1.75rem'></div>", unsafe_allow_html=True)
        if hasattr(st, "popover"):
            with st.popover("Export Full Report ▼", width="stretch"):
                _export_download_buttons(df, idx)
        else:
            with st.expander("Export Full Report ▼", expanded=False):
                _export_download_buttons(df, idx)
    st.markdown('<div class="report-results-spacer"></div>', unsafe_allow_html=True)


def _batch_labels(df: pd.DataFrame) -> tuple[list[str], str]:
    for col in BATCH_ID_CANDIDATES:
        if col in df.columns:
            return df[col].astype(str).tolist(), col
    return [f"Batch {i}" for i in range(len(df))], "row_index"


def _first_column_match(df: pd.DataFrame, candidates: list[str]) -> str | None:
    normalized = {str(col).strip().lower().replace("_", " "): col for col in df.columns}
    for candidate in candidates:
        key = candidate.strip().lower().replace("_", " ")
        if key in normalized:
            return normalized[key]
    return None


def _set_selected_batch_idx(idx: int, df: pd.DataFrame, sync_widget: bool = True) -> int:
    idx = max(0, min(int(idx), len(df) - 1)) if len(df) else 0
    ss["selected_batch_idx"] = idx
    if sync_widget:
        ss["main_batch_select"] = idx
    return idx


def _healthy_fleet_df(df: pd.DataFrame) -> pd.DataFrame:
    if "defect" in df.columns:
        mask = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int) == 0
        sub = df[mask]
        if len(sub) > 0:
            return sub
    if "defect_prob" in df.columns:
        mask = pd.to_numeric(df["defect_prob"], errors="coerce") < 0.5
        sub = df[mask]
        if len(sub) > 0:
            return sub
    return df


def _process_stability_score(df: pd.DataFrame) -> float:
    scores = _stability_scores(df)
    return float(sum(scores.values()) / len(scores)) if scores else 0.5


def _dominant_severity(df: pd.DataFrame) -> str:
    if "anomaly_severity" not in df.columns:
        return "N/A"
    vc = df["anomaly_severity"].astype(str).value_counts()
    return str(vc.index[0]) if len(vc) else "NORMAL"


def _recommendation_summary(df: pd.DataFrame) -> str:
    if "recommendation" not in df.columns:
        return "N/A"
    parts = []
    for rec in ["PROCEED", "MONITOR", "HOLD", "STOP"]:
        n = int((df["recommendation"] == rec).sum())
        if n:
            parts.append(f"{rec}: {n}")
    return " · ".join(parts) if parts else "N/A"


def _risk_category_counts(df: pd.DataFrame) -> dict[str, int]:
    return charts.risk_category_counts(df)


def _extract_engineering_recommendations(summary: str) -> list[str]:
    lines = [line.strip() for line in str(summary or "").splitlines()]
    in_section = False
    out: list[str] = []
    for line in lines:
        marker = line.upper()
        if marker == "ENGINEERING RECOMMENDATIONS":
            in_section = True
            continue
        if in_section and marker in {"RISK FACTORS", "FINAL RECOMMENDATION", "WARNINGS", "PROBABLE CAUSES"}:
            break
        if not in_section:
            continue
        clean = line[2:].strip() if line.startswith("- ") else line.strip()
        if clean and clean.lower() != "none triggered.":
            out.append(clean)
    return out


def _fallback_recommendation_text(row: pd.Series, risk: str) -> str | None:
    rec = str(row.get("recommendation", "PROCEED")).strip().upper()
    if risk == "HEALTHY" and rec == "PROCEED":
        return None
    return {
        "STOP": "Stop affected castings and complete engineering review",
        "HOLD": "Hold affected castings for metallurgical review",
        "MONITOR": "Monitor process controls before release",
        "PROCEED": "Continue standard production controls",
    }.get(rec, "Review casting risk drivers")


def _fleet_recommendation_items(df: pd.DataFrame, limit: int = 3) -> list[dict[str, object]]:
    if df.empty:
        return []
    risk_series = charts.risk_category_series(df).reset_index(drop=True)
    severity_rank = {level: i for i, level in enumerate(charts.RISK_DISPLAY_ORDER)}
    grouped: dict[str, dict[str, object]] = {}
    for row_pos, (_, row) in enumerate(df.iterrows()):
        risk = str(risk_series.iloc[row_pos])
        recs = _extract_engineering_recommendations(str(row.get("qa_summary", "")))
        if not recs:
            fallback = _fallback_recommendation_text(row, risk)
            recs = [fallback] if fallback else []
        for rec in set(recs):
            item = grouped.setdefault(rec, {"text": rec, "count": 0, "severity": risk})
            item["count"] = int(item["count"]) + 1
            if severity_rank.get(risk, 0) > severity_rank.get(str(item["severity"]), 0):
                item["severity"] = risk
    return sorted(grouped.values(), key=lambda item: int(item["count"]), reverse=True)[:limit]


def _render_fleet_recommendations_card(df: pd.DataFrame):
    counts = _risk_category_counts(df)
    total = len(df)
    healthy = counts["HEALTHY"]
    risky = max(0, total - healthy)
    most_common = max(counts.items(), key=lambda item: item[1])[0] if total else "N/A"
    items = _fleet_recommendation_items(df)

    st.markdown('<div class="fleet-recommendations-block">', unsafe_allow_html=True)
    with _border_container():
        st.markdown('<div class="fleet-rec-title">Fleet Recommendations</div>', unsafe_allow_html=True)
        summary_items = [
            ("Total Castings", f"{total:,}", "loaded", ACCENT),
            ("Healthy Castings", f"{healthy:,}", "normal bucket", RISK_COLORS.get("HEALTHY", SUCCESS)),
            ("Risky Castings", f"{risky:,}", "needs review", WARNING if risky else SUCCESS),
            (
                "Most Common Risk",
                most_common.title() if most_common != "N/A" else "N/A",
                "largest bucket",
                RISK_COLORS.get(most_common, ACCENT),
            ),
        ]
        st.markdown('<div class="fleet-rec-summary-row">', unsafe_allow_html=True)
        summary_cols = st.columns(4)
        for col, (label, value, sub, color) in zip(summary_cols, summary_items):
            with col:
                render_kpi(label, value, sub, color)
        st.markdown("</div>", unsafe_allow_html=True)

        if not items:
            st.info("No recurring corrective recommendations found across the loaded fleet.")
        for item in items:
            severity = str(item["severity"])
            count = int(item["count"])
            pct = (count / total * 100.0) if total else 0.0
            color = {
                "CRITICAL": CRITICAL,
                "HIGH": "#f0883e",
                "MEDIUM": WARNING,
                "LOW": ACCENT,
                "HEALTHY": SUCCESS,
            }.get(severity, RISK_COLORS.get(severity, ACCENT))
            text = html.escape(str(item["text"]))
            st.markdown(
                f"""
                <div class="fleet-rec-item">
                  <span class="fleet-rec-badge" style="background:{color}24; border:1px solid {color}88; color:{color};">
                    {severity}
                  </span>
                  <div class="fleet-rec-text">{text}</div>
                  <div class="fleet-rec-affected">Affected: {count:,} castings ({pct:.1f}%)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def _centralized_reason_lines(row: pd.Series) -> list[str]:
    factors = str(row.get("risk_factors", "")).strip()
    if factors:
        return [p.strip() for p in factors.split(";") if p.strip()]
    return [
        f"Final recommendation: {row.get('recommendation', 'N/A')}",
        f"Risk level: {row.get('risk_level', 'N/A')}",
    ]


def _healthy_vs_casting_radar(row: pd.Series, df: pd.DataFrame):
    healthy = _healthy_fleet_df(df)
    specs = [
        ("Si", ["si_", "si"]),
        ("Mn", ["mn_", "mn_1", "mn"]),
        ("Mg recovery", ["mg_recovery_", "mg_recovery"]),
        ("Pouring temp", ["pouring_temp"]),
        ("Tapping temp", ["tapping_temp"]),
        ("Chemistry stability", ["feat_chemistry_instability"]),
        ("Shrinkage index", ["feat_shrinkage_risk_index"]),
        ("Gas index", ["feat_gas_risk_index"]),
    ]
    labels, vals_sel, vals_healthy = [], [], []
    for label, cols in specs:
        col = next((c for c in cols if c in df.columns), None)
        if col is None:
            continue
        labels.append(label)
        vals_sel.append(float(pd.to_numeric(row[col], errors="coerce")))
        vals_healthy.append(float(pd.to_numeric(healthy[col], errors="coerce").mean()))
    if not labels:
        return None
    return charts.fig_radar(labels, vals_sel, vals_healthy, name_a="Selected casting", name_b="Healthy fleet avg")


def _section_upload():
    section("Data upload")
    uploaded = st.file_uploader(
        "Upload melting / casting data (Excel or CSV)",
        type=["xlsx", "xls", "csv"],
        key="main_file_uploader",
    )
    with st.expander("Download sample templates", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "Sample Excel",
                master_template_excel_bytes(),
                "melting_cleaned_template.xlsx",
                key="main_tpl_xlsx",
            )
        with c2:
            st.download_button(
                "Sample CSV",
                master_template_csv_bytes(),
                "melting_cleaned_template.csv",
                key="main_tpl_csv",
            )
        st.caption("Use this template format for successful AI analysis.")

    if uploaded is not None:
        raw = uploaded.getvalue()
        tpl_check, _ = inspect_upload_for_template(raw, uploaded.name)
        pre_report, df_norm = inspect_upload_file(raw, uploaded.name)
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
        if pre_report.will_auto_fill and pre_report.missing_columns:
            st.warning(
                "Some columns will be auto-filled from training medians: "
                + ", ".join(f"`{c}`" for c in pre_report.missing_columns[:12])
                + (" …" if len(pre_report.missing_columns) > 12 else "")
            )
        with st.spinner("Running AI pipeline on uploaded file…"):
            try:
                df = run_full_pipeline(raw, uploaded.name)
                ss["current_data"] = ensure_unified_decisions(df)
                ss["_dashboard_prepared"] = True
                ss["_decision_logic_version"] = DECISION_LOGIC_VERSION
                ss["upload_filename"] = uploaded.name
                st.success(f"✓ Loaded **{uploaded.name}** — {len(df):,} castings processed.")
            except InferenceSchemaError as exc:
                st.error("Uploaded data could not be aligned with the trained ML feature schema.")
                report = getattr(exc, "report", None)
                st.markdown(f"**Preprocessing stage failed:** `{getattr(exc, 'stage', 'inference')}`")
                if report is not None and (report.auto_filled_columns or report.missing_columns):
                    st.markdown("**Missing features**")
                    missing = report.auto_filled_columns or report.missing_columns
                    st.code(", ".join(missing[:60]))
                st.info("Use the sample template and rerun the upload with the trained feature schema.")
            except Exception as exc:
                st.error(f"Upload failed: {exc}")


def _section_fleet_overview(df: pd.DataFrame):
    major_section("Fleet Overview", "Whole-file quality, stability, risk, and recommendation summary.")
    health = _fleet_quality_score(df)
    stability = _process_stability_score(df)
    avg_prob = float(pd.to_numeric(df["defect_prob"], errors="coerce").mean()) if "defect_prob" in df.columns else 0
    risk_counts = _risk_category_counts(df)
    crit = risk_counts["CRITICAL"]
    sev = _dominant_severity(df)

    st.markdown('<div class="fleet-kpi-row">', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_kpi("Fleet Quality", f"{health:.0%}", "combined score", SUCCESS if health > 0.7 else WARNING)
    with c2:
        render_kpi("Process Stability", f"{stability:.0%}", "fleet-wide", ACCENT)
    with c3:
        render_kpi("Dominant Severity", sev, "anomaly band", ACCENT)
    with c4:
        render_kpi("Avg Defect Prob", f"{avg_prob:.1%}", "ML estimate", WARNING if avg_prob > 0.3 else SUCCESS)
    with c5:
        render_kpi("Critical Batches", f"{crit:,}", "need review", CRITICAL if crit else SUCCESS)
    st.markdown("</div>", unsafe_allow_html=True)

    _render_fleet_recommendations_card(df)

    st.markdown('<div class="fleet-chart-grid">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        _fleet_chart_slot("Risk level distribution", charts.fig_risk_distribution, df, tune_fn=_tune_risk_distribution)
    with col2:
        _fleet_chart_slot("Anomaly severity", charts.fig_severity_donut, df, tune_fn=_tune_fleet_donut)

    row2a, row2b = st.columns(2, gap="large")
    with row2a:
        _fleet_chart_slot("Fleet quality score", charts.fig_fleet_quality_gauge, health, tune_fn=_tune_gauge)
    with row2b:
        _fleet_chart_slot(
            "Process stability score",
            charts.fig_gauge,
            stability,
            "Process Stability Score",
            tune_fn=_tune_gauge,
        )
    st.markdown("</div>", unsafe_allow_html=True)


def _section_casting_comparison(df: pd.DataFrame):
    major_section(
        "Casting Comparison",
        "Compare process conditions, chemistry, and risk indicators against fleet averages.",
        help_text=CHART_HELP["Casting Comparison"],
    )
    if df.empty:
        _set_selected_batch_idx(0, df)
        st.caption("Viewing: No casting selected")
        st.info("No casting data available.")
        return

    labels, id_col = _batch_labels(df)
    current_idx = _set_selected_batch_idx(int(ss.get("main_batch_select", ss.get("selected_batch_idx", 0))), df)

    if len(df) <= 1:
        idx = current_idx if current_idx < len(df) else 0
        if labels:
            st.caption(f"Viewing: {labels[idx]}")
        else:
            st.caption("Viewing: No casting selected")
    else:
        st.markdown('<div class="casting-comparison-select">', unsafe_allow_html=True)
        idx = st.selectbox(
            "Select Casting",
            list(range(len(df))),
            format_func=lambda i: labels[i],
            index=current_idx,
            key="main_batch_select",
        )
        st.markdown("</div>", unsafe_allow_html=True)
    _set_selected_batch_idx(idx, df, sync_widget=False)
    row = df.iloc[idx]

    st.markdown('<div class="comparison-block">', unsafe_allow_html=True)
    with _border_container():
        _chart_caption("Chemistry Profile Comparison", "Casting Comparison")
        radar = _healthy_vs_casting_radar(row, df)
        if radar is None:
            st.info("Not enough numeric features for comparison radar.")
        else:
            _show_panel_chart(_tune_radar(radar), "Casting vs healthy fleet")

    st.markdown("</div>", unsafe_allow_html=True)


def _selected_from_plotly_event(event) -> int | None:
    try:
        points = event.selection.points
    except AttributeError:
        points = event.get("selection", {}).get("points", []) if isinstance(event, dict) else []
    if not points:
        return None
    point = points[0]
    customdata = point.get("customdata") if isinstance(point, dict) else getattr(point, "customdata", None)
    if customdata is not None and len(customdata) > 0:
        try:
            return int(customdata[0])
        except (TypeError, ValueError):
            return None
    if isinstance(point, dict):
        point_index = point.get("point_index", point.get("pointNumber"))
        try:
            return int(point_index) if point_index is not None else None
        except (TypeError, ValueError):
            return None
    return None


def _section_cluster_analysis(df: pd.DataFrame):
    st.markdown('<div class="cluster-section-wrap">', unsafe_allow_html=True)
    major_section(
        "Cluster Analysis",
        "Interactive casting map colored by final business decision.",
        help_text=CHART_HELP["Cluster Analysis"],
    )
    selected_idx = _set_selected_batch_idx(int(ss.get("selected_batch_idx", 0)), df)
    fig = charts.fig_cluster_analysis(df, selected_idx=selected_idx)
    st.markdown('<div class="cluster-analysis-block">', unsafe_allow_html=True)
    try:
        event = st.plotly_chart(
            fig,
            use_container_width=True,
            config={**plotly_config(), "scrollZoom": True},
            key="cluster_analysis_scatter",
            on_select="rerun",
            selection_mode="points",
        )
        clicked_idx = _selected_from_plotly_event(event)
        if clicked_idx is not None:
            _set_selected_batch_idx(clicked_idx, df)
    except Exception as exc:
        st.warning(f"Cluster analysis could not be displayed ({exc}).")

    st.markdown("</div>", unsafe_allow_html=True)
    rec_counts = df["recommendation"].astype(str).str.upper().value_counts() if "recommendation" in df.columns else pd.Series(dtype=int)
    risk_counts = _risk_category_counts(df)
    total_clusters = int(pd.to_numeric(df["cluster"], errors="coerce").nunique()) if "cluster" in df.columns else 0
    largest_cluster = int(df["cluster"].value_counts().max()) if "cluster" in df.columns and len(df) else len(df)

    st.markdown('<div class="cluster-summary-row">', unsafe_allow_html=True)
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

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

def _section_fleet_defect_drivers(df: pd.DataFrame):
    major_section(
        "Parameter Correlation",
        "Chemistry, thermal, and engineered drivers across the loaded fleet.",
        help_text=CHART_HELP["Parameter Correlation"],
    )
    st.markdown('<div class="comparison-block">', unsafe_allow_html=True)
    row1a, row1b = st.columns(2)
    with row1a:
        with _border_container():
            _chart_caption("Top defect-driving parameters", "Parameter Correlation")
            try:
                fig = _fig_top_defect_parameters(df, 8)
                if fig is not None:
                    _show_panel_chart(_tune_horizontal(fig, min_height=400), "Top defect-driving parameters")
            except Exception as exc:
                st.warning(f"Top defect-driving parameters: {exc}")
    with row1b:
        with _border_container():
            _chart_caption("Defect-driving features", "Parameter Correlation")
            try:
                fi = get_feature_importance() or correlation_feature_importance(df)
                if fi:
                    names, imp = fi
                    fig = _tune_horizontal(charts.fig_feature_importance(names, imp), min_height=400)
                else:
                    fig = _tune_horizontal(charts.fig_feature_importance_from_df(df), min_height=400)
                _show_panel_chart(fig, "Defect-driving features")
            except Exception as exc:
                st.warning(f"Defect-driving features: {exc}")

    st.markdown('<div class="comparison-heatmap">', unsafe_allow_html=True)
    with _border_container():
        _chart_caption("Parameter correlation (chemistry & process) — fleet-wide", "Parameter Correlation")
        try:
            hm = _fig_correlation_heatmap(df)
            if hm is None:
                st.info("Not enough numeric features for correlation heatmap.")
            else:
                _show_panel_chart(_tune_heatmap(hm), "Parameter correlation")
        except Exception as exc:
            st.warning(f"Parameter correlation: {exc}")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


def _section_single_batch(df: pd.DataFrame, idx: int):
    if df.empty:
        st.info("No casting data available.")
        return

    idx = _set_selected_batch_idx(idx, df, sync_widget=False)
    row = df.iloc[idx]
    fleet_prob = safe_probability(pd.to_numeric(df["defect_prob"], errors="coerce").mean()) if "defect_prob" in df.columns else 0.0
    def_prob = safe_probability(row.get("defect_prob", 0))
    anom = safe_probability(row.get("anomaly_score", 0))
    cluster = int(pd.to_numeric(row.get("cluster", 0), errors="coerce") or 0)
    risk = str(row.get("risk_level", "LOW"))
    rec = str(row.get("recommendation", "PROCEED"))
    final_score = float(pd.to_numeric(row.get("final_risk_score", 0), errors="coerce") or 0)
    confidence = safe_probability(row.get("risk_confidence", 0.5))
    factors = str(row.get("risk_factors", ""))

    help_block("Defect Probability", MAIN_TERMS["Defect Probability"])
    help_block("Risk Score", MAIN_TERMS["Risk Score"])

    risk_panel(risk, rec, final_score, confidence, factors)

    st.metric("vs fleet avg defect prob", f"{def_prob:.1%}", f"{def_prob - fleet_prob:+.1%}")

    left, right = st.columns(2)
    with left:
        section("Defect probability")
        safe_render("Defect probability gauge", charts.fig_gauge, def_prob, "Defect Probability")
        st.markdown("**Cluster**")
        st.write(f"#{cluster}")
    with right:
        section("QA decision")
        color = RISK_COLORS.get(risk, "#888")
        st.markdown(
            f'<div class="ind-card"><span style="color:{color};font-size:1.4rem;font-weight:700;">{risk}</span> '
            f" &nbsp; <span>{rec}</span></div>",
            unsafe_allow_html=True,
        )
        rec_badge(rec)
        st.text_area("Engineering report", str(row.get("qa_summary", "")), height=420, key=f"qa_{idx}")

    section("Key process parameters")
    display_keys = [
        "tapping_temp", "pouring_temp", "feat_temp_loss", "s_", "c_", "si_", "mn_", "ce",
        "feat_ce_calculated", "mg_recovery_", "feat_mn_s_ratio", "feat_sulfur_risk",
        "feat_shrinkage_risk_index", "feat_gas_risk_index", "feat_chemistry_instability",
        "anomaly_score", "defect_prob",
    ]
    avail = [(k, row[k]) for k in display_keys if k in row.index]
    if avail:
        param_df = pd.DataFrame(avail, columns=["Parameter", "Value"])
        param_df["Value"] = param_df["Value"].apply(
            lambda x: f"{float(x):.4f}" if isinstance(x, (int, float, np.floating)) else str(x)
        )
        st.dataframe(param_df.set_index("Parameter"), width="stretch")

    section("Chemistry profile vs fleet")
    chem = chemistry_columns(df)
    if not chem:
        chem = list(df.select_dtypes(include=[np.number]).columns[:6])
    sub = df.iloc[[idx]]
    if chem:
        safe_render("Chemistry bars", charts.fig_chemistry_comparison, df, sub, chem[:6])

    section("Why risky & recommendations")
    st.markdown(f"**Final recommendation:** {rec}")
    st.markdown(f"**Final risk level:** {risk}")
    for line in _centralized_reason_lines(row):
        st.markdown(f"- {line}")


def _section_anomaly(df: pd.DataFrame):
    help_block("Anomaly Severity", MAIN_TERMS["Anomaly Severity"])
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
    sel_sev = st.selectbox("Filter by severity", sev_options, key="main_anomaly_filter")
    filtered = df if sel_sev == "ALL" else df[df["anomaly_severity"].astype(str) == sel_sev]

    c1, c2 = st.columns(2)
    with c1:
        if "anomaly_severity" in df.columns:
            sev_c = df["anomaly_severity"].value_counts()
            order = [s for s in ["NORMAL", "LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL"] if s in sev_c.index]
            safe_render(
                "Severity distribution",
                charts.fig_horizontal_bar,
                order,
                [int(sev_c[o]) for o in order],
                [SEVERITY_COLORS.get(o, "#888") for o in order],
                "Severity distribution",
            )
    with c2:
        safe_render("Anomaly vs defect probability", charts.fig_anomaly_scatter, df)

    section("Top dangerous batches")
    id_cols: list[tuple[str, str]] = []
    pattern_col = _first_column_match(filtered, ["pattern_no", "Pattern No", "pattern no"])
    item_col = _first_column_match(filtered, ["item", "Item Name", "item name"])
    if pattern_col:
        id_cols.append((pattern_col, "Pattern No"))
    if item_col:
        id_cols.append((item_col, "Item Name"))

    metric_cols = [
        c
        for c in [
            "anomaly_score", "anomaly_severity", "defect_prob", "final_risk_score",
            "risk_level", "recommendation", "cluster",
        ]
        if c in filtered.columns
    ]
    show_cols = [source for source, _ in id_cols] + metric_cols
    sort_cols = [c for c in ["final_risk_score", "anomaly_score"] if c in filtered.columns]
    if show_cols and sort_cols:
        top = filtered.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(30)
        display = top[show_cols].rename(columns={source: label for source, label in id_cols}).reset_index(drop=True)
        column_config = {
            "Pattern No": st.column_config.TextColumn(
                "Pattern No",
                help="Pattern No = casting pattern identifier",
            ),
            "Item Name": st.column_config.TextColumn(
                "Item Name",
                help="Item Name = casting component description",
            ),
        }
        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
            column_config={k: v for k, v in column_config.items() if k in display.columns},
        )




def render(df=None):
    del df
    page_header("Casting AI Quality Monitor", "Upload, analyze, and compare every casting in one place")

    if "current_data" not in ss:
        with st.spinner("Loading default dataset…"):
            default = load_default_data()
            if default is not None:
                ss["current_data"] = ensure_unified_decisions(default)
                ss["_dashboard_prepared"] = True
                ss["_decision_logic_version"] = DECISION_LOGIC_VERSION

    _section_upload()

    if "current_data" not in ss:
        empty_state(
            "Upload your casting data",
            "Drag and drop an Excel or CSV file above to run predictions and view fleet analytics.",
            "🏭",
        )
        return

    ss["current_data"] = ensure_unified_decisions(ss["current_data"])
    df = safe_for_plotting(safe_dataframe(ss["current_data"]))
    fname = ss.get("upload_filename", "Default / outputs dataset")
    st.caption(f"Active dataset: **{fname}** · **{len(df):,}** castings")

    _inject_main_layout_css()
    _report_results_header(df)
    _section_fleet_overview(df)
    _section_fleet_defect_drivers(df)
    _section_cluster_analysis(df)

    major_section("Anomaly Report", "Detect unusual process behavior and review the most dangerous batches.", compact=True)
    with st.expander("View anomaly details", expanded=False):
        _section_anomaly(df)

    _section_casting_comparison(df)
    idx = _set_selected_batch_idx(int(ss.get("selected_batch_idx", 0)), df, sync_widget=False)

    with st.expander("Single Batch Analysis", expanded=False):
        _section_single_batch(df, idx)
