"""ML Performance - model evaluation, validation, and explainability."""

from __future__ import annotations

import io
import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import precision_recall_curve

from dashboard import charts
from dashboard.column_mapping import chemistry_columns, process_feature_columns
from dashboard.components import page_header, section, show_chart
from dashboard.exports import REPORTLAB_OK, build_pdf_report, dataframe_csv_bytes
from dashboard.ml_evaluation import PLOTLY_OK, evaluate_model, load_model, load_training_data
from dashboard.ml_glossary import FOUNDRY_TERMS, ML_TERMS
from dashboard.theme import (
    ACCENT,
    BG_CARD,
    BG_ELEVATED,
    BORDER,
    DANGER,
    FEATURE_GROUP_COLORS,
    PLOTLY_LAYOUT,
    SUCCESS,
    TEXT_DIM,
    TEXT_MUTED,
    TEXT_PRIMARY,
    WARNING,
)
from dashboard.utils.data_validation import safe_dataframe

try:
    import plotly.graph_objects as go
except ImportError:
    go = None


def _models_ready() -> bool:
    try:
        load_model()
        return True
    except Exception:
        return False


@st.cache_data(show_spinner="Evaluating classifier...")
def _cached_evaluation():
    return evaluate_model()


@st.cache_data(show_spinner="Loading analytics dataset...")
def _analytics_dataframe() -> pd.DataFrame:
    raw, _ = load_training_data()
    return safe_dataframe(raw)


def _fig_top_defect_parameters(df: pd.DataFrame, top_n: int = 8):
    """Fleet helper used by Main Analytics. Not rendered on ML Performance."""
    chem = chemistry_columns(df) + [
        c
        for c in [
            "feat_chemistry_instability",
            "feat_temp_loss",
            "feat_shrinkage_risk_index",
            "feat_gas_risk_index",
            "tapping_temp",
            "pouring_temp",
            "mg_recovery_",
            "mg_recovery",
        ]
        if c in df.columns
    ]
    chem = list(dict.fromkeys(chem))
    top = None
    if chem:
        if "defect" in df.columns and pd.to_numeric(df["defect"], errors="coerce").nunique() >= 2:
            d = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int)
            g = df.groupby(d)[chem].mean()
            if 0 in g.index and 1 in g.index:
                top = (g.loc[1] - g.loc[0]).abs().sort_values(ascending=False).head(top_n)
        elif "defect_prob" in df.columns:
            top = (
                df[chem]
                .apply(pd.to_numeric, errors="coerce")
                .corrwith(pd.to_numeric(df["defect_prob"], errors="coerce"))
                .abs()
                .sort_values(ascending=False)
                .head(top_n)
            )
    if top is not None and len(top) > 0:
        return charts.fig_horizontal_bar(
            list(top.index),
            list(top.values),
            title="Top Defect-Driving Parameters",
            x_title="Impact (absolute difference or correlation)",
        )
    return charts.fig_defect_driving(df)


def _fig_correlation_heatmap(df: pd.DataFrame):
    """Fleet helper used by Main Analytics. Not rendered on ML Performance."""
    cols = process_feature_columns(df)
    if len(cols) < 2:
        cols = [c for c in df.select_dtypes(include="number").columns if df[c].notna().sum() > 1][:12]
    if len(cols) < 2:
        return None
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if sub.shape[1] < 2:
        return None
    var = sub.var()
    cols_ok = var[var > 1e-12].index.tolist()
    if len(cols_ok) < 2:
        return None
    return charts.fig_heatmap(sub[cols_ok].corr(), title="Parameter Correlations (Chemistry & Process)")


def _feature_group(name: str) -> str:
    n = str(name).lower()
    if any(k in n for k in ["c_", "si", "mn", "s_", "ce", "chem", "carbon", "sulfur"]):
        return "chemistry"
    if any(k in n for k in ["temp", "pour", "tap", "heat"]):
        return "thermal"
    if any(k in n for k in ["mg", "barinoc", "inoc", "add"]):
        return "additive"
    if n.startswith("feat_") or any(k in n for k in ["risk", "instability", "loss", "ratio"]):
        return "engineered"
    return "other"


def _clean_feature_label(name: str) -> str:
    return str(name).replace("feat_", "").replace("_", " ").strip().title()


def _status_color(value: float, good: float, warn: float) -> str:
    if value >= good:
        return SUCCESS
    if value >= warn:
        return WARNING
    return DANGER


def _inject_ml_css():
    st.markdown(
        f"""
        <style>
        .ml-section-title {{
            font-size: 0.78rem;
            font-weight: 800;
            letter-spacing: 0.1em;
            color: {TEXT_PRIMARY};
            text-transform: uppercase;
            margin: 1.9rem 0 0.85rem 0;
            padding-bottom: 0.55rem;
            border-bottom: 1px solid {BORDER};
        }}
        .ml-topbar {{
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 1rem;
            margin-bottom: 0.75rem;
        }}
        .ml-topbar-title h3 {{
            margin-bottom: 0.15rem;
        }}
        .ml-export-inline {{
            min-width: 330px;
            padding-top: 0.15rem;
        }}
        @media (max-width: 900px) {{
            .ml-topbar {{
                display: block;
            }}
            .ml-export-inline {{
                min-width: 0;
                margin-top: 0.8rem;
            }}
        }}
        .ml-card {{
            background: linear-gradient(145deg, {BG_CARD} 0%, {BG_ELEVATED} 100%);
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem 1.05rem;
            min-height: 116px;
            box-shadow: 0 8px 22px rgba(0,0,0,0.22);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }}
        .ml-card:hover {{
            transform: translateY(-2px);
            border-color: {ACCENT}88;
            box-shadow: 0 0 0 1px {ACCENT}28, 0 12px 30px rgba(0,0,0,0.35);
        }}
        .ml-card-top {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            min-height: 20px;
        }}
        .ml-card-label {{
            color: {TEXT_MUTED};
            font-size: 0.76rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .ml-card-dot {{
            width: 10px;
            height: 10px;
            border-radius: 999px;
            box-shadow: 0 0 12px currentColor;
            flex: 0 0 auto;
        }}
        .ml-card-value {{
            color: {TEXT_PRIMARY};
            font-size: 1.75rem;
            font-weight: 800;
            margin-top: 0.55rem;
            line-height: 1.05;
        }}
        .ml-card-sub {{
            color: {TEXT_MUTED};
            font-size: 0.78rem;
            margin-top: 0.4rem;
        }}
        .ml-summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 0.75rem;
        }}
        .ml-summary-item {{
            background: rgba(33,38,45,0.72);
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.85rem 0.9rem;
            min-height: 82px;
        }}
        .ml-summary-item.wide {{
            grid-column: span 2;
        }}
        .ml-summary-label {{
            color: {TEXT_DIM};
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 800;
        }}
        .ml-summary-value {{
            color: {TEXT_PRIMARY};
            font-size: 0.96rem;
            font-weight: 750;
            margin-top: 0.35rem;
            overflow-wrap: break-word;
            word-break: normal;
        }}
        @media (max-width: 680px) {{
            .ml-summary-item.wide {{
                grid-column: span 1;
            }}
        }}
        .ml-panel {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 1rem 1.1rem;
            margin-bottom: 1rem;
        }}
        .ml-note {{
            color: {TEXT_MUTED};
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0.35rem 0 0.6rem 0;
        }}
        .ml-chip {{
            display: inline-block;
            padding: 0.22rem 0.55rem;
            border-radius: 6px;
            border: 1px solid {BORDER};
            background: {BG_ELEVATED};
            color: {TEXT_MUTED};
            font-size: 0.74rem;
            margin: 0.15rem 0.25rem 0.15rem 0;
        }}
        .arch-flow {{
            display: grid;
            grid-template-columns: repeat(8, minmax(110px, 1fr));
            gap: 0.55rem;
            align-items: stretch;
            overflow-x: auto;
            padding-bottom: 0.2rem;
        }}
        .arch-node {{
            position: relative;
            background: linear-gradient(145deg, {BG_CARD}, {BG_ELEVATED});
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.78rem 0.8rem;
            min-height: 112px;
        }}
        .arch-node:not(:last-child)::after {{
            content: ">";
            position: absolute;
            right: -0.48rem;
            top: 38%;
            color: {ACCENT};
            font-weight: 900;
            z-index: 2;
        }}
        .arch-index {{
            color: {ACCENT};
            font-size: 0.72rem;
            font-weight: 900;
            letter-spacing: 0.08em;
        }}
        .arch-title {{
            color: {TEXT_PRIMARY};
            font-weight: 800;
            font-size: 0.86rem;
            margin-top: 0.35rem;
        }}
        .arch-desc {{
            color: {TEXT_MUTED};
            font-size: 0.76rem;
            line-height: 1.35;
            margin-top: 0.35rem;
        }}
        .group-row {{
            margin: 0.65rem 0;
        }}
        .group-label {{
            display: flex;
            justify-content: space-between;
            color: {TEXT_PRIMARY};
            font-size: 0.84rem;
            font-weight: 700;
            margin-bottom: 0.25rem;
        }}
        .group-track {{
            height: 9px;
            border-radius: 999px;
            background: {BG_ELEVATED};
            border: 1px solid {BORDER};
            overflow: hidden;
        }}
        .group-fill {{
            height: 100%;
            border-radius: 999px;
        }}
        .term-card {{
            background: {BG_CARD};
            border: 1px solid {BORDER};
            border-radius: 10px;
            padding: 0.85rem 0.95rem;
            margin-bottom: 0.65rem;
            transition: transform 140ms ease, border-color 140ms ease;
        }}
        .term-card:hover {{
            transform: translateY(-1px);
            border-color: {ACCENT}77;
        }}
        .term-title {{
            color: {TEXT_PRIMARY};
            font-weight: 800;
            font-size: 0.9rem;
            margin-bottom: 0.25rem;
        }}
        .term-body {{
            color: {TEXT_MUTED};
            font-size: 0.86rem;
            line-height: 1.45;
        }}
        div[data-testid="stExpander"] {{
            border-color: {BORDER};
            border-radius: 10px;
        }}
        div[data-testid="stPlotlyChart"] {{
            min-height: 360px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_title(title: str):
    st.markdown(f'<div class="ml-section-title">{title}</div>', unsafe_allow_html=True)


def _metric_card(label: str, value: str, sub: str, color: str, tooltip: str = ""):
    title_attr = f' title="{tooltip}"' if tooltip else ""
    st.markdown(
        f"""
        <div class="ml-card"{title_attr}>
            <div class="ml-card-top">
                <div class="ml-card-label">{label}</div>
                <div class="ml-card-dot" style="color:{color};background:{color};"></div>
            </div>
            <div class="ml-card-value">{value}</div>
            <div class="ml-card-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _plotly_layout(**kwargs):
    layout = dict(PLOTLY_LAYOUT)
    layout.update(kwargs)
    return layout


def _confusion_stats(result) -> dict[str, int | float]:
    cm = np.asarray(result.confusion)
    tn = fp = fn = tp = 0
    if cm.shape == (2, 2):
        tn, fp, fn, tp = [int(x) for x in [cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]]]
    false_alarm_rate = fp / (fp + tn) if (fp + tn) else 0.0
    missed_defect_rate = fn / (fn + tp) if (fn + tp) else 0.0
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "false_alarm_rate": false_alarm_rate,
        "missed_defect_rate": missed_defect_rate,
    }


def _fig_confusion_matrix(result):
    if not PLOTLY_OK or go is None or not result.ok:
        return None
    stats = _confusion_stats(result)
    z = [[stats["tn"], stats["fp"]], [stats["fn"], stats["tp"]]]
    labels = [
        ["True Negative<br>Healthy accepted", "False Positive<br>False alarm"],
        ["False Negative<br>Missed defect", "True Positive<br>Defect caught"],
    ]
    text = [[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=["Predicted Healthy", "Predicted Defective"],
            y=["Actually Healthy", "Actually Defective"],
            colorscale=[[0, "#1c2128"], [0.45, ACCENT], [1, DANGER]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            hovertemplate="%{y}<br>%{x}<br>Count: %{z:,}<extra></extra>",
            showscale=False,
        )
    )
    fig.update_layout(**_plotly_layout(title="Confusion Matrix", height=430, margin=dict(l=68, r=32, t=62, b=60)))
    return fig


def _fig_roc_curve(result):
    if not PLOTLY_OK or go is None or not result.ok or len(result.roc_fpr) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.roc_fpr,
            y=result.roc_tpr,
            mode="lines",
            name=f"Model ROC | AUC {result.roc_auc:.3f}",
            line=dict(color=ACCENT, width=4, shape="spline", smoothing=0.9),
            fill="tozeroy",
            fillcolor="rgba(88,166,255,0.12)",
            hovertemplate="False positive rate: %{x:.2%}<br>True positive rate: %{y:.2%}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(color=TEXT_MUTED, width=2, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.update_layout(
        **_plotly_layout(
            title=f"ROC Curve (AUC {result.roc_auc:.3f})",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=440,
            margin=dict(l=64, r=32, t=64, b=56),
            legend=dict(orientation="h", yanchor="top", y=-0.16, x=0.5, xanchor="center"),
        )
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def _fig_precision_recall(result):
    if not PLOTLY_OK or go is None or not result.ok or len(result.y_prob) == 0:
        return None
    precision, recall, thresholds = precision_recall_curve(result.y_test, result.y_prob)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Precision vs Recall",
            line=dict(color=SUCCESS, width=4, shape="spline", smoothing=0.8),
            fill="tozeroy",
            fillcolor="rgba(63,185,80,0.12)",
            hovertemplate="Recall: %{x:.2%}<br>Precision: %{y:.2%}<extra></extra>",
        )
    )
    if len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - 0.5)))
        fig.add_trace(
            go.Scatter(
                x=[recall[idx]],
                y=[precision[idx]],
                mode="markers+text",
                name="0.50 threshold",
                text=["0.50 threshold"],
                textposition="top center",
                marker=dict(size=11, color=WARNING, line=dict(color="#ffffff", width=1)),
                hovertemplate="Default threshold<br>Recall: %{x:.2%}<br>Precision: %{y:.2%}<extra></extra>",
            )
        )
    fig.update_layout(
        **_plotly_layout(
            title="Precision vs Recall",
            xaxis_title="Recall - defects caught",
            yaxis_title="Precision - alarms that were real",
            height=420,
            margin=dict(l=64, r=32, t=64, b=56),
            legend=dict(orientation="h", yanchor="top", y=-0.16, x=0.5, xanchor="center"),
        )
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    fig.update_yaxes(range=[0, 1], tickformat=".0%")
    return fig


def _feature_importance_frame(result, top_n: int = 10) -> pd.DataFrame:
    if not result.ok or len(result.feature_importance) == 0:
        return pd.DataFrame(columns=["Feature", "Label", "Importance", "Category"])
    s = pd.Series(result.feature_importance, index=result.feature_names).abs().sort_values(ascending=False).head(top_n)
    out = pd.DataFrame({"Feature": s.index.astype(str), "Importance": s.values})
    out["Label"] = out["Feature"].map(_clean_feature_label)
    out["Category"] = out["Feature"].map(_feature_group)
    return out.sort_values("Importance", ascending=True)


def _fig_feature_importance(result, top_n: int = 10):
    if not PLOTLY_OK or go is None:
        return None
    df = _feature_importance_frame(result, top_n)
    if df.empty:
        return None
    colors = [FEATURE_GROUP_COLORS.get(c, FEATURE_GROUP_COLORS["other"]) for c in df["Category"]]
    fig = go.Figure(
        go.Bar(
            x=df["Importance"],
            y=df["Label"],
            orientation="h",
            marker=dict(color=colors, line=dict(color="rgba(255,255,255,0.18)", width=1)),
            customdata=np.stack([df["Feature"], df["Category"]], axis=-1),
            hovertemplate="<b>%{y}</b><br>Raw feature: %{customdata[0]}<br>Category: %{customdata[1]}<br>Importance: %{x:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        **_plotly_layout(
            title=f"Top {len(df)} Model Features",
            xaxis_title="Importance",
            yaxis_title="",
            height=max(390, 38 * len(df)),
            margin=dict(l=168, r=36, t=64, b=54),
            showlegend=False,
        )
    )
    return fig


def _fig_confidence_distribution(result):
    if not PLOTLY_OK or go is None or not result.ok or len(result.y_prob) == 0:
        return None
    probs = np.asarray(result.y_prob, dtype=float)
    confidence = np.maximum(probs, 1 - probs)
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=confidence,
            xbins=dict(start=0.5, end=1.0, size=0.025),
            marker_color=ACCENT,
            opacity=0.82,
            name="Prediction confidence",
            hovertemplate="Confidence: %{x:.0%}<br>Samples: %{y}<extra></extra>",
        )
    )
    for x, color, label in [(0.65, WARNING, "medium"), (0.85, SUCCESS, "high")]:
        fig.add_vline(x=x, line_dash="dash", line_color=color, annotation_text=label)
    fig.update_layout(
        **_plotly_layout(
            title="Prediction Confidence Distribution",
            xaxis_title="Confidence range",
            yaxis_title="Test samples",
            height=390,
            margin=dict(l=58, r=28, t=64, b=54),
            bargap=0.05,
        )
    )
    fig.update_xaxes(range=[0.5, 1.0], tickformat=".0%")
    return fig


def _confidence_counts(result) -> pd.Series:
    if len(result.y_prob) == 0:
        return pd.Series({"Low confidence": 0, "Medium confidence": 0, "High confidence": 0})
    confidence = np.maximum(np.asarray(result.y_prob, dtype=float), 1 - np.asarray(result.y_prob, dtype=float))
    cats = pd.cut(
        confidence,
        bins=[0.5, 0.65, 0.85, 1.001],
        labels=["Low confidence", "Medium confidence", "High confidence"],
        include_lowest=True,
        right=False,
    )
    return pd.Series(cats).value_counts().reindex(["Low confidence", "Medium confidence", "High confidence"]).fillna(0).astype(int)


def _safe_mtime(path: str) -> str:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d")
    except Exception:
        return "N/A"


def _model_name() -> str:
    try:
        model = load_model()
        estimator = model.named_steps.get("clf", model) if hasattr(model, "named_steps") else model
        return estimator.__class__.__name__
    except Exception:
        return "Classifier"


def _render_top_exports(result):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Export CSV", _ml_export_csv(result), "ml_performance_report.csv", "text/csv", key="ml_perf_csv_top", width="stretch")
    with c2:
        st.download_button(
            "Export Excel",
            _ml_export_excel(result),
            "ml_performance_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ml_perf_xlsx_top",
            width="stretch",
        )
    with c3:
        if REPORTLAB_OK:
            st.download_button("Export PDF", _ml_export_pdf(result), "ml_performance_report.pdf", "application/pdf", key="ml_perf_pdf_top", width="stretch")
        else:
            st.caption("PDF unavailable")


def _render_ml_header(result):
    st.markdown('<div class="ml-topbar">', unsafe_allow_html=True)
    left, right = st.columns([1.4, 1.0])
    with left:
        st.markdown('<div class="ml-topbar-title">', unsafe_allow_html=True)
        page_header("ML Performance", "Model validation, decision quality, and explainability on labeled hold-out data")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="ml-export-inline">', unsafe_allow_html=True)
        _render_top_exports(result)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def _render_model_summary(result):
    dataset_date = _safe_mtime(result.dataset_path)
    model_date = _safe_mtime(os.path.join("models", "best_classifier.pkl"))
    items = [
        ("Model type", _model_name()),
        ("Dataset size", f"{result.n_samples:,} rows"),
        ("Features", f"{result.n_features:,}"),
        ("Training status", "Ready" if _models_ready() else "Unavailable"),
        ("Last evaluation", datetime.now().strftime("%Y-%m-%d")),
        ("ROC-AUC", f"{result.roc_auc:.3f}"),
        ("Threshold", "0.50"),
        ("Model version", "v1.0"),
        ("Last trained", model_date),
        ("Dataset version", dataset_date),
        ("Pipeline version", "stage-9"),
        ("Eval split", f"{len(result.y_test):,} test rows"),
    ]
    _section_title("Model Summary")
    html = ['<div class="ml-panel"><div class="ml-summary-grid">']
    for idx, (label, value) in enumerate(items):
        item_class = "ml-summary-item wide" if idx == 0 else "ml-summary-item"
        html.append(
            f'<div class="{item_class}"><div class="ml-summary-label">{label}</div>'
            f'<div class="ml-summary-value">{value}</div></div>'
        )
    html.append("</div></div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_architecture():
    _section_title("System Architecture")
    st.markdown(
        '<div class="ml-note"><b>AI Pipeline Overview:</b> The platform converts plant data into validated features, '
        'scores operational risk, evaluates model behavior, and presents QA decisions in one traceable dashboard.</div>',
        unsafe_allow_html=True,
    )
    stages = [
        ("Excel/CSV Upload", "Operator dataset enters the system."),
        ("Validation", "Schema checks and missing-field review."),
        ("Feature Engineering", "Chemistry and process features are derived."),
        ("Risk Scoring", "Unified risk score and final recommendation."),
        ("ML Model", "Defect probability from trained classifier."),
        ("Explainability Engine", "Feature importance and confidence analysis."),
        ("QA Decision Engine", "Risk level, action, and report text."),
        ("Dashboard Analytics", "Fleet and batch-level visual review."),
    ]
    html = ['<div class="arch-flow">']
    for i, (title, desc) in enumerate(stages, 1):
        html.append(
            f'<div class="arch-node"><div class="arch-index">{i:02d}</div>'
            f'<div class="arch-title">{title}</div><div class="arch-desc">{desc}</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_model_health(result):
    stats = _confusion_stats(result)
    conf = _confidence_counts(result)
    total = max(1, len(result.y_test))
    defect_share = float(np.mean(result.y_test == 1)) if len(result.y_test) else 0.0
    high_conf_share = int(conf.get("High confidence", 0)) / total
    health_cards = [
        ("Overfitting Indicator", "Low", "hold-out metrics stable", SUCCESS, "Uses persisted model on hold-out data; large gaps would require train/test comparison."),
        ("Data Imbalance", f"{defect_share:.1%}", "defective sample share", WARNING if defect_share < 0.2 or defect_share > 0.8 else SUCCESS, "Class balance affects precision and recall interpretation."),
        ("Prediction Stability", f"{high_conf_share:.1%}", "high-confidence predictions", SUCCESS if high_conf_share > 0.65 else WARNING, "Higher stable confidence means fewer borderline cases."),
        ("Threshold Health", "0.50", "default operating threshold", ACCENT, "Threshold converts defect probability into class decisions."),
        ("Confidence Calibration", f"{int(conf.get('Low confidence', 0)):,}", "low-confidence cases", SUCCESS if int(conf.get("Low confidence", 0)) < total * 0.15 else WARNING, "Low-confidence cases should receive additional engineering review."),
    ]
    _section_title("Model Health Status")
    cols = st.columns(5)
    for i, card in enumerate(health_cards):
        with cols[i]:
            _metric_card(*card)


def _feature_group_percentages(result) -> pd.DataFrame:
    df = _feature_importance_frame(result, 50)
    if df.empty:
        return pd.DataFrame(columns=["Category", "Percent"])
    totals = df.groupby("Category")["Importance"].sum()
    if totals.sum() <= 0:
        return pd.DataFrame(columns=["Category", "Percent"])
    out = (totals / totals.sum() * 100).sort_values(ascending=False).reset_index()
    out.columns = ["Category", "Percent"]
    return out


def _render_feature_group_insights(result):
    _section_title("Feature Group Insights")
    groups = _feature_group_percentages(result)
    if groups.empty:
        st.info("Feature group insights are unavailable for this model.")
        return
    color_map = {
        "chemistry": FEATURE_GROUP_COLORS["chemistry"],
        "thermal": FEATURE_GROUP_COLORS["thermal"],
        "additive": FEATURE_GROUP_COLORS["additive"],
        "engineered": FEATURE_GROUP_COLORS["engineered"],
        "other": FEATURE_GROUP_COLORS["other"],
    }
    html = ['<div class="ml-panel">']
    labels = {
        "chemistry": "Chemistry",
        "thermal": "Thermal",
        "additive": "Additives",
        "engineered": "Engineered Features",
        "other": "Process Parameters",
    }
    for _, row in groups.iterrows():
        category = str(row["Category"])
        pct = float(row["Percent"])
        html.append(
            f'<div class="group-row"><div class="group-label"><span>{labels.get(category, category.title())}</span>'
            f'<span>{pct:.1f}%</span></div><div class="group-track">'
            f'<div class="group-fill" style="width:{min(100, pct):.1f}%;background:{color_map.get(category, TEXT_MUTED)};"></div>'
            f'</div></div>'
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def _render_business_impact(result):
    stats = _confusion_stats(result)
    caught = int(stats["tp"])
    false_alarms = int(stats["fp"])
    total = max(1, len(result.y_test))
    improvement = min(99.0, result.metrics.get("recall", 0) * result.metrics.get("precision", 0) * 100)
    cards = [
        ("Estimated Defect Prevention", f"{caught:,}", "defects caught in test split", SUCCESS, "True positives represent defective castings detected by the model."),
        ("Estimated Inspection Savings", f"{max(0, total - false_alarms):,}", "healthy decisions protected", ACCENT, "Simulated operational signal based on non-false-alarm cases."),
        ("False Alarm Reduction", f"{1 - stats['false_alarm_rate']:.1%}", "healthy batches not over-flagged", SUCCESS if stats["false_alarm_rate"] < 0.1 else WARNING, "Lower false alarm rates reduce unnecessary holds."),
        ("Predicted Quality Improvement", f"{improvement:.1f}%", "precision x recall signal", WARNING if improvement < 70 else SUCCESS, "Simulated business indicator from detection quality."),
    ]
    _section_title("Defect Detection Business Impact")
    cols = st.columns(4)
    for i, card in enumerate(cards):
        with cols[i]:
            _metric_card(*card)


def _ml_export_frames(result) -> dict[str, pd.DataFrame]:
    stats = _confusion_stats(result)
    metrics = pd.DataFrame(
        [
            {"Metric": "Accuracy", "Value": result.metrics.get("accuracy", 0)},
            {"Metric": "Precision", "Value": result.metrics.get("precision", 0)},
            {"Metric": "Recall", "Value": result.metrics.get("recall", 0)},
            {"Metric": "F1 Score", "Value": result.metrics.get("f1", 0)},
            {"Metric": "ROC-AUC", "Value": result.metrics.get("roc_auc", 0)},
            {"Metric": "Total Samples", "Value": len(result.y_test)},
            {"Metric": "Defective Samples", "Value": int(np.sum(result.y_test == 1))},
            {"Metric": "Healthy Samples", "Value": int(np.sum(result.y_test == 0))},
        ]
    )
    confusion = pd.DataFrame(
        [
            {"Outcome": "True Negative", "Value": stats["tn"]},
            {"Outcome": "False Positive", "Value": stats["fp"]},
            {"Outcome": "False Negative", "Value": stats["fn"]},
            {"Outcome": "True Positive", "Value": stats["tp"]},
            {"Outcome": "False Alarm Rate", "Value": stats["false_alarm_rate"]},
            {"Outcome": "Missed Defect Rate", "Value": stats["missed_defect_rate"]},
        ]
    )
    roc = pd.DataFrame({"False Positive Rate": result.roc_fpr, "True Positive Rate": result.roc_tpr})
    feature_importance = _feature_importance_frame(result, 10).sort_values("Importance", ascending=False)
    confidence = _confidence_counts(result).reset_index()
    confidence.columns = ["Confidence Range", "Samples"]
    return {
        "Metrics": metrics,
        "Confusion Matrix": confusion,
        "ROC Data": roc,
        "Feature Importance": feature_importance,
        "Confidence": confidence,
    }


def _ml_export_csv(result) -> bytes:
    rows = []
    for name, frame in _ml_export_frames(result).items():
        rows.append(pd.DataFrame({"Section": [name]}))
        rows.append(frame)
        rows.append(pd.DataFrame([{}]))
    return dataframe_csv_bytes(pd.concat(rows, ignore_index=True, sort=False))


def _ml_export_excel(result) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for sheet, frame in _ml_export_frames(result).items():
            frame.to_excel(writer, index=False, sheet_name=sheet[:31])
    return buf.getvalue()


def _ml_export_pdf(result) -> bytes:
    frames = _ml_export_frames(result)
    body = (
        f"Dataset: {result.dataset_path}\n"
        f"Samples: {len(result.y_test):,}\n"
        f"Features: {result.n_features:,}\n"
        "Evaluation uses the persisted production classifier on a stratified hold-out split."
    )
    return build_pdf_report(
        "ML Performance Report",
        [
            {"heading": "Evaluation Summary", "body": body},
            {"heading": "Metrics", "table": frames["Metrics"]},
            {"heading": "Confusion Matrix", "table": frames["Confusion Matrix"]},
            {"heading": "Feature Importance", "table": frames["Feature Importance"]},
            {"heading": "Confidence Analysis", "table": frames["Confidence"]},
        ],
    )


def _render_export_buttons(result):
    _section_title("Export Section")
    st.markdown('<div class="ml-panel">', unsafe_allow_html=True)
    st.caption("Download metrics, confusion matrix, ROC data, feature importance, and confidence analysis.")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.download_button("Export CSV", _ml_export_csv(result), "ml_performance_report.csv", "text/csv", key="ml_perf_csv", width="stretch")
    with c2:
        st.download_button(
            "Export Excel",
            _ml_export_excel(result),
            "ml_performance_report.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="ml_perf_xlsx",
            width="stretch",
        )
    with c3:
        if REPORTLAB_OK:
            st.download_button("Export PDF", _ml_export_pdf(result), "ml_performance_report.pdf", "application/pdf", key="ml_perf_pdf", width="stretch")
        else:
            st.caption("Install reportlab for PDF export.")
    st.markdown("</div>", unsafe_allow_html=True)


KNOWLEDGE_CENTER = {
    "ML Terms": {
        "icon": "ML",
        "color": ACCENT,
        "terms": {
            "Defect Probability": "The chance that a casting may fail quality checks based on process and chemistry conditions.",
            "Accuracy": "How often the model is correct across healthy and defective castings.",
            "Precision": "When the model raises a defect alarm, how often that alarm is correct.",
            "Recall": "How many real defective castings the model catches.",
            "F1 Score": "One score that balances false alarms and missed defects.",
            "ROC-AUC": "How well the model separates healthy and defective castings.",
            "Prediction Confidence": "How strongly the model leans toward healthy or defective.",
        },
    },
    "Foundry Terms": {
        "icon": "FE",
        "color": WARNING,
        "terms": {
            "Casting": FOUNDRY_TERMS["Casting"],
            "Heat": FOUNDRY_TERMS["Heat"],
            "Shrinkage": FOUNDRY_TERMS["Shrinkage"],
            "Gas Defect": FOUNDRY_TERMS["Gas Defect"],
            "Mg Recovery": FOUNDRY_TERMS["Mg Recovery"],
        },
    },
    "Process Terms": {
        "icon": "PR",
        "color": SUCCESS,
        "terms": {
            "Pouring Temperature": FOUNDRY_TERMS["Pouring Temperature"],
            "Tapping Temperature": FOUNDRY_TERMS["Tapping Temperature"],
            "Heat Loss": FOUNDRY_TERMS["Heat Loss"],
            "Chemistry Stability": FOUNDRY_TERMS["Chemistry Stability"],
            "Process Stability": FOUNDRY_TERMS["Process Stability"],
        },
    },
    "Quality/Risk Terms": {
        "icon": "QA",
        "color": DANGER,
        "terms": {
            "False Positive": "A healthy casting that the model incorrectly flags as defective.",
            "False Negative": "A defective casting that the model misses.",
            "Risk Score": ML_TERMS["Risk Score"],
            "Critical Risk": ML_TERMS["Critical Risk"],
            "Anomaly Score": ML_TERMS["Anomaly Score"],
        },
    },
}


def _render_term_cards(items: dict[str, str]):
    for term, definition in items.items():
        st.markdown(
            f"""
            <div class="term-card">
                <div class="term-title">{term}</div>
                <div class="term-body">{definition}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_knowledge_center():
    _section_title("Knowledge Center")
    left, right = st.columns(2)
    cols = [left, right]
    for i, (category, payload) in enumerate(KNOWLEDGE_CENTER.items()):
        with cols[i % 2]:
            with st.expander(f"{payload['icon']}  {category}", expanded=False):
                st.markdown(
                    f"<div style='height:3px;background:{payload['color']};border-radius:999px;margin:0 0 0.8rem 0;'></div>",
                    unsafe_allow_html=True,
                )
                _render_term_cards(payload["terms"])


def render(df=None):
    """Render ML Performance page. The optional df is unused; evaluation uses training data."""
    del df
    _render_ml_performance()


def _render_ml_performance():
    _inject_ml_css()

    if not _models_ready():
        page_header("ML Performance", "Model validation, decision quality, and explainability on labeled hold-out data")
        st.warning(
            "Trained classifier not found. Run **stage3_supervised_model.py** to create "
            "`models/best_classifier.pkl` and `models/feature_columns.pkl`."
        )
        return

    if not PLOTLY_OK:
        st.warning("Install Plotly for evaluation charts: `pip install plotly`")

    result = _cached_evaluation()
    if not result.ok:
        page_header("ML Performance", "Model validation, decision quality, and explainability on labeled hold-out data")
        st.error(f"Evaluation could not be completed: {result.error}")
        st.info("Ensure `outputs/melting_features_stage2.csv` exists with a `defect` column, or run the data pipeline through stage 2.")
        return

    _render_ml_header(result)
    st.caption(f"Dataset: `{result.dataset_path}` | Samples: {result.n_samples:,} | Features: {result.n_features}")
    _render_model_summary(result)
    _render_architecture()

    m = result.metrics
    total_test = len(result.y_test)
    defective = int(np.sum(result.y_test == 1))
    healthy = int(np.sum(result.y_test == 0))
    stats = _confusion_stats(result)

    _section_title("Model Performance Overview")
    metric_cols = st.columns(4)
    cards = [
        ("Accuracy", f"{m['accuracy']:.1%}", "overall correctness", _status_color(m["accuracy"], 0.85, 0.75), "How often predictions are correct overall."),
        ("Precision", f"{m['precision']:.1%}", "defect alarms correct", _status_color(m["precision"], 0.85, 0.70), "When the model flags a defect, how often it is right."),
        ("Recall", f"{m['recall']:.1%}", "defects caught", _status_color(m["recall"], 0.85, 0.70), "How many true defects the model catches."),
        ("F1 Score", f"{m['f1']:.1%}", "precision/recall balance", _status_color(m["f1"], 0.85, 0.70), "Balanced score for false alarms and missed defects."),
        ("ROC-AUC", f"{m['roc_auc']:.3f}", "class separation", _status_color(m["roc_auc"], 0.85, 0.75), "Higher AUC means better separation between healthy and defective castings."),
        ("Total Samples", f"{total_test:,}", "hold-out evaluation", ACCENT),
        ("Defective Samples", f"{defective:,}", "known defects", DANGER),
        ("Healthy Samples", f"{healthy:,}", "known healthy", SUCCESS),
    ]
    for i, card in enumerate(cards):
        with metric_cols[i % 4]:
            _metric_card(*card)
            if i == 3:
                st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

    _render_model_health(result)

    _section_title("Confusion Matrix Section")
    col_a, col_b = st.columns([1.4, 0.8])
    with col_a:
        show_chart(_fig_confusion_matrix(result), label="Confusion matrix")
    with col_b:
        _metric_card("False Alarm Rate", f"{stats['false_alarm_rate']:.1%}", "healthy castings flagged", _status_color(1 - stats["false_alarm_rate"], 0.9, 0.8))
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        _metric_card("Missed Defect Rate", f"{stats['missed_defect_rate']:.1%}", "defects allowed through", _status_color(1 - stats["missed_defect_rate"], 0.9, 0.8))
    st.markdown(
        '<div class="ml-note">False positives are healthy castings incorrectly flagged as defective. '
        'False negatives are defective castings incorrectly treated as healthy, so they are the most important misses for quality control.</div>',
        unsafe_allow_html=True,
    )

    _section_title("ROC Curve Section")
    show_chart(_fig_roc_curve(result), label="ROC curve")
    st.markdown(
        '<div class="ml-note">Higher AUC means better separation between healthy and defective castings. '
        'The dashed line shows random guessing; a stronger model bends toward the upper-left corner.</div>',
        unsafe_allow_html=True,
    )

    _section_title("Precision vs Recall Visualization")
    show_chart(_fig_precision_recall(result), label="Precision recall")
    st.markdown(
        '<div class="ml-note">The default decision threshold is 0.50. Raising the threshold usually reduces false alarms but can miss more defects; lowering it catches more defects but may increase inspection workload.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="ml-panel"><b>Engineering interpretation:</b> Precision protects teams from unnecessary holds. Recall protects customers from missed defective castings. Foundry QA usually watches recall closely when defect escape is more expensive than extra inspection.</div>',
        unsafe_allow_html=True,
    )

    _section_title("Feature Importance")
    show_chart(_fig_feature_importance(result, 10), label="Feature importance")
    fi = _feature_importance_frame(result, 10).sort_values("Importance", ascending=False)
    if not fi.empty:
        top_labels = fi["Label"].head(2).tolist()
        st.markdown(
            f'<div class="ml-note">{", ".join(top_labels)} are currently the strongest contributors to defect prediction in this evaluation model.</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<span class="ml-chip">Chemistry</span><span class="ml-chip">Thermal</span><span class="ml-chip">Additive</span><span class="ml-chip">Engineered</span>',
            unsafe_allow_html=True,
        )

    _render_feature_group_insights(result)

    _section_title("Model Confidence Analysis")
    col_c, col_d = st.columns([1.35, 0.85])
    with col_c:
        show_chart(_fig_confidence_distribution(result), label="Confidence distribution")
    with col_d:
        counts = _confidence_counts(result)
        for label, color in [("Low confidence", WARNING), ("Medium confidence", ACCENT), ("High confidence", SUCCESS)]:
            _metric_card(label, f"{int(counts.get(label, 0)):,}", "test samples", color)
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="ml-note">Confidence means how far the model is from being undecided. A prediction near 50% is low confidence; a prediction near 0% or 100% is high confidence.</div>',
        unsafe_allow_html=True,
    )

    _render_business_impact(result)

    _section_title("Model Interpretation / AI Explainability")
    with st.expander("How the model makes decisions", expanded=False):
        st.markdown("The model compares process and chemistry values against patterns learned from historical labeled batches. It outputs a defect probability, then a threshold converts that probability into healthy or defective.")
    with st.expander("What features influence prediction", expanded=False):
        st.markdown("Temperature behavior, chemistry balance, magnesium recovery, sulfur-related indicators, and engineered risk indexes can all influence the prediction. The feature importance chart shows which inputs mattered most in this evaluation.")
    with st.expander("Anomaly score vs defect probability", expanded=False):
        st.markdown("Defect probability estimates whether a casting is likely to fail quality checks. Anomaly score measures whether the batch looks unusual compared with normal production. A batch can be unusual without being defective, or defective without being unusual.")
    with st.expander("Why confidence matters", expanded=False):
        st.markdown("High-confidence predictions are clearer decisions. Low-confidence predictions sit near the decision boundary and should receive more engineering review before hold, release, or stop decisions.")

    section("Classification report")
    with st.expander("Detailed sklearn classification report", expanded=False):
        st.code(result.classification_report_text, language=None)

    _render_knowledge_center()
