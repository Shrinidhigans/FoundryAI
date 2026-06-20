"""Plotly chart factories — always return valid figures with computed fallbacks."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY_AVAILABLE = True
except ImportError:
    px = go = None
    PLOTLY_AVAILABLE = False

from dashboard.theme import (
    ACCENT,
    DANGER,
    FEATURE_GROUP_COLORS,
    PLOTLY_LAYOUT,
    REC_COLORS,
    RISK_COLORS,
    SEVERITY_COLORS,
    SUCCESS,
    WARNING,
)

RISK_DISPLAY_ORDER = ["HEALTHY", "LOW", "MEDIUM", "HIGH", "CRITICAL"]


def _layout(**kwargs):
    base = dict(PLOTLY_LAYOUT)
    base.update(kwargs)
    return base


def _empty_fig(title: str = "Chart"):
    if not PLOTLY_AVAILABLE or go is None:
        return None
    fig = go.Figure()
    fig.update_layout(**_layout(title=title, height=320))
    return fig


def _require_plotly(title: str = "Chart"):
    if not PLOTLY_AVAILABLE or go is None:
        return None
    return _empty_fig(title)


def _numeric_cols(df: pd.DataFrame, exclude: set | None = None) -> list[str]:
    exclude = exclude or set()
    out = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            out.append(c)
    return out


def _derived_score(df: pd.DataFrame, col: str) -> pd.Series:
    """Fallback score from z-scored numeric features when primary column missing."""
    if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
        s = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if s.max() > 0 or s.std() > 0:
            return s
    nums = _numeric_cols(df)
    if not nums:
        return pd.Series(0.0, index=df.index)
    X = df[nums].apply(pd.to_numeric, errors="coerce").fillna(df[nums].median())
    z = (X - X.mean()) / (X.std() + 1e-9)
    raw = z.abs().mean(axis=1)
    return (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)


def _feature_group(name: str) -> str:
    n = name.lower()
    if any(x in n for x in ("s_", "c_", "si_", "mn_", "ce", "sulfur", "chem", "p__")):
        return "chemistry"
    if any(x in n for x in ("temp", "pour", "tap", "thermal")):
        return "thermal"
    if any(x in n for x in ("mg", "fsm", "additive", "recovery")):
        return "additive"
    if n.startswith("feat_"):
        return "engineered"
    return "other"


def fig_horizontal_bar(categories, values, colors=None, title="", x_title="Count"):
    if not PLOTLY_AVAILABLE:
        return _require_plotly(title or "Bar chart")
    if not categories:
        categories, values = ["N/A"], [0]
    if colors is None:
        colors = ACCENT
    fig = go.Figure(
        go.Bar(
            y=[str(c) for c in categories],
            x=values,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v:.3g}" if isinstance(v, (float, np.floating)) else str(v) for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(**_layout(title=title, xaxis_title=x_title, height=max(280, 40 * len(categories))))
    return fig


def fig_donut(labels, values, colors, title="", hole=0.55):
    if not PLOTLY_AVAILABLE:
        return _require_plotly(title or "Distribution")
    if not labels or sum(values) == 0:
        labels, values = ["All batches"], [1]
        colors = [ACCENT]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=hole,
            marker=dict(colors=colors),
            textinfo="label+percent",
            textposition="outside",
        )
    )
    fig.update_layout(**_layout(title=title, showlegend=True, height=360))
    return fig


def risk_category_series(df: pd.DataFrame) -> pd.Series:
    """Return dashboard risk buckets that include healthy/normal castings."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=str)
    if "risk_level" in df.columns:
        raw = df["risk_level"].astype(str).str.strip().str.upper()
    else:
        raw = pd.Series("HEALTHY", index=df.index)

    normalized = raw.replace(
        {
            "NORMAL": "HEALTHY",
            "OK": "HEALTHY",
            "PASS": "HEALTHY",
            "ACCEPT": "HEALTHY",
            "ACCEPTABLE": "HEALTHY",
            "NONE": "HEALTHY",
            "NAN": "HEALTHY",
        }
    )
    normalized = normalized.where(normalized.isin(RISK_DISPLAY_ORDER), "LOW")
    return normalized


def risk_category_counts(df: pd.DataFrame) -> dict[str, int]:
    risk = risk_category_series(df)
    vc = risk.value_counts()
    return {level: int(vc.get(level, 0)) for level in RISK_DISPLAY_ORDER}


def fig_histogram(series, title="", threshold=None, color=ACCENT):
    if not PLOTLY_AVAILABLE:
        return _require_plotly(title or "Histogram")
    if series is None or len(series) == 0:
        series = pd.Series([0.0])
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        s = pd.Series([0.0])
    fig = go.Figure(go.Histogram(x=s, nbinsx=min(40, max(5, len(s))), marker_color=color, opacity=0.85))
    if threshold is not None:
        fig.add_vline(x=threshold, line_dash="dash", line_color=DANGER, annotation_text="Threshold")
    fig.update_layout(**_layout(title=title, xaxis_title="Value", yaxis_title="Batches", height=340))
    return fig


def fig_gauge(value: float, title: str, max_val: float = 1.0):
    if not PLOTLY_AVAILABLE:
        return _require_plotly(title)
    pct = min(100, max(0, 100 * float(value) / max_val))
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 28}},
            title={"text": title, "font": {"size": 14}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": ACCENT},
                "steps": [
                    {"range": [0, 30], "color": SUCCESS},
                    {"range": [30, 60], "color": WARNING},
                    {"range": [60, 80], "color": DANGER},
                    {"range": [80, 100], "color": "#9b59b6"},
                ],
            },
        )
    )
    fig.update_layout(**_layout(height=280))
    return fig


def fig_trend_line(x, y, title="", y_title="Value", highlight_mask=None):
    if not PLOTLY_AVAILABLE:
        return _require_plotly(title)
    x_arr = np.asarray(x) if x is not None else np.arange(len(y) if y is not None else 1)
    y_arr = np.asarray(y) if y is not None else np.zeros(len(x_arr))
    fig = go.Figure(go.Scatter(x=x_arr, y=y_arr, mode="lines", line=dict(color=ACCENT, width=1.5), name="Series"))
    if highlight_mask is not None and np.asarray(highlight_mask).any():
        fig.add_trace(
            go.Scatter(
                x=x_arr[highlight_mask],
                y=y_arr[highlight_mask],
                mode="markers",
                marker=dict(color=DANGER, size=7),
                name="Highlight",
            )
        )
    fig.update_layout(**_layout(title=title, yaxis_title=y_title, height=300))
    return fig


def fig_pca_clusters(df, highlight_cluster=None, highlight_idx=None):
    if not PLOTLY_AVAILABLE or len(df) == 0:
        return _require_plotly("PCA Process Map")
    plot_df = df.copy()
    if "pca_pc1" not in plot_df.columns or "pca_pc2" not in plot_df.columns:
        nums = _numeric_cols(plot_df)
        if len(nums) >= 2:
            X = plot_df[nums].apply(pd.to_numeric, errors="coerce").fillna(plot_df[nums].median())
            plot_df["pca_pc1"] = X.iloc[:, 0]
            plot_df["pca_pc2"] = X.iloc[:, 1]
        else:
            plot_df["pca_pc1"] = np.arange(len(plot_df))
            plot_df["pca_pc2"] = _derived_score(plot_df, "anomaly_score")
    color_col = plot_df["cluster"].astype(str) if "cluster" in plot_df.columns else None
    fig = px.scatter(
        plot_df,
        x="pca_pc1",
        y="pca_pc2",
        color=color_col,
        hover_data={c: True for c in ["defect_prob", "anomaly_score", "risk_level", "recommendation", "defect"] if c in plot_df.columns},
        opacity=0.65,
    )
    if highlight_cluster is not None and "cluster" in plot_df.columns:
        sub = plot_df[plot_df["cluster"] == highlight_cluster]
        fig.add_trace(
            go.Scatter(
                x=sub["pca_pc1"],
                y=sub["pca_pc2"],
                mode="markers",
                marker=dict(size=11, color=ACCENT, line=dict(width=2, color="white")),
                name=f"Cluster {highlight_cluster}",
            )
        )
    if highlight_idx is not None and highlight_idx in plot_df.index:
        row = plot_df.loc[highlight_idx]
        fig.add_trace(
            go.Scatter(
                x=[row["pca_pc1"]],
                y=[row["pca_pc2"]],
                mode="markers",
                marker=dict(size=14, color=DANGER, symbol="x", line=dict(width=2, color="white")),
                name="Selected batch",
            )
        )
    fig.update_traces(marker=dict(size=9), selector=dict(type="scatter"))
    fig.update_layout(**_layout(title="PCA Process Map", height=440))
    return fig


def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower:
            return lower[candidate.lower()]
    return None


def cluster_scatter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare one plotted record per casting for the cluster analysis scatter."""
    plot_df = df.copy().reset_index(drop=False).rename(columns={"index": "_source_index"})
    plot_df["_row_pos"] = np.arange(len(plot_df))

    if "pca_pc1" in plot_df.columns and "pca_pc2" in plot_df.columns:
        plot_df["_x"] = pd.to_numeric(plot_df["pca_pc1"], errors="coerce").fillna(0.0)
        plot_df["_y"] = pd.to_numeric(plot_df["pca_pc2"], errors="coerce").fillna(0.0)
        plot_df["_x_title"] = "PCA Component 1"
        plot_df["_y_title"] = "PCA Component 2"
    else:
        plot_df["_x"] = 100.0 * _derived_score(plot_df, "defect_prob").clip(0.0, 1.0)
        plot_df["_y"] = _derived_score(plot_df, "anomaly_score").clip(0.0, 1.0)
        plot_df["_x_title"] = "Defect Probability (%)"
        plot_df["_y_title"] = "Anomaly Score"

    rec = plot_df["recommendation"] if "recommendation" in plot_df.columns else pd.Series("PROCEED", index=plot_df.index)
    plot_df["_status"] = rec.astype(str).str.upper().replace({"REJECT": "STOP"})
    plot_df["_status"] = plot_df["_status"].where(plot_df["_status"].isin(["STOP", "HOLD", "MONITOR", "PROCEED"]), "PROCEED")
    plot_df["_risk_level"] = risk_category_series(plot_df)
    plot_df["_defect_prob"] = _derived_score(plot_df, "defect_prob").clip(0.0, 1.0)
    plot_df["_anomaly_score"] = _derived_score(plot_df, "anomaly_score").clip(0.0, 1.0)
    plot_df["_cluster"] = (
        pd.to_numeric(plot_df["cluster"], errors="coerce").fillna(-1).astype(int)
        if "cluster" in plot_df.columns
        else pd.Series(-1, index=plot_df.index)
    )

    batch_col = _first_existing(plot_df, ["heat", "heat_no", "heat_number", "batch", "batch_id", "casting", "casting_no", "serial", "aa"])
    material_col = _first_existing(plot_df, ["material_number", "material_no", "material", "grade", "material_grade"])
    grade_col = _first_existing(plot_df, ["grade", "material_grade", "grade_name"])
    pattern_col = _first_existing(plot_df, ["pattern_number", "pattern_no", "pattern", "pattern_id"])

    plot_df["_batch"] = plot_df[batch_col].astype(str) if batch_col else plot_df["_row_pos"].map(lambda i: f"Batch {i}")
    plot_df["_material"] = plot_df[material_col].astype(str) if material_col else "N/A"
    plot_df["_grade"] = plot_df[grade_col].astype(str) if grade_col else "N/A"
    plot_df["_pattern"] = plot_df[pattern_col].astype(str) if pattern_col else "N/A"
    return plot_df


def fig_cluster_analysis(df: pd.DataFrame, selected_idx: int | None = None):
    if not PLOTLY_AVAILABLE or len(df) == 0:
        return _require_plotly("Cluster Analysis")

    plot_df = cluster_scatter_dataframe(df)
    fig = go.Figure()
    order = ["STOP", "HOLD", "MONITOR", "PROCEED"]
    labels = {
        "STOP": "Red = STOP",
        "HOLD": "Orange = HOLD",
        "MONITOR": "Yellow = MONITOR",
        "PROCEED": "Green = PROCEED",
    }
    for status in order:
        sub = plot_df[plot_df["_status"] == status]
        if len(sub) == 0:
            continue
        custom = np.stack(
            [
                sub["_row_pos"],
                sub["_batch"],
                sub["_material"],
                sub["_grade"],
                sub["_pattern"],
                sub["_status"],
                sub["_risk_level"],
                sub["_defect_prob"],
                sub["_anomaly_score"],
                sub["_cluster"],
            ],
            axis=-1,
        )
        fig.add_trace(
            go.Scattergl(
                x=sub["_x"],
                y=sub["_y"],
                mode="markers",
                name=labels[status],
                customdata=custom,
                marker=dict(
                    size=9,
                    color=REC_COLORS.get(status, ACCENT),
                    opacity=0.78,
                    line=dict(width=0.7, color="rgba(255,255,255,0.55)"),
                ),
                hovertemplate=(
                    "<b>Batch ID</b>: %{customdata[1]}<br>"
                    "<b>Pattern Number</b>: %{customdata[4]}<br>"
                    "<b>Grade</b>: %{customdata[3]}<br>"
                    "<b>Risk Level</b>: %{customdata[6]}<br>"
                    "<b>Recommendation</b>: %{customdata[5]}<extra></extra>"
                ),
            )
        )

    if selected_idx is not None and 0 <= int(selected_idx) < len(plot_df):
        row = plot_df.iloc[int(selected_idx)]
        fig.add_trace(
            go.Scatter(
                x=[row["_x"]],
                y=[row["_y"]],
                mode="markers",
                name="Selected casting",
                marker=dict(size=16, color="rgba(255,255,255,0)", line=dict(width=3, color=ACCENT)),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    x_title = str(plot_df["_x_title"].iloc[0])
    y_title = str(plot_df["_y_title"].iloc[0])
    fig.update_layout(
        **_layout(
            title="Cluster Analysis",
            height=500,
            dragmode="zoom",
            margin=dict(l=72, r=42, t=96, b=58),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
            xaxis_title=x_title,
            yaxis_title=y_title,
            hoverlabel=dict(bgcolor="#21262d", bordercolor="#30363d", font=dict(color="#e6edf3", size=12)),
        )
    )
    return fig


def fig_anomaly_scatter(df):
    if not PLOTLY_AVAILABLE or len(df) == 0:
        return _require_plotly("Anomaly vs Defect Probability")
    plot_df = df.copy()
    plot_df["anomaly_score"] = _derived_score(plot_df, "anomaly_score")
    plot_df["defect_prob"] = _derived_score(plot_df, "defect_prob")
    fig = px.scatter(
        plot_df,
        x="anomaly_score",
        y="defect_prob",
        color="anomaly_score",
        color_continuous_scale="RdYlGn_r",
        opacity=0.45,
        hover_data=[c for c in ["risk_level", "recommendation", "cluster"] if c in plot_df.columns],
    )
    fig.add_vline(x=0.60, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_hline(y=0.50, line_dash="dash", line_color="white", opacity=0.5)
    fig.add_annotation(x=0.85, y=0.9, text="Critical zone", showarrow=False, font=dict(color=DANGER))
    fig.update_layout(**_layout(title="Anomaly vs Defect Probability", height=420))
    return fig


def fig_feature_importance(feat_names, importance):
    if not PLOTLY_AVAILABLE:
        return _require_plotly("Defect-Driving Features")
    if not feat_names or importance is None or len(feat_names) == 0:
        return fig_horizontal_bar(["No features"], [0], title="Defect-Driving Features")
    fi = pd.Series(importance, index=feat_names).sort_values(ascending=True).tail(20)
    colors = [FEATURE_GROUP_COLORS[_feature_group(c)] for c in fi.index]
    return fig_horizontal_bar(list(fi.index), list(fi.values), colors=colors, title="Defect-Driving Features", x_title="Importance")


def fig_feature_importance_from_df(df: pd.DataFrame, target: str = "defect_prob"):
    if not PLOTLY_AVAILABLE or len(df) == 0:
        return fig_feature_importance([], [])
    tgt = target if target in df.columns else ("defect" if "defect" in df.columns else None)
    nums = _numeric_cols(df, exclude={tgt} if tgt else set())
    if not nums or tgt is None:
        means = df[_numeric_cols(df)].mean().sort_values(ascending=False).head(10)
        return fig_horizontal_bar(list(means.index), list(means.values), title="Top Numeric Features (mean)", x_title="Mean value")
    corr = df[nums].corrwith(pd.to_numeric(df[tgt], errors="coerce")).abs().fillna(0).sort_values(ascending=False).head(20)
    return fig_feature_importance(list(corr.index), list(corr.values))


def fig_heatmap(corr: pd.DataFrame, title="Correlation"):
    if not PLOTLY_AVAILABLE:
        return _require_plotly(title)
    if corr is None or corr.empty:
        return _require_plotly(title)
    labels = [str(c).replace("feat_", "").replace("_", " ").title()[:18] for c in corr.columns]
    n = len(corr)
    size = max(420, 28 * n)
    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=labels,
            y=labels,
            colorscale="RdBu",
            zmid=0,
            text=np.round(corr.values, 2),
            texttemplate="%{text}",
            textfont={"size": 9},
        )
    )
    fig.update_layout(**_layout(title=title, height=size, xaxis_tickangle=-35))
    return fig


def fig_heatmap_from_df(df: pd.DataFrame, cols: list[str] | None = None, title="Correlation"):
    if cols is None:
        cols = _numeric_cols(df)[:12]
    if len(cols) < 2:
        cols = _numeric_cols(df)
    if len(cols) < 2:
        return fig_heatmap(pd.DataFrame([[1.0]], columns=["x"], index=["x"]), title)
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    return fig_heatmap(sub.corr(), title)


def fig_stacked_risk(df):
    if not PLOTLY_AVAILABLE or len(df) == 0:
        return _require_plotly("Risk vs Recommendation")
    plot_df = df.copy()
    plot_df["_risk_category"] = risk_category_series(plot_df)
    if "recommendation" not in plot_df.columns:
        plot_df["recommendation"] = "PROCEED"
    ct = pd.crosstab(plot_df["_risk_category"], plot_df["recommendation"])
    ct = ct.reindex(RISK_DISPLAY_ORDER, fill_value=0)
    fig = go.Figure()
    for col in ct.columns:
        fig.add_trace(
            go.Bar(
                name=col,
                x=ct.index,
                y=ct[col],
                marker_color=REC_COLORS.get(col, ACCENT),
            )
        )
    fig.update_layout(**_layout(barmode="stack", title="Risk vs Recommendation", height=360))
    return fig


def fig_process_stability_bars(scores: dict):
    if not PLOTLY_AVAILABLE:
        return _require_plotly("Process Stability Scores")
    if not scores:
        scores = {"Process stability": 0.5}
    labels = list(scores.keys())
    vals = list(scores.values())
    colors = [SUCCESS if v >= 0.7 else WARNING if v >= 0.5 else DANGER for v in vals]
    return fig_horizontal_bar(labels, [v * 100 for v in vals], colors=colors, title="Process Stability Scores", x_title="Score (%)")


def fig_radar(labels, values_a, values_b, name_a="Selected", name_b="Fleet mean"):
    if not PLOTLY_AVAILABLE:
        return _require_plotly("Chemistry Profile Comparison")
    if not labels:
        labels, values_a, values_b = ["Feature A"], [0.5], [0.5]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=values_a, theta=labels, fill="toself", name=name_a, line_color=ACCENT))
    fig.add_trace(go.Scatterpolar(r=values_b, theta=labels, fill="toself", name=name_b, line_color=WARNING, opacity=0.5))
    fig.update_layout(**_layout(title="Chemistry Profile Comparison", height=400), polar=dict(radialaxis=dict(visible=True)))
    return fig


def fig_chemistry_comparison(df: pd.DataFrame, subset: pd.DataFrame, chem_cols: list):
    if not PLOTLY_AVAILABLE or len(df) == 0:
        return _require_plotly("Chemistry: Selected vs Fleet")
    if not chem_cols:
        chem_cols = _numeric_cols(df)[:6]
    if not chem_cols:
        return fig_horizontal_bar(["No chemistry"], [0], title="Chemistry: Selected vs Fleet")
    fleet = df[chem_cols].apply(pd.to_numeric, errors="coerce").mean()
    sel = subset[chem_cols].apply(pd.to_numeric, errors="coerce").mean()
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Fleet mean", x=chem_cols, y=fleet.values, marker_color=WARNING))
    fig.add_trace(go.Bar(name="Selected", x=chem_cols, y=sel.values, marker_color=ACCENT))
    fig.update_layout(**_layout(barmode="group", title="Chemistry: Selected vs Fleet", height=380))
    return fig


def fig_fleet_quality_gauge(health_score: float):
    return fig_gauge(health_score, "Fleet Quality Score", max_val=1.0)


def fig_grouped_counts(df: pd.DataFrame, col: str, title: str):
    if not PLOTLY_AVAILABLE or len(df) == 0:
        return _require_plotly(title)
    if col not in df.columns:
        col = "risk_level" if "risk_level" in df.columns else _numeric_cols(df)[0] if _numeric_cols(df) else df.columns[0]
    vc = df[col].value_counts()
    return fig_horizontal_bar(list(vc.index.astype(str)), list(vc.values), title=title)


def fig_risk_distribution(df: pd.DataFrame):
    if not PLOTLY_AVAILABLE:
        return _require_plotly("Risk Level Distribution")
    counts = risk_category_counts(df)
    labels = RISK_DISPLAY_ORDER
    values = [counts[level] for level in labels]
    colors = [RISK_COLORS[level] for level in labels]
    total = sum(values)
    fig = go.Figure()
    for label, value, color in zip(labels, values, colors):
        fig.add_trace(
            go.Bar(
                y=[label],
                x=[value],
                orientation="h",
                name=label,
                marker=dict(color=color),
                width=0.62,
                text=[str(value)],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="Risk: %{y}<br>Castings: %{x}<extra></extra>",
            )
        )
    fig.update_layout(
        **_layout(
            title=f"Risk Level Distribution ({total:,})",
            height=380,
            barmode="stack",
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis_title="Castings",
            yaxis_title="",
        )
    )
    return fig


def fig_severity_donut(df: pd.DataFrame):
    order = ["NORMAL", "LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL"]
    if "anomaly_severity" in df.columns:
        sev_c = df["anomaly_severity"].value_counts()
        present = [s for s in order if s in sev_c.index]
        if not present:
            present = list(sev_c.index.astype(str))
        return fig_donut(present, [sev_c.get(s, 0) for s in present], [SEVERITY_COLORS.get(s, ACCENT) for s in present], title="Anomaly Severity")
    if "anomaly_score" in df.columns:
        scores = _derived_score(df, "anomaly_score")
        bins = pd.cut(scores, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=order, include_lowest=True)
        sev_c = bins.value_counts()
        return fig_donut(list(sev_c.index.astype(str)), list(sev_c.values), [SEVERITY_COLORS.get(str(s), ACCENT) for s in sev_c.index], title="Anomaly Severity (derived)")
    return fig_donut(["All batches"], [len(df)], [ACCENT], title="Batch Count")


def fig_defect_trend(df: pd.DataFrame):
    window = min(50, max(2, len(df)))
    min_p = min(2, len(df))
    if "defect" in df.columns and pd.api.types.is_numeric_dtype(df["defect"]):
        rolling = pd.to_numeric(df["defect"], errors="coerce").rolling(window, min_periods=min_p).mean()
        y = rolling.bfill().ffill().values
        return fig_trend_line(np.arange(len(df)), y, title="Defect rate trend (rolling avg)", y_title="Defect rate")
    if "defect_prob" in df.columns:
        rolling = _derived_score(df, "defect_prob").rolling(window, min_periods=min_p).mean()
        y = rolling.bfill().ffill().values
        return fig_trend_line(np.arange(len(df)), y, title="Defect probability trend (rolling avg)", y_title="Defect prob")
    nums = _numeric_cols(df)
    if nums:
        rolling = df[nums[0]].rolling(50, min_periods=5).mean()
        return fig_trend_line(rolling.index, rolling.values, title=f"{nums[0]} trend (50-batch rolling avg)", y_title=nums[0])
    return fig_trend_line(np.arange(len(df)), np.zeros(len(df)), title="Batch index trend", y_title="Value")


def fig_cluster_donut(df: pd.DataFrame):
    if "cluster" in df.columns:
        clusters = sorted(df["cluster"].dropna().unique())
        counts = df["cluster"].value_counts().sort_index()
        palette = [ACCENT, WARNING, DANGER, SUCCESS, CRITICAL]
        return fig_donut(
            [f"C{c}" for c in clusters],
            [counts.get(c, 0) for c in clusters],
            palette[: len(clusters)],
            title="Cluster Distribution",
        )
    return fig_risk_distribution(df)


def fig_defect_driving(df: pd.DataFrame):
    from dashboard.column_mapping import chemistry_columns

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
        ]
        if c in df.columns
    ]
    chem = list(dict.fromkeys(chem))
    top = None
    if chem:
        if "defect" in df.columns and df["defect"].nunique() >= 2:
            g = df.groupby("defect")[chem].mean()
            top = (g.loc[1] - g.loc[0]).abs().sort_values(ascending=False).head(10)
        elif "defect_prob" in df.columns:
            top = df[chem].corrwith(_derived_score(df, "defect_prob")).abs().sort_values(ascending=False).head(10)
    if top is not None and len(top) > 0:
        return fig_horizontal_bar(list(top.index), list(top.values), title="Top defect-driving parameters")
    return fig_feature_importance_from_df(df)


def fig_chemistry_radar(df: pd.DataFrame):
    from dashboard.column_mapping import chemistry_columns

    chem = chemistry_columns(df)
    if len(chem) < 2:
        chem = _numeric_cols(df)[:6]
    if not chem:
        return fig_radar(["Process"], [0.5], [0.5])
    radar_labels = [c.replace("_", " ").upper() for c in chem[:6]]
    fleet_vals = [float(pd.to_numeric(df[c], errors="coerce").mean()) for c in chem[:6]]
    normed = []
    for c, v in zip(chem[:6], fleet_vals):
        col = pd.to_numeric(df[c], errors="coerce")
        lo, hi = col.min(), col.max()
        normed.append((v - lo) / (hi - lo + 1e-9))
    return fig_radar(radar_labels, normed, [0.5] * len(normed), name_a="Fleet (normalized)", name_b="Midpoint")
