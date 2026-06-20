"""Safe dataframe preparation for analytics and plotting."""

from __future__ import annotations

from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


def safe_numeric_series(series: pd.Series) -> pd.Series:
    """Coerce a single Series to numeric; never pass a DataFrame."""
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype(str).str.strip()
    s = s.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    sentinels = ["-", "na", "null", "none", "n/a", "nan", ""]
    s = s.replace(sentinels, np.nan)
    return pd.to_numeric(s, errors="coerce")


def remove_inf_nan(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    num = out.select_dtypes(include=[np.number]).columns
    if len(num):
        out[num] = out[num].replace([np.inf, -np.inf], np.nan)
    return out


def ensure_required_columns(
    df: pd.DataFrame,
    cols: Iterable[str],
    fill: float = 0.0,
) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            out[col] = fill
    return out


def safe_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize dataframe for analytics:
    - drop duplicate column labels
    - replace inf
    - coerce numeric-like columns
    - median-fill numeric NaNs
  """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    out = df.copy()
    if out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]

    out = remove_inf_nan(out)

    skip = {"qa_summary", "recommendation", "risk_level", "anomaly_severity", "risk_factors"}
    for col in list(out.columns):
        if col in skip:
            continue
        block = out[col]
        if isinstance(block, pd.DataFrame):
            block = block.iloc[:, 0]
            out[col] = block
        if pd.api.types.is_numeric_dtype(block):
            out[col] = pd.to_numeric(block, errors="coerce")
            continue
        converted = safe_numeric_series(block)
        if converted.notna().mean() >= 0.25:
            out[col] = converted

    num_cols = out.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        med = out[col].median()
        out[col] = out[col].fillna(med if pd.notna(med) else 0.0)

    if "cluster" in out.columns:
        out["cluster"] = pd.to_numeric(out["cluster"], errors="coerce").fillna(0).astype(int)
    if "anomaly_score" not in out.columns:
        out["anomaly_score"] = 0.0
    if "defect_prob" not in out.columns:
        out["defect_prob"] = 0.0
    if "pca_pc1" not in out.columns:
        out["pca_pc1"] = 0.0
    if "pca_pc2" not in out.columns:
        out["pca_pc2"] = 0.0

    return out


def safe_for_plotting(df: pd.DataFrame, min_rows: int = 1) -> pd.DataFrame:
    """Ensure dataframe is safe for Plotly; pad PCA if single row."""
    out = safe_dataframe(df)
    if len(out) < min_rows:
        return out
    if len(out) == 1:
        if "pca_pc1" not in out.columns or out["pca_pc1"].isna().all():
            out["pca_pc1"] = 0.0
        if "pca_pc2" not in out.columns or out["pca_pc2"].isna().all():
            out["pca_pc2"] = 0.0
    return out


def numeric_columns(df: pd.DataFrame, exclude: Optional[set] = None) -> List[str]:
    exclude = exclude or set()
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c]):
            cols.append(c)
    return cols


def safe_probability(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return float(min(1.0, max(0.0, v)))
