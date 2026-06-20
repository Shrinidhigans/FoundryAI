"""Canonical column names and aliases for melting / chemistry data."""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

# Canonical name -> accepted aliases (after basic clean_col normalization)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "c_": ["c_", "c", "carbon", "carbon_pct", "c_pct"],
    "si_": ["si_", "si", "silicon", "si_pct"],
    "s_": ["s_", "s", "sulfur", "sulphur", "s_pct"],
    "mn_": ["mn_", "mn", "mn_1", "manganese", "mn_pct"],
    "p__": ["p__", "p_", "p", "phosphorus", "phos", "p_pct"],
    "ce": ["ce", "carbon_equivalent"],
    "mg_recovery_": ["mg_recovery_", "mg_recovery", "mg_recovery_pct", "mg_recovery_percent"],
    "mg_": ["mg_", "mg", "magnesium"],
    "tapping_temp": ["tapping_temp", "tap_temp", "tapping_temperature"],
    "pouring_temp": ["pouring_temp", "pour_temp", "pouring_temperature"],
    "defect": ["defect", "defective", "is_defect", "quality_defect"],
}


def clean_col_name(col: str) -> str:
    """Normalize a column name without stripping meaningful trailing underscores (e.g. c_, mn_)."""
    col = str(col).strip().lower()
    col = re.sub(r"[%\n\r]", "", col)
    col = re.sub(r"[^a-z0-9_\s]", "", col)
    col = re.sub(r"\s+", "_", col)
    return col.strip()


def normalize_column_name(col: str) -> str:
    """Public alias used by upload + template validation (same rules as clean_col_name)."""
    return clean_col_name(col)


def apply_column_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add canonical chemistry/process columns as copies for charts and rules.

    Original column names are preserved so ML models trained on stage-2 names
    (e.g. c, si, s) continue to receive the expected features.
    """
    df = df.copy()

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue
        for alias in aliases:
            if alias in df.columns:
                df[canonical] = df[alias]
                break
    return df


def chemistry_columns(df: pd.DataFrame) -> List[str]:
    """Numeric chemistry columns present in dataframe."""
    candidates = ["c_", "si_", "mn_", "p__", "s_", "ce", "feat_ce_calculated"]
    return [c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def process_feature_columns(df: pd.DataFrame) -> List[str]:
    """Columns commonly used for correlation / importance fallbacks."""
    cols = chemistry_columns(df)
    extras = [
        "tapping_temp",
        "pouring_temp",
        "feat_temp_loss",
        "feat_chemistry_instability",
        "feat_shrinkage_risk_index",
        "feat_gas_risk_index",
        "mg_recovery_",
        "anomaly_score",
        "defect_prob",
    ]
    for c in extras:
        if c in df.columns and c not in cols and pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def missing_reason(df: pd.DataFrame, required: List[str], label: str = "data") -> Optional[str]:
    """Human-readable reason when required columns are absent."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        return f"No {label}: missing columns {', '.join(missing)}."
    if len(df) == 0:
        return f"No {label}: dataframe is empty."
    return None
