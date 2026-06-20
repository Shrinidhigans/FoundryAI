"""
Upload normalization, feature alignment, and safe numeric utilities.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from dashboard.column_mapping import clean_col_name, normalize_column_name
from dashboard.config.features import (
    UPLOAD_COLUMN_ALIASES,
    get_expected_features,
    get_training_fill_values,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class SchemaValidationReport:
    uploaded_columns: int = 0
    expected_columns: int = 0
    matched_columns: List[str] = field(default_factory=list)
    missing_columns: List[str] = field(default_factory=list)
    extra_columns: List[str] = field(default_factory=list)
    duplicate_columns_removed: List[str] = field(default_factory=list)
    auto_filled_columns: List[str] = field(default_factory=list)
    renamed_columns: Dict[str, str] = field(default_factory=dict)
    dtype_mismatches: List[str] = field(default_factory=list)
    nan_columns: List[str] = field(default_factory=list)
    infinite_columns: List[str] = field(default_factory=list)
    duplicate_columns: List[str] = field(default_factory=list)
    generated_columns: int = 0
    schema_status: str = "UNKNOWN"
    stage: str = "unknown"

    @property
    def success(self) -> bool:
        return len(self.missing_columns) == 0

    @property
    def had_auto_fill(self) -> bool:
        return len(self.auto_filled_columns) > 0

    @property
    def will_auto_fill(self) -> bool:
        return len(self.missing_columns) > 0 or len(self.auto_filled_columns) > 0


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


def normalize_dataframe_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names only — never drop columns (safe for header-only templates)."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    if len(out.columns) and out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    out.columns = [normalize_column_name(c) for c in out.columns]
    return out


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Dedupe columns, normalize names, drop empty rows/cols."""
    if df is None:
        return pd.DataFrame()

    df = df.copy()
    dup_mask = df.columns.duplicated()
    if dup_mask.any():
        dup_names = list(df.columns[dup_mask].unique())
        log.info("Removing duplicate columns: %s", dup_names)
        df = df.loc[:, ~dup_mask]

    df.columns = [normalize_column_name(c) for c in df.columns]
    if len(df) == 0:
        return df
    df = df.dropna(how="all").dropna(axis=1, how="all")
    return df


def apply_upload_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Rename uploaded columns to model / training names."""
    df = df.copy()
    renamed: Dict[str, str] = {}
    for col in list(df.columns):
        key = clean_col_name(col)
        target = UPLOAD_COLUMN_ALIASES.get(key)
        if target and target != col and target not in df.columns:
            df = df.rename(columns={col: target})
            renamed[col] = target
    if renamed:
        log.info("Renamed upload columns: %s", renamed)
    return df


def _bridge_chemistry_for_stage2(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure stage-2 feature functions see c_/si_/mn_/s_/p__ aliases."""
    df = df.copy()
    bridges = {
        "c_": ["c", "carbon"],
        "si_": ["si", "silicon"],
        "s_": ["s", "sulfur", "sulphur"],
        "mn_": ["mn_1", "mn", "manganese"],
        "p__": ["p", "phosphorus", "phos"],
        "mg_recovery_": ["mg_recovery", "mg_recovery_pct"],
        "ce_": ["ce", "carbon_equivalent"],
        "n__": ["n"],
        "al__": ["al", "al_1"],
        "cr_": ["cr"],
        "mo_": ["mo"],
    }
    for canonical, sources in bridges.items():
        if canonical in df.columns:
            continue
        for src in sources:
            if src in df.columns:
                df[canonical] = safe_numeric_series(df[src])
                break
    return df


def run_stage2_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply stage-2 metallurgical features (same as training pipeline)."""
    try:
        from stage2_feature_engineering import (
            add_carbon_equivalent,
            add_ce_zone_flags,
            add_c_si_ratio,
            add_chemistry_stability,
            add_fsm_efficiency,
            add_gas_risk_index,
            add_graphitization_index,
            add_heel_charge_ratio,
            add_mg_recovery_features,
            add_mn_s_ratio,
            add_oxidation_risk,
            add_shrinkage_risk,
            add_sulfur_risk,
            add_temperature_features,
        )
    except ImportError as exc:
        log.warning("stage2_feature_engineering unavailable: %s", exc)
        return df

    df = _bridge_chemistry_for_stage2(df)
    fns = [
        add_carbon_equivalent,
        add_ce_zone_flags,
        add_c_si_ratio,
        add_mn_s_ratio,
        add_sulfur_risk,
        add_mg_recovery_features,
        add_temperature_features,
        add_oxidation_risk,
        add_graphitization_index,
        add_shrinkage_risk,
        add_gas_risk_index,
        add_chemistry_stability,
        add_fsm_efficiency,
        add_heel_charge_ratio,
    ]
    for fn in fns:
        df = fn(df)
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if feat_cols:
        df[feat_cols] = df[feat_cols].apply(lambda col: safe_numeric_series(col)).fillna(0)
    return df


def align_ml_features(
    df: pd.DataFrame,
    features: Optional[List[str]] = None,
    fill_values: Optional[Dict[str, float]] = None,
) -> tuple[pd.DataFrame, SchemaValidationReport]:
    """
    Build model input matrix with exact training feature names and order.
    Missing columns are created and filled from training medians.
    """
    features = list(features or get_expected_features())
    fill_values = fill_values or get_training_fill_values()
    report = SchemaValidationReport(
        uploaded_columns=len(df.columns),
        expected_columns=len(features),
    )

    if not features:
        return pd.DataFrame(index=df.index), report

    work = df.copy()
    report.matched_columns = [c for c in features if c in work.columns]
    report.extra_columns = [c for c in work.columns if c not in features]
    report.missing_columns = [c for c in features if c not in work.columns]

    for col in report.missing_columns:
        fill = float(fill_values.get(col, 0.0))
        work[col] = fill
        report.auto_filled_columns.append(col)

    X = pd.DataFrame(index=work.index)
    for col in features:
        series = work[col] if col in work.columns else pd.Series(fill_values.get(col, 0.0), index=work.index)
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        X[col] = safe_numeric_series(series)

    X = X.fillna({c: float(fill_values.get(c, 0.0)) for c in features})
    report.missing_columns = []  # all resolved after align
    return X, report


def align_features_to_training_schema(
    df: pd.DataFrame,
    feature_names: List[str],
    fill_values: Optional[Dict[str, float]] = None,
    stage: str = "inference",
) -> tuple[pd.DataFrame, SchemaValidationReport]:
    """
    Strictly align generated inference features to a saved training schema.

    The returned frame contains exactly feature_names, in the original training
    order. Missing fields are filled from training medians/defaults, unexpected
    fields are dropped, and all model inputs are numeric float columns.
    """
    fill_values = fill_values or get_training_fill_values()
    features = [str(c) for c in feature_names]
    work = df.copy()
    duplicate_columns = [str(c) for c in work.columns[work.columns.duplicated()].unique()]
    if duplicate_columns:
        work = work.loc[:, ~work.columns.duplicated()]

    report = SchemaValidationReport(
        uploaded_columns=len(df.columns),
        expected_columns=len(features),
        matched_columns=[c for c in features if c in work.columns],
        missing_columns=[c for c in features if c not in work.columns],
        extra_columns=[c for c in work.columns if c not in features],
        duplicate_columns=duplicate_columns,
        duplicate_columns_removed=duplicate_columns,
        generated_columns=len(work.columns),
        stage=stage,
    )

    aligned = pd.DataFrame(index=work.index)
    for col in features:
        if col in work.columns:
            raw = work[col]
            if isinstance(raw, pd.DataFrame):
                raw = raw.iloc[:, 0]
            series = safe_numeric_series(raw)
        else:
            series = pd.Series(float(fill_values.get(col, 0.0)), index=work.index)
            report.auto_filled_columns.append(col)

        before_na = int(series.isna().sum())
        series = series.replace([np.inf, -np.inf], np.nan)
        if before_na or int(series.isna().sum()):
            report.nan_columns.append(col)
        if np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).any():
            report.infinite_columns.append(col)

        fill = float(fill_values.get(col, 0.0))
        aligned[col] = series.fillna(fill).astype(float)

    report.schema_status = "MATCHED" if list(aligned.columns) == features else "MISMATCHED"
    return aligned, report


def validate_inference_schema(
    X: pd.DataFrame,
    feature_names: List[str],
    report: Optional[SchemaValidationReport] = None,
    stage: str = "inference",
    strict_missing: bool = False,
) -> SchemaValidationReport:
    """Validate an aligned inference matrix before any saved transformer/model runs."""
    features = [str(c) for c in feature_names]
    report = report or SchemaValidationReport(
        uploaded_columns=len(X.columns),
        expected_columns=len(features),
        stage=stage,
    )
    report.stage = stage
    report.generated_columns = len(X.columns)

    if X.columns.duplicated().any():
        report.duplicate_columns = [str(c) for c in X.columns[X.columns.duplicated()].unique()]
    actual = list(map(str, X.columns))
    missing_after_alignment = [c for c in features if c not in actual]
    extra_after_alignment = [c for c in actual if c not in features]
    non_numeric = [c for c in actual if not pd.api.types.is_numeric_dtype(X[c])]
    report.dtype_mismatches = sorted(set(report.dtype_mismatches + non_numeric))

    numeric = X.apply(pd.to_numeric, errors="coerce")
    report.nan_columns = sorted(set(report.nan_columns + [c for c in numeric.columns if numeric[c].isna().any()]))
    report.infinite_columns = sorted(
        set(report.infinite_columns + [c for c in numeric.columns if np.isinf(numeric[c].to_numpy()).any()])
    )

    ok = (
        len(actual) == len(features)
        and actual == features
        and not report.duplicate_columns
        and not report.dtype_mismatches
        and not report.infinite_columns
        and (not strict_missing or not report.auto_filled_columns)
        and not missing_after_alignment
        and not extra_after_alignment
    )
    report.schema_status = "MATCHED" if ok else "MISMATCHED"
    if ok:
        log.info("\u2705 feature schema matched")
    else:
        log.error(
            "[%s] feature schema mismatch: expected=%s generated=%s missing=%s extra=%s dtype=%s inf=%s",
            stage,
            len(features),
            len(actual),
            missing_after_alignment or report.auto_filled_columns[:20],
            extra_after_alignment[:20],
            report.dtype_mismatches[:20],
            report.infinite_columns[:20],
        )
    return report


def validate_upload_schema(df: pd.DataFrame) -> SchemaValidationReport:
    """Validate uploaded columns against EXPECTED_FEATURES (before auto-fill)."""
    features = list(get_expected_features())
    report = SchemaValidationReport(
        uploaded_columns=len(df.columns),
        expected_columns=len(features),
    )
    report.matched_columns = [c for c in features if c in df.columns]
    report.missing_columns = [c for c in features if c not in df.columns]
    report.extra_columns = [c for c in df.columns if c not in features]
    return report


def log_schema_debug(stage: str, df: pd.DataFrame, report: Optional[SchemaValidationReport] = None) -> None:
    """Debug logging for upload / prediction pipeline."""
    log.info("[%s] rows=%s cols=%s", stage, len(df), len(df.columns))
    log.info("[%s] uploaded columns (%s): %s", stage, len(df.columns), list(df.columns)[:40])
    if report:
        log.info("[%s] expected=%s matched=%s auto_filled=%s", stage, report.expected_columns, len(report.matched_columns), len(report.auto_filled_columns))
        if report.missing_columns:
            log.info("[%s] missing before fill: %s", stage, report.missing_columns[:25])
        if report.auto_filled_columns:
            log.info("[%s] auto-filled: %s", stage, report.auto_filled_columns[:25])
        if report.renamed_columns:
            log.info("[%s] renamed: %s", stage, report.renamed_columns)
        extra_count = 0 if report.schema_status == "MATCHED" and len(df.columns) == report.expected_columns else len(report.extra_columns)
        log.info(
            "\nTRAINING FEATURES: %s\nGENERATED FEATURES: %s\nMISSING FEATURES: %s\nEXTRA FEATURES: %s\nSCHEMA STATUS: %s",
            report.expected_columns,
            report.generated_columns or len(report.matched_columns),
            len(report.auto_filled_columns or report.missing_columns),
            extra_count,
            report.schema_status,
        )
