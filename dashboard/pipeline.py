"""Data loading and ML pipeline for the dashboard."""

from __future__ import annotations

import io
import os
import pickle
import re

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer

from dashboard.column_mapping import (
    COLUMN_ALIASES,
    apply_column_aliases,
    clean_col_name,
    chemistry_columns,
)
from dashboard.config.features import (
    get_expected_features,
    get_isolation_forest_feature_names,
    get_pca_feature_names,
    sample_template_row,
)
from dashboard.column_mapping import normalize_column_name
from dashboard.config.features import UPLOAD_COLUMN_ALIASES
from dashboard.preprocessing import (
    align_features_to_training_schema,
    align_ml_features,
    apply_upload_aliases,
    log_schema_debug,
    normalize_dataframe,
    normalize_dataframe_headers,
    run_stage2_feature_engineering,
    validate_inference_schema,
    validate_upload_schema,
)
from dashboard.risk_scoring import DECISION_LOGIC_VERSION, enrich_dataframe, ensure_unified_decisions

import logging

log = logging.getLogger(__name__)


class InferenceSchemaError(ValueError):
    """Raised when uploaded data cannot satisfy the trained ML schema."""

    def __init__(self, message: str, report=None, stage: str = "inference"):
        super().__init__(message)
        self.report = report
        self.stage = stage


def _normalize_unit_interval(series: pd.Series, name: str = "series") -> pd.Series:
    """
    Ensure values behave as probabilities/scores on [0, 1].

    Common failure mode: spreadsheets store defect probabilities as percentages
    (0–100); thresholds in risk scoring expect fractions (0–1). Unguarded inputs
    make every batch look CRITICAL / STOP.

    If a few rows are on a 0–100 scale alongside mostly [0, 1] values, clipping is
    used instead of dividing the whole column.
    """
    s = pd.to_numeric(series, errors="coerce")
    mask = s.notna()
    if not bool(mask.any()):
        return s.fillna(0.0)
    vals = s[mask]
    mx = float(vals.max())
    mn = float(vals.min())
    if mx > 1.0:
        frac_over_1 = float((vals > 1.0).mean())
        if mx <= 100.0 and frac_over_1 >= 0.40:
            s.loc[mask] = vals / 100.0
            log.warning(
                "%s: treating as 0–100 scale (max=%.4f, %.0f%% rows >1) → divide by 100",
                name,
                mx,
                100 * frac_over_1,
            )
        elif mx <= 100.0:
            s.loc[mask] = vals.clip(0.0, 1.0)
            log.warning(
                "%s: values exceed 1.0 (max=%.4f) but column is mostly≤1 — clipping to [0,1]",
                name,
                mx,
            )
        else:
            denom = mx - mn + 1e-9
            s.loc[mask] = (vals - mn) / denom
            log.warning("%s: max=%.4f — min-max scaled to [0,1]", name, mx)
    return s.fillna(0.0).clip(0.0, 1.0)


MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MASTER_TEMPLATE_NAME = "melting_cleaned_template.xlsx"
MASTER_TEMPLATE_PATH = os.path.join(PROJECT_ROOT, MASTER_TEMPLATE_NAME)
TEMPLATE_VALIDATION_VERSION = "semantic-v2"

# Semantic groups: any variant maps to the same required template field
TEMPLATE_SEMANTIC_GROUPS: dict[str, tuple[str, ...]] = {
    "pouring_wt": ("pouring_wt", "pouring_wt", "pouring_weight"),
    "furnace_on_time": ("furnace_on_time", "furnace_on", "furnace_on_time_min"),
    "furnace_off_time": ("furnace_off_time", "furnace_off", "furnace_off_time_min"),
    "tapping_temp": ("tapping_temp", "tapping_temperature", "tap_temp"),
    "pouring_temp": ("pouring_temp", "pouring_temperature", "pour_temp"),
    "mg_recovery": ("mg_recovery", "mg_recovery_", "mg_recovery_pct", "magnesium_recovery"),
    "defect": ("defect", "defected", "defective", "is_defect", "quality_defect"),
}

LEAKAGE_COLS = ["sulphur", "phos", "crom", "copper", "heel_metal", "nickel", "mg"]

ALWAYS_EXCL = set(
    LEAKAGE_COLS
    + [
        "defect",
        "cluster",
        "dbscan_cluster",
        "pca_pc1",
        "pca_pc2",
        "pc1",
        "pc2",
        "anomaly_score",
        "anomaly_flag",
        "anomaly_severity",
        "anomaly_level",
        "iso_flag",
        "lof_flag",
        "risk_level",
        "recommendation",
        "qa_summary",
        "final_risk_score",
        "risk_confidence",
        "risk_factors",
        "defect_prob",
        "defect_pred",
    ]
)

# Explicit process / chemistry columns always coerced to numeric
PRIORITY_NUMERIC = [
    "ce",
    "tapping_temp",
    "pouring_temp",
    "feat_temp_loss",
    "defect_prob",
    "anomaly_score",
    "mg",
    "si",
    "carbon",
    "sulfur",
    "phosphorus",
    "c",
    "s",
    "p",
    "mn",
    "mn_1",
    "mg_recovery",
    "pouring_time_sec",
    "graphite",
    "fsm",
]


def _clean_col(col):
    return clean_col_name(col)


def _safe_numeric(s):
    sentinels = ["-", "na", "null", "none", "n/a", "nan", "", " "]
    s = s.astype(str).str.strip().str.lower().replace(sentinels, np.nan)
    return pd.to_numeric(s, errors="coerce")


def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _column_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a single Series even when duplicate column labels exist."""
    data = df[col]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    return data


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicate column names (keep first non-all-null column)."""
    if not df.columns.duplicated().any():
        return df
    out = {}
    for col in dict.fromkeys(df.columns):
        block = df.loc[:, df.columns == col]
        if isinstance(block, pd.Series):
            out[col] = block
        else:
            chosen = block.iloc[:, 0]
            for i in range(block.shape[1]):
                s = block.iloc[:, i]
                if s.notna().any():
                    chosen = s
                    break
            out[col] = chosen
    return pd.DataFrame(out, index=df.index)


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert process/chemistry columns to numeric before any aggregation."""
    df = df.copy()

    if "defect" in df.columns:
        df["defect"] = pd.to_numeric(df["defect"], errors="coerce").fillna(0).astype(int)

    priority = set(PRIORITY_NUMERIC)
    for aliases in COLUMN_ALIASES.values():
        priority.update(aliases)

    for col in list(dict.fromkeys(df.columns)):
        if col in ALWAYS_EXCL or col == "defect":
            continue
        series = _column_series(df, col)
        if col in priority or col.startswith("feat_"):
            df[col] = pd.to_numeric(series, errors="coerce")
            continue
        if _is_numeric_series(series):
            df[col] = pd.to_numeric(series, errors="coerce")
            continue
        converted = _safe_numeric(series)
        if converted.notna().mean() >= 0.3:
            df[col] = converted

    return _dedupe_columns(df)


def _fill_numeric_nan(df: pd.DataFrame) -> pd.DataFrame:
    """Median imputation for numeric columns only."""
    df = df.copy()
    for col in list(dict.fromkeys(df.columns)):
        series = _column_series(df, col)
        if not _is_numeric_series(series):
            continue
        med = series.median()
        fill = med if pd.notna(med) else 0.0
        df[col] = series.fillna(fill)
    return _dedupe_columns(df)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [_clean_col(c) for c in df.columns]
    seen: dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    return df


def _ensure_base_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create default ML columns if missing (filled by downstream steps)."""
    df = df.copy()
    defaults: dict[str, object] = {
        "defect_prob": 0.0,
        "anomaly_score": 0.0,
        "cluster": 0,
        "anomaly_severity": "NORMAL",
        "risk_level": "LOW",
        "recommendation": "PROCEED",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default
    if "pca_pc1" not in df.columns and "pc1" in df.columns:
        df["pca_pc1"] = pd.to_numeric(df["pc1"], errors="coerce")
    if "pca_pc2" not in df.columns and "pc2" in df.columns:
        df["pca_pc2"] = pd.to_numeric(df["pc2"], errors="coerce")
    return df


def _engineer_features(df):
    safe = lambda n, d, fill=0: (n / d.replace(0, np.nan)).fillna(fill)

    if all(c in df for c in ["c_", "si_", "p__"]):
        df["feat_ce_calculated"] = df["c_"] + (df["si_"] + df["p__"]) / 3
    elif "ce" in df.columns:
        df["feat_ce_calculated"] = pd.to_numeric(df["ce"], errors="coerce")

    ce = df.get("feat_ce_calculated", df.get("ce", None))
    if ce is not None and _is_numeric_series(ce):
        df["feat_ce_hypo_risk"] = (ce < 4.2).astype(int)
        df["feat_ce_hyper_risk"] = (ce > 4.3).astype(int)

    if "c_" in df and "si_" in df:
        df["feat_c_si_ratio"] = safe(df["c_"], df["si_"])
    if "mn_" in df and "s_" in df:
        df["feat_mn_s_ratio"] = safe(df["mn_"], df["s_"])

    if "s_" in df.columns:
        df["feat_sulfur_risk"] = np.select(
            [df["s_"] > 0.025, df["s_"] > 0.015], [2, 1], default=0
        )

    # -----------------------------------
    # SAFE Mg RECOVERY FEATURE
    # -----------------------------------

    if "mg_recovery_" in df.columns:

        mg_series = (
            df["mg_recovery_"]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        mg_series = pd.to_numeric(
            mg_series,
            errors="coerce"
        )

        # If values are percentages like 35 instead of 0.35
        if mg_series.dropna().mean() > 1:
            mg_series = mg_series / 100

        df["mg_recovery_"] = mg_series

        df["feat_mg_recovery_risk"] = (
            mg_series < 0.40
        ).fillna(False).astype(int)

    else:

        df["feat_mg_recovery_risk"] = 0

    if "tapping_temp" in df and "pouring_temp" in df:
        df["feat_temp_loss"] = df["tapping_temp"] - df["pouring_temp"]
        df["feat_temp_loss_risk"] = (df["feat_temp_loss"] > 80).astype(int)

    score = pd.Series(0.0, index=df.index)
    if ce is not None and _is_numeric_series(ce):
        score += (ce < 4.2).astype(float) * 1.5
    if "si_" in df:
        score += (df["si_"] < 2.0).astype(float)
    df["feat_shrinkage_risk_index"] = score

    gas = pd.Series(0.0, index=df.index)
    if "n__" in df:
        gas += (df["n__"] > 0.008).astype(float) * 2
    df["feat_gas_risk_index"] = gas

    chem = pd.Series(0.0, index=df.index)
    for col, (lo, hi) in [("c_", (3.5, 3.8)), ("si_", (2.0, 2.8)), ("mn_", (0.1, 0.4))]:
        if col in df and _is_numeric_series(df[col]):
            chem += np.maximum(0, lo - df[col]) + np.maximum(0, df[col] - hi)
    df["feat_chemistry_instability"] = chem

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if feat_cols:
        df[feat_cols] = df[feat_cols].fillna(0)
    return df


def _ml_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        c
        for c in df.columns
        if _is_numeric_series(df[c]) and c not in ALWAYS_EXCL
    ]
    if not numeric:
        return pd.DataFrame(index=df.index)
    medians = df[numeric].median(numeric_only=True)
    return df[numeric].fillna(medians).fillna(0)


def _load_pickle_required(path: str, stage: str):
    if not os.path.exists(path):
        raise InferenceSchemaError(f"Missing required trained artifact for {stage}: {path}", stage=stage)
    with open(path, "rb") as f:
        return pickle.load(f)


def _strict_model_matrix(df: pd.DataFrame, feature_names: list[str], stage: str) -> pd.DataFrame:
    X, report = align_features_to_training_schema(df, feature_names, stage=stage)
    report = validate_inference_schema(X, feature_names, report=report, stage=stage)
    log_schema_debug(stage, X, report)
    if report.schema_status != "MATCHED":
        raise InferenceSchemaError(
            "Uploaded data could not be aligned with the trained ML feature schema.",
            report=report,
            stage=stage,
        )
    return X


def _ensure_defect_prob(df: pd.DataFrame) -> pd.DataFrame:
    if "defect_prob" in df.columns:
        existing_raw = pd.to_numeric(_column_series(df, "defect_prob"), errors="coerce")
        if existing_raw.max(skipna=True) > 0:
            df["defect_prob"] = _normalize_unit_interval(existing_raw, "defect_prob (input column)")
            if "defect_pred" not in df.columns:
                df["defect_pred"] = (df["defect_prob"] >= 0.5).astype(int)
            return df

    clf_path = os.path.join(MODELS_DIR, "best_classifier.pkl")
    feat_list = list(get_expected_features())
    if not feat_list:
        raise InferenceSchemaError("No saved classifier feature schema found.", stage="defect_prediction")

    clf = _load_pickle_required(clf_path, "defect_prediction")
    X = _strict_model_matrix(df, feat_list, "defect_prediction")
    proba = np.asarray(clf.predict_proba(X)[:, 1], dtype=float)
    df["defect_prob"] = _normalize_unit_interval(pd.Series(proba), "defect_prob (classifier)")
    df["defect_pred"] = clf.predict(X)
    return df


def _ensure_pca_and_cluster(df: pd.DataFrame) -> pd.DataFrame:
    need_pca = "pca_pc1" not in df.columns or pd.to_numeric(df.get("pca_pc1"), errors="coerce").isna().all()
    need_cluster = "cluster" not in df.columns or pd.to_numeric(df.get("cluster"), errors="coerce").isna().all()
    if not need_pca and not need_cluster:
        df["pca_pc1"] = pd.to_numeric(df["pca_pc1"], errors="coerce").fillna(0)
        df["pca_pc2"] = pd.to_numeric(df["pca_pc2"], errors="coerce").fillna(0)
        df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(0).astype(int)
        return df

    pca = _load_pickle_required(os.path.join(MODELS_DIR, "pca_model.pkl"), "pca")
    km = _load_pickle_required(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "pca")
    scaler = _load_pickle_required(os.path.join(MODELS_DIR, "pca_scaler.pkl"), "pca")
    feature_names = list(get_pca_feature_names())
    if not feature_names:
        raise InferenceSchemaError("No saved PCA feature schema found.", stage="pca")
    X = _strict_model_matrix(df, feature_names, "pca")
    expected = getattr(pca, "n_features_in_", X.shape[1])
    if X.shape[1] != expected:
        raise InferenceSchemaError(f"PCA feature mismatch ({X.shape[1]} vs {expected}).", stage="pca")
    X_pca = pca.transform(scaler.transform(X))
    log.info("\u2705 PCA features matched")
    if need_pca:
        df["pca_pc1"] = X_pca[:, 0]
        df["pca_pc2"] = X_pca[:, 1] if X_pca.shape[1] > 1 else 0.0
    if need_cluster or need_pca:
        df["cluster"] = km.predict(X_pca)
    df["cluster"] = pd.to_numeric(df["cluster"], errors="coerce").fillna(0).astype(int)
    return df


def _ensure_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    if "anomaly_score" in df.columns:
        scores_raw = pd.to_numeric(_column_series(df, "anomaly_score"), errors="coerce")
        if scores_raw.max(skipna=True) > 0:
            df["anomaly_score"] = _normalize_unit_interval(scores_raw.fillna(0), "anomaly_score (input column)")
            if "anomaly_severity" not in df.columns:
                df["anomaly_severity"] = df["anomaly_score"].apply(_anomaly_severity_label)
            if "anomaly_flag" not in df.columns:
                df["anomaly_flag"] = (df["anomaly_score"] >= 0.55).astype(int)
            return df

    iso = _load_pickle_required(os.path.join(MODELS_DIR, "isolation_forest.pkl"), "isolation_forest")
    scaler = _load_pickle_required(os.path.join(MODELS_DIR, "anomaly_scaler.pkl"), "isolation_forest")
    feature_names = list(get_isolation_forest_feature_names())
    if not feature_names:
        raise InferenceSchemaError("No saved IsolationForest feature schema found.", stage="isolation_forest")
    X = _strict_model_matrix(df, feature_names, "isolation_forest")
    expected = getattr(iso, "n_features_in_", X.shape[1])
    if X.shape[1] != expected:
        raise InferenceSchemaError(
            f"IsolationForest feature mismatch ({X.shape[1]} vs {expected}).",
            stage="isolation_forest",
        )
    X_scaled = scaler.transform(X)
    raw = iso.score_samples(X_scaled)
    norm = 1 - (raw - raw.min()) / (raw.max() - raw.min() + 1e-9)
    df["anomaly_score"] = norm
    df["anomaly_flag"] = (iso.predict(X_scaled) == -1).astype(int)
    df["anomaly_severity"] = df["anomaly_score"].apply(_anomaly_severity_label)
    log.info("\u2705 IsolationForest features matched")
    return df


def _anomaly_severity_label(s: float) -> str:
    if s >= 0.80:
        return "CRITICAL"
    if s >= 0.60:
        return "HIGH RISK"
    if s >= 0.40:
        return "MEDIUM RISK"
    if s >= 0.20:
        return "LOW RISK"
    return "NORMAL"


def _prepare_impl(df: pd.DataFrame) -> pd.DataFrame:
    """Core preparation logic (uncached)."""
    if df is None or len(df) == 0:
        return pd.DataFrame()

    df = normalize_dataframe(df)
    pre_report = validate_upload_schema(df)
    log_schema_debug("upload_raw", df, pre_report)

    df = apply_upload_aliases(df)
    df = _dedupe_columns(_standardize_columns(df))
    df = df.dropna(how="all").dropna(axis=1, how="all").drop_duplicates()

    target_col = next((c for c in df.columns if "defect" in c and c != "defect_prob"), None)
    if target_col and target_col != "defect":
        df = df.rename(columns={target_col: "defect"})

    df = _coerce_numeric_columns(df)
    df = apply_column_aliases(df)
    df = _coerce_numeric_columns(df)
    df = _fill_numeric_nan(df)
    df = _ensure_base_columns(df)
    df = run_stage2_feature_engineering(df)
    df = _engineer_features(df)
    df = _fill_numeric_nan(df)
    df = _ensure_defect_prob(df)
    df = _ensure_pca_and_cluster(df)
    df = _ensure_anomaly(df)
    df = enrich_dataframe(df)
    return df


def _df_cache_key(df: pd.DataFrame) -> str:
    h = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return f"{DECISION_LOGIC_VERSION}_{df.shape[0]}_{df.shape[1]}_{hash(h)}"


@st.cache_data(show_spinner=False)
def _cached_prepare(cache_key: str, df_bytes: bytes) -> pd.DataFrame:
    df = pd.read_pickle(io.BytesIO(df_bytes))
    return _prepare_impl(df)


def prepare_dashboard_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Single entry point: normalize columns, coerce numerics, derive ML fields,
    and apply unified risk scoring. All pages must consume only this output.
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()
    buf = io.BytesIO()
    df.copy().to_pickle(buf)
    return ensure_unified_decisions(_cached_prepare(_df_cache_key(df), buf.getvalue()))


def get_feature_importance() -> tuple[list[str], np.ndarray] | None:
    """Load model feature importances or compute correlation fallback."""
    clf_path = os.path.join(MODELS_DIR, "best_classifier.pkl")
    feat_path = os.path.join(MODELS_DIR, "feature_columns.pkl")
    if os.path.exists(clf_path) and os.path.exists(feat_path):
        with open(clf_path, "rb") as f:
            clf = pickle.load(f)
        with open(feat_path, "rb") as f:
            feat_list = pickle.load(f)
        clf_step = clf.named_steps.get("clf", clf) if hasattr(clf, "named_steps") else clf
        if hasattr(clf_step, "feature_importances_"):
            imp = clf_step.feature_importances_
            if len(imp) == len(feat_list):
                return feat_list, imp
        if hasattr(clf_step, "coef_"):
            imp = np.abs(clf_step.coef_[0])
            if len(imp) == len(feat_list):
                return feat_list, imp
    return None


@st.cache_data(show_spinner=False)
def correlation_feature_importance(df: pd.DataFrame) -> tuple[list[str], np.ndarray] | None:
    if "defect" not in df.columns or df["defect"].nunique() < 2:
        target = "defect_prob" if "defect_prob" in df.columns else None
    else:
        target = "defect"
    if target is None:
        return None
    X = _ml_feature_matrix(df)
    if X.shape[1] == 0:
        return None
    corr = X.corrwith(df[target]).abs().fillna(0).sort_values(ascending=False)
    top = corr.head(20)
    return list(top.index), top.values


def models_ready() -> bool:
    return os.path.exists(os.path.join(MODELS_DIR, "best_classifier.pkl"))


def _load_raw_file(file_bytes: bytes, filename: str) -> pd.DataFrame:
    if filename.lower().endswith(".csv"):
        return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)
    return pd.read_excel(io.BytesIO(file_bytes))


def _load_master_template_file() -> pd.DataFrame:
    if not os.path.exists(MASTER_TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found: {MASTER_TEMPLATE_PATH}")
    return pd.read_excel(MASTER_TEMPLATE_PATH)


@st.cache_data(show_spinner=False)
def master_template_dataframe(_cache_version: str = TEMPLATE_VALIDATION_VERSION) -> pd.DataFrame:
    """Master upload template as-is (source of truth for sample downloads)."""
    del _cache_version
    return _load_master_template_file()


@st.cache_data(show_spinner=False)
def master_template_normalized_headers(_cache_version: str = TEMPLATE_VALIDATION_VERSION) -> tuple[str, ...]:
    """Normalized header names from melting_cleaned_template.xlsx (header-only, no row drop)."""
    del _cache_version
    tpl = normalize_dataframe_headers(_load_master_template_file())
    seen: list[str] = []
    for c in tpl.columns:
        if c and c not in seen:
            seen.append(c)
    return tuple(seen)


@st.cache_data(show_spinner=False)
def master_template_csv_bytes() -> bytes:
    """CSV rendition of the same master template rows/columns."""
    df = master_template_dataframe()
    return df.to_csv(index=False).encode("utf-8")


def master_template_excel_bytes() -> bytes:
    """Binary bytes of the master template file for direct download."""
    if not os.path.exists(MASTER_TEMPLATE_PATH):
        raise FileNotFoundError(f"Template not found: {MASTER_TEMPLATE_PATH}")
    with open(MASTER_TEMPLATE_PATH, "rb") as f:
        return f.read()


def _semantic_column_lookup() -> dict[str, str]:
    """Map normalized column name -> canonical semantic key for template matching."""
    lookup: dict[str, str] = {}
    for canonical, variants in TEMPLATE_SEMANTIC_GROUPS.items():
        key = normalize_column_name(canonical)
        lookup[key] = key
        for variant in variants:
            lookup[normalize_column_name(variant)] = key
    for src, tgt in UPLOAD_COLUMN_ALIASES.items():
        lookup[normalize_column_name(src)] = normalize_column_name(tgt)
    return lookup


def semantic_column_key(col: str, lookup: dict[str, str] | None = None) -> str:
    """Resolve a column to its canonical semantic key."""
    lookup = lookup or _semantic_column_lookup()
    normalized = normalize_column_name(col)
    return lookup.get(normalized, normalized)


def prepare_upload_for_template_validation(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize upload headers + aliases without dropping columns on empty sheets."""
    return apply_upload_aliases(normalize_dataframe_headers(df_raw))


def template_column_validation(df_upload: pd.DataFrame) -> dict:
    """
    Compare uploaded columns to master template using normalized + semantic matching.

    Fails only when a required template field is genuinely missing on the upload.
    Extra upload columns are allowed.
    """
    lookup = _semantic_column_lookup()
    template_cols = list(master_template_normalized_headers())
    upload_df = normalize_dataframe_headers(df_upload)
    upload_cols = list(upload_df.columns)

    template_by_key: dict[str, str] = {}
    for col in template_cols:
        key = semantic_column_key(col, lookup)
        template_by_key.setdefault(key, col)

    upload_by_key: dict[str, str] = {}
    for col in upload_cols:
        key = semantic_column_key(col, lookup)
        upload_by_key.setdefault(key, col)

    template_keys = set(template_by_key)
    upload_keys = set(upload_by_key)
    matched_keys = template_keys & upload_keys
    missing_keys = template_keys - upload_keys
    extra_keys = upload_keys - template_keys

    missing_columns = [template_by_key[k] for k in sorted(missing_keys)]
    extra_columns = [upload_by_key[k] for k in sorted(extra_keys)]
    matched_columns = [template_by_key[k] for k in sorted(matched_keys)]

    return {
        "template_columns": template_cols,
        "uploaded_columns": upload_cols,
        "normalized_template_columns": template_cols,
        "normalized_uploaded_columns": upload_cols,
        "matched_columns": matched_columns,
        "missing_columns": missing_columns,
        "extra_columns": extra_columns,
        "matches_template": len(missing_keys) == 0,
        "debug": {
            "normalized_template_columns": template_cols,
            "normalized_uploaded_columns": upload_cols,
            "matched_columns": matched_columns,
            "true_missing_columns": missing_columns,
            "extra_columns": extra_columns,
        },
    }


def inspect_upload_file(file_bytes: bytes, filename: str):
    """Parse upload and return schema validation before full ML pipeline."""
    df_raw = _load_raw_file(file_bytes, filename)
    df_norm = apply_upload_aliases(normalize_dataframe(df_raw))
    return validate_upload_schema(df_norm), df_norm


def inspect_upload_for_template(file_bytes: bytes, filename: str) -> tuple[dict, pd.DataFrame]:
    """Parse upload and return template validation + header-normalized frame."""
    df_raw = _load_raw_file(file_bytes, filename)
    df_headers = prepare_upload_for_template_validation(df_raw)
    return template_column_validation(df_headers), df_headers


@st.cache_data(show_spinner=False)
def run_full_pipeline(file_bytes: bytes, filename: str, logic_version: str = DECISION_LOGIC_VERSION):
    """Load uploaded file and run centralized preparation once."""
    del logic_version
    df_raw = _load_raw_file(file_bytes, filename)
    return ensure_unified_decisions(prepare_dashboard_dataframe(df_raw))


@st.cache_data(show_spinner=False)
def load_default_data(logic_version: str = DECISION_LOGIC_VERSION):
    del logic_version
    best = None
    for fname in [
        "melting_with_anomalies_stage5.csv",
        "melting_clustered_stage4.csv",
        "melting_features_stage2.csv",
    ]:
        p = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(p):
            best = p
            break
    if not best:
        return None
    df = pd.read_csv(best, low_memory=False)
    return ensure_unified_decisions(prepare_dashboard_dataframe(df))


def sample_template_dataframe() -> pd.DataFrame:
    """Upload template with all ML features from training schema."""
    feats = list(get_expected_features())
    row0 = sample_template_row(defect=0)
    row1 = sample_template_row(defect=1)
    row1["c"] = row0.get("c", 3.65) - 0.05
    row1["si"] = row0.get("si", 1.97) + 0.1
    cols = feats + (["defect"] if "defect" not in feats else [])
    if "defect" not in cols:
        cols.append("defect")
    return pd.DataFrame([{c: row0.get(c, 0) for c in cols}, {c: row1.get(c, 0) for c in cols}])
