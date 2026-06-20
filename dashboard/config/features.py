"""
Central ML feature schema for the Casting AI dashboard.

EXPECTED_FEATURES is loaded from models/feature_columns.pkl (training source of truth).
Training fill values (medians) come from models/feature_training_stats.pkl.
"""

from __future__ import annotations

import logging
import os
import pickle
import json
from functools import lru_cache
from typing import Dict, List

import pandas as pd

log = logging.getLogger(__name__)

MODELS_DIR = "models"
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")
FEATURE_NAMES_JSON_PATH = os.path.join(MODELS_DIR, "feature_names.json")
PCA_FEATURE_NAMES_JSON_PATH = os.path.join(MODELS_DIR, "pca_feature_names.json")
ISOLATION_FEATURE_NAMES_JSON_PATH = os.path.join(MODELS_DIR, "isolation_forest_feature_names.json")
FEATURE_STATS_PATH = os.path.join(MODELS_DIR, "feature_training_stats.pkl")
TRAINING_CSV = os.path.join("outputs", "melting_features_stage2.csv")

# Human / Excel upload names -> model column names (after clean_col_name)
UPLOAD_COLUMN_ALIASES: Dict[str, str] = {
    "carbon": "c",
    "carbon_pct": "c",
    "c_pct": "c",
    "silicon": "si",
    "silicon_pct": "si",
    "si_pct": "si",
    "sulfur": "s",
    "sulphur": "s",
    "s_pct": "s",
    "manganese": "mn_1",
    "mn_pct": "mn_1",
    "phosphorus": "p",
    "phos": "p",
    "p_pct": "p",
    "carbon_equivalent": "ce",
    "mg_recovery": "mg_recovery",
    "mg_recovery_pct": "mg_recovery",
    "magnesium_recovery": "mg_recovery",
    "tap_temp": "tapping_temp",
    "tapping_temperature": "tapping_temp",
    "pour_temp": "pouring_temp",
    "pouring_temperature": "pouring_temp",
    "pour_time": "pouring_time_sec",
    "pouring_time": "pouring_time_sec",
    "bath_sulphur": "bath_s",
    "bath_sulfur": "bath_s",
    "iron": "fe",
    "chromium": "cr",
    "nickel": "ni",
    "copper": "cu",
    "cobalt": "co",
    "defective": "defect",
    "defected": "defect",
    "is_defect": "defect",
    "quality_defect": "defect",
    "furnace_on": "furnace_on_time",
    "furnace_off": "furnace_off_time",
}

# Sample values for template download (representative melting heat)
SAMPLE_FEATURE_VALUES: Dict[str, float] = {
    "tapped_wt": 12.5,
    "pouring_wt": 12.0,
    "last_heel_metal": 2.0,
    "graphite": 15.0,
    "tapping_temp": 1435.0,
    "pouring_temp": 1370.0,
    "pouring_time_sec": 81.0,
    "fsm": 1.0,
    "ce": 4.16,
    "c": 3.65,
    "si": 1.97,
    "mn_1": 0.16,
    "p": 0.034,
    "s": 0.009,
    "mg_recovery": 0.45,
    "bath_s": 0.012,
    "fe": 93.5,
    "cu": 0.05,
    "co": 0.01,
    "cr": 0.03,
    "ni": 0.02,
    "mn": 0.16,
    "crca": 25.0,
    "defect": 0,
}


@lru_cache(maxsize=1)
def get_expected_features() -> tuple[str, ...]:
    """Feature names required by best_classifier.pkl (exact training order)."""
    if os.path.exists(FEATURE_NAMES_JSON_PATH):
        return tuple(_load_feature_schema_json(FEATURE_NAMES_JSON_PATH))
    if not os.path.exists(FEATURE_COLUMNS_PATH):
        log.warning("feature_columns.pkl not found — using empty feature list")
        return tuple()
    with open(FEATURE_COLUMNS_PATH, "rb") as f:
        feats = pickle.load(f)
    return tuple(str(c) for c in feats)


def _load_feature_schema_json(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, list):
        return [str(c) for c in payload]
    if isinstance(payload, dict):
        names = payload.get("feature_names") or payload.get("features") or []
        return [str(c) for c in names]
    return []


def save_feature_schema_json(path: str, feature_names: List[str], source: str) -> None:
    """Persist an ordered training feature schema for deterministic inference."""
    names = [str(c) for c in feature_names]
    engineered = [c for c in names if c.startswith("feat_")]
    payload = {
        "source": source,
        "feature_count": len(names),
        "feature_names": names,
        "engineered_feature_names": engineered,
        "engineered_feature_count": len(engineered),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


@lru_cache(maxsize=1)
def get_pca_feature_names() -> tuple[str, ...]:
    """Feature names required by pca_model.pkl/pca_scaler.pkl."""
    if os.path.exists(PCA_FEATURE_NAMES_JSON_PATH):
        return tuple(_load_feature_schema_json(PCA_FEATURE_NAMES_JSON_PATH))
    return get_expected_features()


@lru_cache(maxsize=1)
def get_isolation_forest_feature_names() -> tuple[str, ...]:
    """Feature names required by isolation_forest.pkl/anomaly_scaler.pkl."""
    if os.path.exists(ISOLATION_FEATURE_NAMES_JSON_PATH):
        return tuple(_load_feature_schema_json(ISOLATION_FEATURE_NAMES_JSON_PATH))
    return get_pca_feature_names()


# Module-level constant for imports
EXPECTED_FEATURES: List[str] = []  # populated on first access


def _refresh_expected_features() -> None:
    global EXPECTED_FEATURES
    EXPECTED_FEATURES[:] = list(get_expected_features())


_refresh_expected_features()


def _build_stats_from_training_csv() -> Dict[str, float]:
    if not os.path.exists(TRAINING_CSV):
        return {f: 0.0 for f in get_expected_features()}
    df = pd.read_csv(TRAINING_CSV, low_memory=False)
    stats: Dict[str, float] = {}
    for feat in get_expected_features():
        if feat in df.columns:
            s = pd.to_numeric(df[feat], errors="coerce")
            stats[feat] = float(s.median()) if s.notna().any() else 0.0
        else:
            stats[feat] = float(SAMPLE_FEATURE_VALUES.get(feat, 0.0))
    return stats


@lru_cache(maxsize=1)
def get_training_fill_values() -> Dict[str, float]:
    """Per-feature median values from training data for missing-column imputation."""
    if os.path.exists(FEATURE_STATS_PATH):
        with open(FEATURE_STATS_PATH, "rb") as f:
            stats = pickle.load(f)
        if isinstance(stats, dict) and stats:
            return {str(k): float(v) for k, v in stats.items()}

    log.info("Building feature_training_stats.pkl from training CSV")
    stats = _build_stats_from_training_csv()
    try:
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(FEATURE_STATS_PATH, "wb") as f:
            pickle.dump(stats, f)
    except OSError as exc:
        log.warning("Could not persist feature stats: %s", exc)
    return stats


def sample_template_row(defect: int = 0) -> Dict[str, float]:
    """One example row for Excel/CSV templates."""
    fills = get_training_fill_values()
    row = {feat: fills.get(feat, SAMPLE_FEATURE_VALUES.get(feat, 0.0)) for feat in get_expected_features()}
    if "defect" in row or "defect" in get_expected_features():
        row["defect"] = defect
    # Ensure key process fields use readable demo values
    for k, v in SAMPLE_FEATURE_VALUES.items():
        if k in row and k != "defect":
            row[k] = v
    row["defect"] = defect
    return row
