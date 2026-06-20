"""
=============================================================================
STAGE 5 — INDUSTRIAL ANOMALY DETECTION
=============================================================================
Purpose : Flag process anomalies using Isolation Forest and Local Outlier
          Factor.  Because we now have BOTH defective and healthy batches,
          anomaly detection learns a richer normal-process envelope.
Input   : outputs/melting_clustered_stage4.csv
Output  : outputs/melting_with_anomalies_stage5.csv
          models/isolation_forest.pkl
          models/lof_model.pkl
=============================================================================
"""

import os
import pickle
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_FILE  = os.path.join("outputs", "melting_clustered_stage4.csv")
OUTPUT_FILE = os.path.join("outputs", "melting_with_anomalies_stage5.csv")
MODELS_DIR  = "models"
PLOTS_DIR   = "plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# Contamination = approximate fraction of outliers expected
# ~29% of our data is defective → use a slightly conservative 0.20
CONTAMINATION = 0.20

# Columns excluded from anomaly scoring
EXCLUDE_COLS = [
    "defect", "cluster", "dbscan_cluster", "pca_pc1", "pca_pc2"
]


# ---------------------------------------------------------------------------
# SEVERITY MAPPING
# ---------------------------------------------------------------------------
def score_to_severity(score: float) -> str:
    """
    Convert a normalised anomaly score (0–1, higher=more anomalous) to
    a human-readable severity label used in QA reports.
    """
    if score >= 0.80:  return "CRITICAL"
    if score >= 0.60:  return "HIGH RISK"
    if score >= 0.40:  return "MEDIUM RISK"
    if score >= 0.20:  return "LOW RISK"
    return "NORMAL"


def severity_to_int(sev: str) -> int:
    mapping = {"NORMAL": 0, "LOW RISK": 1, "MEDIUM RISK": 2, "HIGH RISK": 3, "CRITICAL": 4}
    return mapping.get(sev, 0)


# ---------------------------------------------------------------------------
# FEATURE PREPARATION
# ---------------------------------------------------------------------------
def prepare_features(input_file: str):
    df = pd.read_csv(input_file, low_memory=False)
    log.info(f"Loaded: {df.shape}")

    numeric_cols = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and c not in EXCLUDE_COLS
    ]

    X = df[numeric_cols].copy()
    imp = SimpleImputer(strategy="median")
    X_arr = imp.fit_transform(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_arr)

    return df, X_scaled, numeric_cols, scaler


def save_feature_schema_json(path: str, feature_cols: list, source: str):
    names = [str(c) for c in feature_cols]
    engineered = [c for c in names if c.startswith("feat_")]
    payload = {
        "source": source,
        "feature_count": len(names),
        "feature_names": names,
        "engineered_feature_names": engineered,
        "engineered_feature_count": len(engineered),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# ISOLATION FOREST
# ---------------------------------------------------------------------------
def run_isolation_forest(X_scaled, contamination: float = CONTAMINATION):
    """
    Isolation Forest: isolates anomalies by randomly partitioning data.
    Anomalies require fewer splits → shorter path lengths → lower score.
    Returns raw decision scores and binary flag.
    """
    log.info(f"Isolation Forest: contamination={contamination}")
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)

    # score_samples returns negative scores: more negative = more anomalous
    raw_scores = iso.score_samples(X_scaled)
    labels     = iso.predict(X_scaled)   # 1=normal, -1=anomaly

    # Normalise to [0, 1] where 1 = most anomalous
    norm_scores = 1 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    iso_flag    = (labels == -1).astype(int)

    log.info(f"  Anomalies detected: {iso_flag.sum()} ({100*iso_flag.mean():.1f}%)")
    return iso, norm_scores, iso_flag


# ---------------------------------------------------------------------------
# LOCAL OUTLIER FACTOR
# ---------------------------------------------------------------------------
def run_lof(X_scaled, contamination: float = CONTAMINATION):
    """
    Local Outlier Factor: compares local density of a point to its neighbours.
    Points in sparse regions (low density vs neighbours) are anomalies.
    LOF is good at finding cluster-relative outliers.
    """
    log.info("Local Outlier Factor …")
    lof = LocalOutlierFactor(
        n_neighbors=20,
        contamination=contamination,
        n_jobs=-1
    )
    labels     = lof.fit_predict(X_scaled)      # 1=normal, -1=anomaly
    raw_scores = -lof.negative_outlier_factor_  # higher = more anomalous

    # Normalise
    norm_scores = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)
    lof_flag    = (labels == -1).astype(int)

    log.info(f"  LOF anomalies: {lof_flag.sum()} ({100*lof_flag.mean():.1f}%)")
    return lof, norm_scores, lof_flag


# ---------------------------------------------------------------------------
# ENSEMBLE SCORING
# ---------------------------------------------------------------------------
def ensemble_score(iso_scores, lof_scores, iso_flag, lof_flag):
    """
    Combine Isolation Forest and LOF scores:
    - Weighted average of normalised scores (equal weight)
    - Ensemble flag: anomaly if EITHER detector flags it
    - Severity based on combined score
    """
    combined_score = 0.5 * iso_scores + 0.5 * lof_scores
    ensemble_flag  = np.maximum(iso_flag, lof_flag)   # 1 if either flags
    severity       = [score_to_severity(s) for s in combined_score]
    severity_int   = [severity_to_int(s) for s in severity]
    return combined_score, ensemble_flag, severity, severity_int


# ---------------------------------------------------------------------------
# VISUALISATION
# ---------------------------------------------------------------------------
def plot_anomaly_distribution(df_out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Severity counts
    sev_counts = df_out["anomaly_severity"].value_counts()
    order = ["NORMAL", "LOW RISK", "MEDIUM RISK", "HIGH RISK", "CRITICAL"]
    sev_counts = sev_counts.reindex([o for o in order if o in sev_counts.index], fill_value=0)
    colors = ["#27ae60","#f1c40f","#e67e22","#e74c3c","#8e44ad"][:len(sev_counts)]
    sev_counts.plot(kind="bar", ax=axes[0], color=colors, edgecolor="white")
    axes[0].set_title("Anomaly Severity Distribution", fontweight="bold")
    axes[0].set_xlabel("Severity"); axes[0].set_ylabel("Batch Count")
    axes[0].tick_params(axis="x", rotation=30)

    # Anomaly score by defect label
    if "defect" in df_out.columns:
        df_out.boxplot(column="anomaly_score", by="defect", ax=axes[1])
        axes[1].set_title("Anomaly Score: Healthy vs Defective")
        axes[1].set_xlabel("Defect (0=Healthy, 1=Defective)")
        axes[1].set_ylabel("Anomaly Score")
        plt.suptitle("")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "anomaly_distribution.png"), dpi=150)
    plt.close()


def plot_anomaly_pca(df_out):
    if "pca_pc1" not in df_out or "pca_pc2" not in df_out:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = df_out["anomaly_severity"].map({
        "NORMAL":"#27ae60","LOW RISK":"#f1c40f",
        "MEDIUM RISK":"#e67e22","HIGH RISK":"#e74c3c","CRITICAL":"#8e44ad"
    }).fillna("#3498db")

    ax.scatter(df_out["pca_pc1"], df_out["pca_pc2"],
               c=colors, alpha=0.5, s=12)
    ax.set_title("Anomaly Severity in PCA Space", fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")

    # Legend patches
    import matplotlib.patches as mpatches
    patches = [
        mpatches.Patch(color="#27ae60", label="NORMAL"),
        mpatches.Patch(color="#f1c40f", label="LOW RISK"),
        mpatches.Patch(color="#e67e22", label="MEDIUM RISK"),
        mpatches.Patch(color="#e74c3c", label="HIGH RISK"),
        mpatches.Patch(color="#8e44ad", label="CRITICAL"),
    ]
    ax.legend(handles=patches, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "anomaly_pca.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_stage5(input_file: str = INPUT_FILE):
    df, X_scaled, numeric_cols, scaler = prepare_features(input_file)

    iso, iso_scores, iso_flag = run_isolation_forest(X_scaled)
    lof, lof_scores, lof_flag = run_lof(X_scaled)

    combined, ensemble_flag, severity, severity_int = ensemble_score(
        iso_scores, lof_scores, iso_flag, lof_flag
    )

    df["anomaly_score"]    = combined
    df["iso_flag"]         = iso_flag
    df["lof_flag"]         = lof_flag
    df["anomaly_flag"]     = ensemble_flag
    df["anomaly_severity"] = severity
    df["anomaly_level"]    = severity_int

    # Summary stats
    sev_count = pd.Series(severity).value_counts()
    log.info(f"\nAnomaly severity summary:\n{sev_count.to_string()}")

    if "defect" in df.columns:
        log.info("\nAnomaly flag vs Defect label:")
        log.info(pd.crosstab(df["defect"], df["anomaly_flag"],
                              rownames=["Defect"], colnames=["Anomaly"]).to_string())

    plot_anomaly_distribution(df)
    plot_anomaly_pca(df)

    df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Saved: {OUTPUT_FILE}")

    # Persist Isolation Forest (LOF is transductive; not saved separately)
    with open(os.path.join(MODELS_DIR, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso, f)
    with open(os.path.join(MODELS_DIR, "anomaly_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    save_feature_schema_json(
        os.path.join(MODELS_DIR, "isolation_forest_feature_names.json"),
        numeric_cols,
        "stage5_isolation_forest",
    )

    log.info("Stage 5 complete.")
    return df


if __name__ == "__main__":
    df = run_stage5()
    print(f"\nStage 5 complete. Shape: {df.shape}")
    print(df["anomaly_severity"].value_counts().to_string())
