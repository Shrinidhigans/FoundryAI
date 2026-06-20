"""
=============================================================================
STAGE 4 — PCA + PROCESS CLUSTERING
=============================================================================
Purpose : Discover process patterns and group batches by behaviour.
          Clustering is UNSUPERVISED — no use of defect label.
          This gives process-intelligence beyond defect/healthy binary.
Input   : outputs/melting_features_stage2.csv
Output  : outputs/melting_clustered_stage4.csv
          models/pca_model.pkl
          models/kmeans_model.pkl
          plots/pca_variance.png  |  pca_clusters.png  |  elbow.png
          outputs/cluster_profiles.csv
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

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

INPUT_FILE     = os.path.join("outputs", "melting_features_stage2.csv")
OUTPUT_FILE    = os.path.join("outputs", "melting_clustered_stage4.csv")
CLUSTER_PROF   = os.path.join("outputs", "cluster_profiles.csv")
MODELS_DIR     = "models"
PLOTS_DIR      = "plots"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# Do NOT use the target when clustering
EXCLUDE_COLS = ["defect"]

# Max clusters to test in elbow/silhouette analysis
MAX_K = 8


# ---------------------------------------------------------------------------
# DATA PREPARATION
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
    X = pd.DataFrame(X_arr, columns=numeric_cols)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return df, X, X_scaled, numeric_cols


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
# PCA ANALYSIS
# ---------------------------------------------------------------------------
def run_pca(X_scaled, n_components=None):
    """
    Fit PCA and return:
    - pca model
    - transformed data
    - number of components explaining ≥ 90% variance
    """
    pca_full = PCA(random_state=42)
    pca_full.fit(X_scaled)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n90 = int(np.argmax(cumvar >= 0.90)) + 1
    log.info(f"Components for 90% variance: {n90}")

    # Plot cumulative variance
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(range(1, min(26, len(cumvar)+1)),
                pca_full.explained_variance_ratio_[:25],
                color="#3498db", edgecolor="white")
    axes[0].set_title("Explained Variance per Component", fontweight="bold")
    axes[0].set_xlabel("Principal Component")
    axes[0].set_ylabel("Explained Variance Ratio")

    axes[1].plot(range(1, len(cumvar)+1), cumvar, marker="o", color="#e74c3c", ms=3)
    axes[1].axhline(0.90, ls="--", color="gray", alpha=0.7, label="90% threshold")
    axes[1].axvline(n90, ls="--", color="green", alpha=0.7, label=f"n={n90}")
    axes[1].set_title("Cumulative Explained Variance", fontweight="bold")
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance Ratio")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca_variance.png"), dpi=150)
    plt.close()

    # Final PCA with chosen components
    n_use = n_components or n90
    pca = PCA(n_components=n_use, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    log.info(f"PCA fitted: {n_use} components retain {cumvar[n_use-1]*100:.1f}% variance")
    return pca, X_pca, n_use


# ---------------------------------------------------------------------------
# ELBOW + SILHOUETTE ANALYSIS
# ---------------------------------------------------------------------------
def find_optimal_k(X_pca, max_k: int = MAX_K):
    """
    Run KMeans for k=2..max_k and compute inertia + silhouette.
    Returns the optimal k (highest silhouette score).
    """
    inertias   = []
    silhouettes= []
    k_range    = range(2, max_k + 1)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_pca, labels, sample_size=min(2000, len(X_pca))))
        log.info(f"  k={k}: inertia={km.inertia_:.0f}, silhouette={silhouettes[-1]:.4f}")

    # Plot elbow + silhouette
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(k_range, inertias, marker="o", color="#3498db")
    axes[0].set_title("Elbow Method — Inertia vs k", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(k_range, silhouettes, marker="s", color="#e74c3c")
    best_k_idx = int(np.argmax(silhouettes))
    axes[1].axvline(k_range[best_k_idx], ls="--", color="green",
                    label=f"Best k={k_range[best_k_idx]}")
    axes[1].set_title("Silhouette Score vs k", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "elbow_silhouette.png"), dpi=150)
    plt.close()

    optimal_k = k_range[best_k_idx]
    log.info(f"Optimal k (best silhouette): {optimal_k}")
    return optimal_k


# ---------------------------------------------------------------------------
# KMEANS CLUSTERING
# ---------------------------------------------------------------------------
def run_kmeans(X_pca, k: int):
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels, sample_size=min(2000, len(X_pca)))
    log.info(f"KMeans k={k}: silhouette={sil:.4f}")
    return km, labels


# ---------------------------------------------------------------------------
# DBSCAN COMPARISON
# ---------------------------------------------------------------------------
def run_dbscan(X_pca):
    """
    DBSCAN to find dense clusters vs noise.
    eps chosen heuristically (0.5 on PCA-reduced space).
    """
    db = DBSCAN(eps=0.8, min_samples=10, n_jobs=-1)
    labels = db.fit_predict(X_pca[:, :5])  # use first 5 PCs for DBSCAN
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = (labels == -1).sum()
    log.info(f"DBSCAN: {n_clusters} clusters, {n_noise} noise points ({100*n_noise/len(labels):.1f}%)")
    return labels


# ---------------------------------------------------------------------------
# VISUALISE CLUSTERS (PCA 2-D)
# ---------------------------------------------------------------------------
def plot_clusters(X_pca, labels, defect_labels=None):
    fig, axes = plt.subplots(1, 2 if defect_labels is not None else 1, figsize=(14, 5))
    if defect_labels is None:
        axes = [axes]

    cmap = plt.cm.get_cmap("tab10", max(labels)+1)
    scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=labels, cmap=cmap, alpha=0.5, s=10)
    axes[0].set_title("Process Clusters (PCA 2-D)", fontweight="bold")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(scatter, ax=axes[0], label="Cluster")

    if defect_labels is not None:
        axes[1].scatter(X_pca[:, 0], X_pca[:, 1],
                        c=defect_labels, cmap="RdYlGn_r", alpha=0.4, s=10)
        axes[1].set_title("Defect Labels (PCA 2-D)", fontweight="bold")
        axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pca_clusters.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# CLUSTER PROFILING & INDUSTRIAL INTERPRETATION
# ---------------------------------------------------------------------------
def profile_clusters(df: pd.DataFrame, labels: np.ndarray, feature_cols: list):
    """
    Compute mean chemistry + defect rate per cluster.
    Auto-generate an industrial narrative for each cluster.
    """
    df = df.copy()
    df["cluster"] = labels

    # Key variables for profiling
    profile_vars = [
        c for c in feature_cols
        if any(kw in c for kw in [
            "s_", "c_", "si_", "mn_", "mg_", "ce", "temp",
            "feat_sulfur", "feat_mg_recovery", "feat_temp_loss",
            "feat_shrinkage", "feat_chemistry"
        ])
    ]
    profile_vars = [v for v in profile_vars if v in df.columns]

    cluster_profile = df.groupby("cluster")[profile_vars].mean().round(4)
    if "defect" in df.columns:
        cluster_profile["defect_rate"] = (
            df.groupby("cluster")["defect"].mean().round(4)
        )
        cluster_profile["batch_count"] = df.groupby("cluster")["defect"].count()

    # Industrial narrative
    interpretations = {}
    for c in sorted(df["cluster"].unique()):
        row = cluster_profile.loc[c] if c in cluster_profile.index else {}
        desc = []
        dr = row.get("defect_rate", None)
        if dr is not None:
            desc.append(f"Defect rate: {dr:.1%}")

        s_val = row.get("s_", row.get("feat_sulfur_risk", None))
        if s_val is not None:
            if float(s_val) > 0.015:
                desc.append("High sulfur → nodularisation risk")
            else:
                desc.append("Sulfur within acceptable range")

        mg_val = row.get("mg_recovery_", row.get("feat_mg_recovery_risk", None))
        if mg_val is not None:
            if float(mg_val) > 0.5 and "risk" in str(mg_val):
                desc.append("Poor Mg recovery → flake graphite risk")
            elif float(mg_val) < 0.3:
                desc.append("Low Mg recovery → unstable nodularisation")

        tl = row.get("feat_temp_loss", None)
        if tl is not None and float(tl) > 70:
            desc.append("High temperature loss → cold pour risk")

        sh = row.get("feat_shrinkage_risk_index", None)
        if sh is not None and float(sh) > 2:
            desc.append("Elevated shrinkage risk index")

        ch = row.get("feat_chemistry_instability", None)
        if ch is not None and float(ch) > 0.2:
            desc.append("Chemistry deviation from target")

        interpretations[c] = "; ".join(desc) if desc else "Nominal process behaviour"

    cluster_profile["interpretation"] = pd.Series(interpretations)

    log.info("\n=== CLUSTER PROFILES ===")
    for c, interp in interpretations.items():
        cnt = int(cluster_profile.loc[c, "batch_count"]) if "batch_count" in cluster_profile else "?"
        log.info(f"  Cluster {c} ({cnt} batches): {interp}")

    return cluster_profile


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_stage4(input_file: str = INPUT_FILE):
    df, X, X_scaled, numeric_cols = prepare_features(input_file)

    pca, X_pca, n_components = run_pca(X_scaled)

    optimal_k = find_optimal_k(X_pca, max_k=MAX_K)

    km, km_labels = run_kmeans(X_pca, optimal_k)

    dbscan_labels = run_dbscan(X_pca)

    defect_labels = df["defect"].values if "defect" in df.columns else None
    plot_clusters(X_pca, km_labels, defect_labels)

    profile = profile_clusters(df, km_labels, numeric_cols)
    profile.to_csv(CLUSTER_PROF)
    log.info(f"Cluster profiles saved: {CLUSTER_PROF}")

    # Add cluster labels to main dataset
    df["cluster"]          = km_labels
    df["dbscan_cluster"]   = dbscan_labels
    df["pca_pc1"]          = X_pca[:, 0]
    df["pca_pc2"]          = X_pca[:, 1]

    df.to_csv(OUTPUT_FILE, index=False)
    log.info(f"Clustered dataset saved: {OUTPUT_FILE}")

    # Save models
    with open(os.path.join(MODELS_DIR, "pca_model.pkl"), "wb") as f:
        pickle.dump(pca, f)
    with open(os.path.join(MODELS_DIR, "kmeans_model.pkl"), "wb") as f:
        pickle.dump(km, f)
    with open(os.path.join(MODELS_DIR, "pca_scaler.pkl"), "wb") as f:
        pickle.dump(StandardScaler().fit(X), f)
    save_feature_schema_json(os.path.join(MODELS_DIR, "pca_feature_names.json"), numeric_cols, "stage4_pca")

    log.info("Stage 4 complete.")
    return df, pca, km, profile


if __name__ == "__main__":
    df, pca, km, profile = run_stage4()
    print(f"\nStage 4 complete. Shape: {df.shape}")
    print(f"\nCluster summary:\n{profile[['defect_rate','batch_count','interpretation']].to_string()}")
