"""
=============================================================================
VALIDATION & KPI — INDUSTRIAL QUALITY METRICS REPORT
=============================================================================
Purpose : Generate a comprehensive validation report covering all pipeline
          stages: classification, clustering, anomaly detection, and KPIs.
Input   : Reads from outputs/ and models/ directories.
Output  : outputs/validation_report.txt
          plots/validation_*.png
=============================================================================
"""

import os
import pickle
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score,
    silhouette_score
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

OUTPUT_DIR  = "outputs"
MODELS_DIR  = "models"
PLOTS_DIR   = "plots"
REPORT_FILE = os.path.join(OUTPUT_DIR, "validation_report.txt")

os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# HELPER
# ---------------------------------------------------------------------------
def safe_load(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    log.warning(f"File not found: {path}")
    return None


def section(title, lines):
    sep = "=" * 60
    lines.insert(0, f"\n{sep}\n{title}\n{sep}")
    return lines


# ---------------------------------------------------------------------------
# 1. CLASSIFICATION VALIDATION
# ---------------------------------------------------------------------------
def validate_classifier(report_lines: list):
    report_lines += section("CLASSIFICATION VALIDATION", [])

    pipe    = safe_load(os.path.join(MODELS_DIR, "best_classifier.pkl"))
    feat_cols = safe_load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    df_feat   = None

    for fname in ["melting_with_anomalies_stage5.csv",
                  "melting_clustered_stage4.csv",
                  "melting_features_stage2.csv"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            df_feat = pd.read_csv(fpath, low_memory=False)
            break

    if pipe is None or feat_cols is None or df_feat is None:
        report_lines.append("  [SKIP] Classifier or feature file not found.")
        return report_lines

    from sklearn.impute import SimpleImputer
    available = [c for c in feat_cols if c in df_feat.columns]
    X = df_feat[available].copy()
    imp = SimpleImputer(strategy="median")
    X_arr = imp.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=available)
    y = df_feat["defect"].astype(int)

    # 5-fold CV on full dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc    = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_recall = cross_val_score(pipe, X, y, cv=cv, scoring="recall", n_jobs=-1)
    cv_f1     = cross_val_score(pipe, X, y, cv=cv, scoring="f1", n_jobs=-1)

    report_lines += [
        f"  CV ROC-AUC : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}",
        f"  CV Recall  : {cv_recall.mean():.4f} ± {cv_recall.std():.4f}",
        f"  CV F1      : {cv_f1.mean():.4f} ± {cv_f1.std():.4f}",
    ]

    # Load comparison table
    comp_path = os.path.join(OUTPUT_DIR, "model_comparison.csv")
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path)
        report_lines.append("\n  Model Comparison Table:")
        report_lines.append(comp.to_string(index=False))

    # Holistic predictions on full set (for reporting)
    y_pred  = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    report_lines.append("\n  Full-Dataset Classification Report:")
    report_lines.append(
        classification_report(y, y_pred, target_names=["Healthy", "Defective"])
    )

    # Plot CV score distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    metrics = {
        "ROC-AUC": cv_auc,
        "Recall":  cv_recall,
        "F1":      cv_f1,
    }
    positions = range(len(metrics))
    ax.boxplot(list(metrics.values()), labels=list(metrics.keys()),
               patch_artist=True,
               boxprops=dict(facecolor="#3498db", color="navy"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_title("5-Fold CV Metric Distribution", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "validation_cv_scores.png"), dpi=150)
    plt.close()

    return report_lines


# ---------------------------------------------------------------------------
# 2. CLUSTERING VALIDATION
# ---------------------------------------------------------------------------
def validate_clustering(report_lines: list):
    report_lines += section("CLUSTERING VALIDATION", [])

    cluster_file = os.path.join(OUTPUT_DIR, "melting_clustered_stage4.csv")
    if not os.path.exists(cluster_file):
        report_lines.append("  [SKIP] Cluster file not found.")
        return report_lines

    df = pd.read_csv(cluster_file, low_memory=False)
    if "cluster" not in df.columns:
        report_lines.append("  [SKIP] 'cluster' column not found.")
        return report_lines

    # Silhouette score on PCA components
    pca_cols = [c for c in df.columns if c.startswith("pca_")]
    if pca_cols:
        X_vis = df[pca_cols].values
        labels = df["cluster"].values
        sil = silhouette_score(X_vis, labels, sample_size=min(2000, len(df)))
        report_lines.append(f"  Silhouette Score (PCA space): {sil:.4f}")

    # Defect rate per cluster
    if "defect" in df.columns:
        cluster_stats = df.groupby("cluster").agg(
            batch_count=("defect", "count"),
            defect_rate=("defect", "mean"),
        ).round(4)
        report_lines.append("\n  Cluster Statistics:")
        report_lines.append(cluster_stats.to_string())

        # Plot defect rate per cluster
        fig, ax = plt.subplots(figsize=(8, 4))
        cluster_stats["defect_rate"].plot(
            kind="bar", ax=ax, color="#e74c3c", edgecolor="white"
        )
        ax.set_title("Defect Rate per Cluster", fontweight="bold")
        ax.set_xlabel("Cluster"); ax.set_ylabel("Defect Rate")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "validation_cluster_defect_rate.png"), dpi=150)
        plt.close()

    # Load cluster profiles if available
    prof_file = os.path.join(OUTPUT_DIR, "cluster_profiles.csv")
    if os.path.exists(prof_file):
        prof = pd.read_csv(prof_file, index_col=0)
        if "interpretation" in prof.columns:
            report_lines.append("\n  Cluster Interpretations:")
            for idx, row in prof.iterrows():
                report_lines.append(f"    Cluster {idx}: {row.get('interpretation','')}")

    return report_lines


# ---------------------------------------------------------------------------
# 3. ANOMALY DETECTION VALIDATION
# ---------------------------------------------------------------------------
def validate_anomaly(report_lines: list):
    report_lines += section("ANOMALY DETECTION VALIDATION", [])

    anom_file = os.path.join(OUTPUT_DIR, "melting_with_anomalies_stage5.csv")
    if not os.path.exists(anom_file):
        report_lines.append("  [SKIP] Anomaly file not found.")
        return report_lines

    df = pd.read_csv(anom_file, low_memory=False)

    sev_counts = df["anomaly_severity"].value_counts()
    report_lines.append("  Severity Distribution:")
    report_lines.append(sev_counts.to_string())

    anom_frac = df["anomaly_flag"].mean()
    report_lines.append(f"\n  Anomaly flag rate: {anom_frac:.2%}")

    if "defect" in df.columns:
        ct = pd.crosstab(df["defect"], df["anomaly_flag"],
                          rownames=["Defect"], colnames=["Anomaly"])
        report_lines.append("\n  Anomaly vs Defect cross-tab:")
        report_lines.append(ct.to_string())

    return report_lines


# ---------------------------------------------------------------------------
# 4. INDUSTRIAL KPI SUMMARY
# ---------------------------------------------------------------------------
def kpi_summary(report_lines: list):
    report_lines += section("INDUSTRIAL KPI SUMMARY", [])

    # Use the richest available file
    for fname in ["melting_with_anomalies_stage5.csv",
                  "melting_features_stage2.csv",
                  "melting_cleaned_stage1.csv"]:
        fpath = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, low_memory=False)
            break
    else:
        report_lines.append("  [SKIP] No output files found.")
        return report_lines

    kpis = {}

    if "defect" in df.columns:
        kpis["Total Batches"]      = len(df)
        kpis["Defective Batches"]  = int(df["defect"].sum())
        kpis["Healthy Batches"]    = int((df["defect"] == 0).sum())
        kpis["Overall Defect Rate"]= f"{df['defect'].mean():.2%}"

    if "s_" in df.columns:
        kpis["Avg Sulfur %"]       = f"{df['s_'].mean():.5f}"
        kpis["Max Sulfur %"]       = f"{df['s_'].max():.5f}"
        kpis["Batches High S"]     = int((df["s_"] > 0.015).sum())

    if "mg_recovery_" in df.columns:
        kpis["Avg Mg Recovery"]    = f"{df['mg_recovery_'].mean():.3f}"
        kpis["Low Mg Recovery (n)"]= int((df["mg_recovery_"] < 0.40).sum())

    if "feat_temp_loss" in df.columns:
        kpis["Avg Temp Loss (°C)"] = f"{df['feat_temp_loss'].mean():.1f}"
        kpis["High Temp Loss (n)"] = int((df["feat_temp_loss"] > 80).sum())

    if "anomaly_flag" in df.columns:
        kpis["Anomalies Detected"] = int(df["anomaly_flag"].sum())
        kpis["Anomaly Rate"]       = f"{df['anomaly_flag'].mean():.2%}"

    for k, v in kpis.items():
        report_lines.append(f"  {k:<30}: {v}")

    # KPI bar chart
    numeric_kpis = {k: float(str(v).replace("%", "").replace(",", ""))
                    for k, v in kpis.items()
                    if str(v).replace(".","").replace("%","").replace(",","").isdigit()
                    or (str(v).endswith("%") and str(v)[:-1].replace(".","").isdigit())}

    if numeric_kpis:
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(list(numeric_kpis.keys()), list(numeric_kpis.values()),
                       color="#3498db", edgecolor="white")
        ax.set_title("Industrial KPI Summary", fontweight="bold")
        ax.set_xlabel("Value")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "validation_kpis.png"), dpi=150)
        plt.close()

    return report_lines


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_validation():
    report_lines = [
        "=" * 60,
        "INDUSTRIAL AI CASTING QUALITY MONITORING PLATFORM",
        "VALIDATION & KPI REPORT",
        "=" * 60,
    ]

    report_lines = validate_classifier(report_lines)
    report_lines = validate_clustering(report_lines)
    report_lines = validate_anomaly(report_lines)
    report_lines = kpi_summary(report_lines)

    report_text = "\n".join(report_lines)
    with open(REPORT_FILE, "w") as f:
        f.write(report_text)

    log.info(f"Validation report saved: {REPORT_FILE}")
    print(report_text)
    return report_text


if __name__ == "__main__":
    run_validation()
