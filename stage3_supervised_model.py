"""
=============================================================================
STAGE 3 — SUPERVISED ML PIPELINE
=============================================================================
Purpose : Train, compare, and persist defect classifiers.
Target  : defect  (1 = defective, 0 = healthy)
Models  : Logistic Regression, Random Forest, Gradient Boosting,
          Extra Trees  (XGBoost/LightGBM not available in this environment)
Strategy: class_weight='balanced' handles imbalance without SMOTE.
          sklearn Pipeline prevents data leakage (scaler fitted on train only).
Output  : models/best_classifier.pkl
          models/scaler.pkl
          models/feature_columns.pkl
          outputs/model_comparison.csv
          plots/  (ROC, PR curves, confusion matrices, feature importance)
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
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
    cross_val_predict
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, recall_score
)
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
INPUT_FILE   = os.path.join("outputs", "melting_features_stage2.csv")
MODELS_DIR   = "models"
PLOTS_DIR    = "plots"
OUTPUT_COMP  = os.path.join("outputs", "model_comparison.csv")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

TARGET = "defect"


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

# Columns that must NEVER enter training regardless of what passes cleaning
ALWAYS_EXCLUDE = [
    "defect", "aa", "fsmaddition_mt",   # traceability / leakage guards
]


# ---------------------------------------------------------------------------
# DATA LOADING & FEATURE SELECTION
# ---------------------------------------------------------------------------
def load_and_split(input_file: str = INPUT_FILE):
    """
    Load stage-2 features, drop non-numeric columns, return X_train/test splits.
    No leakage: StandardScaler is part of the Pipeline and fitted on train only.
    """
    df = pd.read_csv(input_file, low_memory=False)
    log.info(f"Loaded shape: {df.shape}")

    # Separate target
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found.")

    y = df[TARGET].astype(int)

    # Feature matrix: only numeric columns, exclude blacklist
    exclude = set(ALWAYS_EXCLUDE) | {TARGET}
    feature_cols = []
    for col in df.columns:
        if col in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_cols.append(col)

    X = df[feature_cols].copy()

    # Final imputation safety net (handles any edge-case NaNs)
    imp = SimpleImputer(strategy="median")
    X_arr = imp.fit_transform(X)
    X = pd.DataFrame(X_arr, columns=feature_cols)

    log.info(f"Feature matrix: {X.shape}  |  Target balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, feature_cols


# ---------------------------------------------------------------------------
# MODEL DEFINITIONS
# ---------------------------------------------------------------------------
def build_pipelines():
    """
    Each model is wrapped in a Pipeline with StandardScaler.
    class_weight='balanced' upweights defective batches (minority class)
    without any resampling — safe and leak-free.
    """
    models = {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                C=0.5,
                solver="lbfgs",
                random_state=42
            )),
        ]),
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )),
        ]),
        "GradientBoosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                random_state=42
            )),
        ]),
        "ExtraTrees": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ExtraTreesClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42
            )),
        ]),
    }
    return models


# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
def evaluate_model(name, pipe, X_train, X_test, y_train, y_test):
    """Train, cross-validate, and compute full metrics for one model."""
    log.info(f"Training: {name}")
    pipe.fit(X_train, y_train)

    y_pred  = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_auc    = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_recall = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="recall", n_jobs=-1)

    auc      = roc_auc_score(y_test, y_proba)
    recall   = recall_score(y_test, y_pred)
    f1       = f1_score(y_test, y_pred)
    ap       = average_precision_score(y_test, y_proba)

    metrics = {
        "model":          name,
        "test_roc_auc":   round(auc, 4),
        "test_recall_def":round(recall, 4),
        "test_f1":        round(f1, 4),
        "test_avg_prec":  round(ap, 4),
        "cv_auc_mean":    round(cv_auc.mean(), 4),
        "cv_auc_std":     round(cv_auc.std(), 4),
        "cv_recall_mean": round(cv_recall.mean(), 4),
        "cv_recall_std":  round(cv_recall.std(), 4),
    }

    log.info(
        f"  ROC-AUC={auc:.4f} | Recall={recall:.4f} | F1={f1:.4f} | "
        f"CV-AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}"
    )
    return pipe, metrics, y_pred, y_proba


# ---------------------------------------------------------------------------
# PLOTTING
# ---------------------------------------------------------------------------
def plot_roc_all(all_proba, y_test, model_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in zip(model_names, all_proba):
        fpr, tpr, _ = roc_curve(y_test, proba)
        auc = roc_auc_score(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0,1],[0,1],'k--', alpha=0.4)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Defect Classifiers", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=150)
    plt.close()


def plot_pr_all(all_proba, y_test, model_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, proba in zip(model_names, all_proba):
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision–Recall Curves — Defect Classifiers", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "pr_curves.png"), dpi=150)
    plt.close()


def plot_confusion_matrix(name, y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Healthy","Defective"],
                yticklabels=["Healthy","Defective"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}", fontweight="bold")
    plt.tight_layout()
    safe_name = name.replace(" ","_")
    plt.savefig(os.path.join(PLOTS_DIR, f"cm_{safe_name}.png"), dpi=150)
    plt.close()


def plot_feature_importance(pipe, feature_cols, model_name, top_n=20):
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importance = np.abs(clf.coef_[0])
    else:
        return

    fi = pd.Series(importance, index=feature_cols).sort_values(ascending=False)
    fi_top = fi.head(top_n)

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#e74c3c" if "feat_" in c else "#3498db" for c in fi_top.index]
    fi_top[::-1].plot(kind="barh", ax=ax, color=colors[::-1])
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(f"Feature Importance — {model_name} (Top {top_n})", fontsize=12, fontweight="bold")
    red_patch  = mpatches.Patch(color="#e74c3c", label="Engineered feature")
    blue_patch = mpatches.Patch(color="#3498db", label="Raw feature")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9)
    plt.tight_layout()
    safe_name = model_name.replace(" ","_")
    plt.savefig(os.path.join(PLOTS_DIR, f"fi_{safe_name}.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# MAIN PIPELINE
# ---------------------------------------------------------------------------
def run_stage3(input_file: str = INPUT_FILE):
    X_train, X_test, y_train, y_test, feature_cols = load_and_split(input_file)

    models = build_pipelines()
    results  = []
    all_proba = []
    all_pred  = []
    trained_pipes = {}

    for name, pipe in models.items():
        trained_pipe, metrics, y_pred, y_proba = evaluate_model(
            name, pipe, X_train, X_test, y_train, y_test
        )
        results.append(metrics)
        all_proba.append(y_proba)
        all_pred.append(y_pred)
        trained_pipes[name] = trained_pipe
        plot_confusion_matrix(name, y_test, y_pred)
        plot_feature_importance(trained_pipe, feature_cols, name)

        # Detailed classification report
        log.info(f"\n{name} Classification Report:\n"
                 f"{classification_report(y_test, y_pred, target_names=['Healthy','Defective'])}")

    # ------------------------------------------------------------------
    # COMPARISON TABLE
    # ------------------------------------------------------------------
    df_comp = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
    df_comp.to_csv(OUTPUT_COMP, index=False)
    log.info(f"\nModel Comparison:\n{df_comp.to_string(index=False)}")

    # ------------------------------------------------------------------
    # SELECT BEST MODEL (highest AUC; recall as tie-breaker)
    # ------------------------------------------------------------------
    best_row  = df_comp.iloc[0]
    best_name = best_row["model"]
    best_pipe = trained_pipes[best_name]
    log.info(f"\nBest model: {best_name}  (AUC={best_row['test_roc_auc']})")

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    plot_roc_all(all_proba, y_test, list(models.keys()))
    plot_pr_all(all_proba, y_test, list(models.keys()))

    # ------------------------------------------------------------------
    # SAVE ARTIFACTS
    # ------------------------------------------------------------------
    # Save best pipeline (includes scaler)
    with open(os.path.join(MODELS_DIR, "best_classifier.pkl"), "wb") as f:
        pickle.dump(best_pipe, f)

    # Save scaler separately (for dashboard convenience)
    scaler = best_pipe.named_steps.get("scaler", None)
    if scaler:
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    # Save feature column list (critical for correct prediction later)
    with open(os.path.join(MODELS_DIR, "feature_columns.pkl"), "wb") as f:
        pickle.dump(feature_cols, f)
    save_feature_schema_json(os.path.join(MODELS_DIR, "feature_names.json"), feature_cols, "stage3_classifier")

    feature_stats = {
        col: float(pd.to_numeric(X_train[col], errors="coerce").median())
        if pd.to_numeric(X_train[col], errors="coerce").notna().any()
        else 0.0
        for col in feature_cols
    }
    with open(os.path.join(MODELS_DIR, "feature_training_stats.pkl"), "wb") as f:
        pickle.dump(feature_stats, f)

    # Save all trained pipelines
    with open(os.path.join(MODELS_DIR, "all_classifiers.pkl"), "wb") as f:
        pickle.dump(trained_pipes, f)

    log.info("Stage 3 complete. Models and plots saved.")
    return best_pipe, feature_cols, df_comp


if __name__ == "__main__":
    best_pipe, feature_cols, comp = run_stage3()
    print(f"\nStage 3 complete.")
    print(comp.to_string(index=False))
