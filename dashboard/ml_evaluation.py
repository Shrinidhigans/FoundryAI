"""
Offline evaluation of the trained defect classifier for the ML Performance dashboard.

Read-only: loads persisted model + training CSV; does not alter pipeline or models.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from dashboard.config.features import get_expected_features
from dashboard.preprocessing import align_ml_features, safe_numeric_series
from dashboard.theme import ACCENT, DANGER, PLOTLY_LAYOUT, SUCCESS, WARNING

try:
    import plotly.graph_objects as go

    PLOTLY_OK = True
except ImportError:
    go = None
    PLOTLY_OK = False

MODELS_DIR = "models"
CLASSIFIER_PATH = os.path.join(MODELS_DIR, "best_classifier.pkl")
FEATURE_COLUMNS_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")

TARGET_CANDIDATES = ["defect", "defective", "is_defect", "quality_defect"]

TRAINING_DATA_CANDIDATES = [
    os.path.join("outputs", "melting_features_stage2.csv"),
    os.path.join("outputs", "melting_with_anomalies_stage5.csv"),
    os.path.join("outputs", "melting_clustered_stage4.csv"),
    os.path.join("outputs", "melting_cleaned_stage1.csv"),
    os.path.join("data", "melting_features_stage2.csv"),
    os.path.join("datasets", "melting_features_stage2.csv"),
    os.path.join("processed", "melting_features_stage2.csv"),
]


@dataclass
class EvaluationResult:
    """Container for model evaluation outputs."""

    ok: bool = False
    error: str = ""
    dataset_path: str = ""
    n_samples: int = 0
    n_features: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((2, 2)))
    classification_report_text: str = ""
    roc_fpr: np.ndarray = field(default_factory=lambda: np.array([]))
    roc_tpr: np.ndarray = field(default_factory=lambda: np.array([]))
    roc_auc: float = 0.0
    y_test: np.ndarray = field(default_factory=lambda: np.array([]))
    y_pred: np.ndarray = field(default_factory=lambda: np.array([]))
    y_prob: np.ndarray = field(default_factory=lambda: np.array([]))
    feature_names: List[str] = field(default_factory=list)
    feature_importance: np.ndarray = field(default_factory=lambda: np.array([]))
    importance_source: str = ""


def _layout(**kwargs):
    base = dict(PLOTLY_LAYOUT)
    base.update(kwargs)
    return base


def load_model():
    """Load trained sklearn pipeline (scaler + classifier)."""
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(f"Model not found: {CLASSIFIER_PATH}")
    with open(CLASSIFIER_PATH, "rb") as f:
        return pickle.load(f)


def load_feature_columns() -> List[str]:
    """Load feature list used at training time."""
    if os.path.exists(FEATURE_COLUMNS_PATH):
        with open(FEATURE_COLUMNS_PATH, "rb") as f:
            cols = pickle.load(f)
        return [str(c) for c in cols]
    return list(get_expected_features())


def _detect_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_CANDIDATES:
        if col in df.columns:
            return col
    for col in df.columns:
        if "defect" in str(col).lower() and col != "defect_prob" and col != "defect_pred":
            return col
    raise ValueError(
        f"No defect target column found. Expected one of: {TARGET_CANDIDATES}"
    )


def load_training_data() -> Tuple[pd.DataFrame, str]:
    """Locate and load the best available labeled training/evaluation dataset."""
    for path in TRAINING_DATA_CANDIDATES:
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            if len(df) == 0:
                continue
            _detect_target_column(df)
            return df, path
    raise FileNotFoundError(
        "No training dataset found. Run stages 1–3 to create outputs/melting_features_stage2.csv"
    )


def _estimator_from_pipeline(model) -> Any:
    if hasattr(model, "named_steps"):
        return model.named_steps.get("clf", model)
    return model


def _feature_importance_vector(model, feature_names: List[str], X_test: pd.DataFrame, y_test: np.ndarray):
    est = _estimator_from_pipeline(model)
    if hasattr(est, "feature_importances_"):
        imp = np.asarray(est.feature_importances_, dtype=float)
        if len(imp) == len(feature_names):
            return imp, "model.feature_importances_"
    if hasattr(est, "coef_"):
        coef = np.asarray(est.coef_)
        imp = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)
        if len(imp) == len(feature_names):
            return imp, "model.coef_"
    try:
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            n_jobs=1,
        )
        return perm.importances_mean, "permutation_importance"
    except Exception:
        return np.zeros(len(feature_names)), "unavailable"


def evaluate_model(
    test_size: float = 0.2,
    random_state: int = 42,
) -> EvaluationResult:
    """Evaluate persisted classifier on a hold-out split of the training dataset."""
    result = EvaluationResult()
    try:
        model = load_model()
        feature_names = load_feature_columns()
        raw_df, path = load_training_data()
        result.dataset_path = path
        result.n_samples = len(raw_df)

        target_col = _detect_target_column(raw_df)
        y = safe_numeric_series(raw_df[target_col]).fillna(0).astype(int)
        if y.nunique() < 2:
            raise ValueError("Target column must contain at least two classes (0 and 1).")

        X, _ = align_ml_features(raw_df, feature_names)
        if X.shape[1] != len(feature_names):
            raise ValueError("Feature alignment failed for evaluation matrix.")

        result.n_features = X.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = y_pred.astype(float)

        result.y_test = np.asarray(y_test)
        result.y_pred = np.asarray(y_pred)
        result.y_prob = np.asarray(y_prob)

        result.metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_prob)) if y.nunique() > 1 else 0.0,
        }
        result.roc_auc = result.metrics["roc_auc"]

        result.confusion = confusion_matrix(y_test, y_pred, labels=[0, 1])
        result.classification_report_text = classification_report(
            y_test, y_pred, target_names=["Healthy (0)", "Defective (1)"], zero_division=0
        )

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        result.roc_fpr = fpr
        result.roc_tpr = tpr

        imp, source = _feature_importance_vector(model, feature_names, X_test, y_test)
        result.feature_names = feature_names
        result.feature_importance = imp
        result.importance_source = source

        result.ok = True
    except Exception as exc:
        result.ok = False
        result.error = str(exc)
    return result


def generate_confusion_matrix(result: EvaluationResult):
    """Plotly heatmap for 2x2 confusion matrix."""
    if not PLOTLY_OK or not result.ok:
        return None
    cm = result.confusion
    labels = [["TN", "FP"], ["FN", "TP"]]
    text = [[f"{labels[i][j]}<br>{cm[i, j]}" for j in range(2)] for i in range(2)]
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=["Predicted 0", "Predicted 1"],
            y=["Actual 0", "Actual 1"],
            colorscale=[[0, "#1a2332"], [0.5, ACCENT], [1, DANGER]],
            text=text,
            texttemplate="%{text}",
            textfont={"size": 14, "color": "white"},
            showscale=True,
        )
    )
    fig.update_layout(**_layout(title="Confusion Matrix", height=380))
    return fig


def generate_roc_curve(result: EvaluationResult):
    """Plotly ROC curve with AUC annotation."""
    if not PLOTLY_OK or not result.ok or len(result.roc_fpr) == 0:
        return None
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=result.roc_fpr,
            y=result.roc_tpr,
            mode="lines",
            name=f"ROC (AUC = {result.roc_auc:.3f})",
            line=dict(color=ACCENT, width=2.5),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random",
            line=dict(color="#666", dash="dash"),
        )
    )
    fig.update_layout(
        **_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=400,
        )
    )
    return fig


def generate_feature_importance(result: EvaluationResult, top_n: int = 15):
    """Horizontal bar chart of top feature importances."""
    if not PLOTLY_OK or not result.ok or len(result.feature_importance) == 0:
        return None
    series = pd.Series(result.feature_importance, index=result.feature_names)
    top = series.sort_values(ascending=True).tail(top_n)
    fig = go.Figure(
        go.Bar(
            y=[str(i) for i in top.index],
            x=top.values,
            orientation="h",
            marker_color=ACCENT,
            text=[f"{v:.4f}" for v in top.values],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_layout(
            title=f"Top {top_n} Features ({result.importance_source})",
            xaxis_title="Importance",
            height=max(320, 28 * len(top)),
        )
    )
    return fig


def generate_prediction_distribution(result: EvaluationResult):
    """Bar + donut of predicted classes on hold-out test set."""
    if not PLOTLY_OK or not result.ok:
        return None
    pred = pd.Series(result.y_pred).value_counts().sort_index()
    labels = ["Healthy (0)", "Defective (1)"]
    values = [int(pred.get(0, 0)), int(pred.get(1, 0))]
    colors = [SUCCESS, DANGER]
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors),
            textinfo="label+percent+value",
        )
    )
    fig.update_layout(**_layout(title="Prediction Distribution (test set)", height=360))
    return fig


def generate_probability_histogram(result: EvaluationResult):
    """Distribution of predicted defect probabilities on test set."""
    if not PLOTLY_OK or not result.ok or len(result.y_prob) == 0:
        return None
    fig = go.Figure(go.Histogram(x=result.y_prob, nbinsx=30, marker_color=WARNING, opacity=0.85))
    fig.add_vline(x=0.5, line_dash="dash", line_color=DANGER, annotation_text="Threshold 0.5")
    fig.update_layout(
        **_layout(title="Defect Probability Distribution (test set)", xaxis_title="P(defect)", height=340)
    )
    return fig
