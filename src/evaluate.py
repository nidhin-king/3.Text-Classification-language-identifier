"""Model evaluation utilities.

Computes the full set of classification metrics (accuracy, macro precision /
recall / F1, per-class classification report, confusion matrix, cross
validation and ROC-AUC for multiclass) and persists results to ``reports/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.utils import get_logger, save_json
from src.visualizer import plot_confusion_matrix

logger = get_logger("evaluate")


def evaluate_model(
    model,
    X_test,
    y_test,
    label_encoder,
    reports_dir: Path,
    prefix: str = "final",
    X_train: np.ndarray | None = None,
    y_train: np.ndarray | None = None,
) -> dict:
    """Evaluate a fitted model and persist all outputs.

    Args:
        model: Fitted classifier with ``predict`` (and ideally ``predict_proba``).
        X_test: Test feature matrix.
        y_test: Test labels (already label-encoded).
        label_encoder: Fitted ``LabelEncoder`` used to map back to names.
        reports_dir: Where to write plots / reports.
        prefix: File prefix, e.g. ``"final"`` or ``"cv"``.
        X_train, y_train: Optional training data used to fit an OvR ROC plot.

    Returns:
        Dictionary of metrics (also persisted to ``reports_dir/{prefix}_metrics.json``).
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_pred = model.predict(X_test)

    metrics: dict = {}
    metrics["accuracy"] = round(float(accuracy_score(y_test, y_pred)), 4)
    metrics["precision_macro"] = round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    metrics["recall_macro"] = round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    metrics["f1_macro"] = round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4)
    metrics["precision_weighted"] = round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
    metrics["recall_weighted"] = round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4)
    metrics["f1_weighted"] = round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4)

    # Per-class classification report
    target_names = list(label_encoder.classes_)
    report = classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0, output_dict=True
    )
    metrics["classification_report"] = report

    # Machine-readable per-class table
    report_df = pd.DataFrame(report).T
    report_df.to_csv(reports_dir / f"{prefix}_classification_report.csv", encoding="utf-8")

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(
        cm, target_names, str(reports_dir / f"{prefix}_confusion_matrix.png"),
        title=f"Confusion Matrix ({prefix})",
    )

    # ROC-AUC (OvR) if probabilities are available
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X_test)
            roc_micro = roc_auc_score(y_test, proba, multi_class="ovr", average="micro")
            roc_macro = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
            metrics["roc_auc_micro"] = round(float(roc_micro), 4)
            metrics["roc_auc_macro"] = round(float(roc_macro), 4)
        except Exception as exc:  # noqa: BLE001 - ROC is best-effort
            logger.warning("ROC-AUC computation skipped: %s", exc)

    # Feature importances when available
    if hasattr(model, "feature_importances_"):
        fi = np.asarray(model.feature_importances_)
        metrics["top_features"] = {
            "n_features": int(fi.size),
            "sum": float(fi.sum()),
        }

    save_json(metrics, reports_dir / f"{prefix}_metrics.json")
    _log_summary(metrics, target_names)
    return metrics


def cross_validate_model(
    model,
    X,
    y,
    folds: int = 5,
    seed: int = 42,
    n_jobs: int = 1,
) -> dict:
    """Stratified K-fold cross-validation accuracy.

    Args:
        model: Unfitted classifier.
        X: Feature matrix.
        y: Encoded labels.
        folds: Number of folds.
        seed: Random seed.
        n_jobs: Parallel jobs for ``cross_val_score``.

    Returns:
        Dict with ``fold_scores``, ``mean``, ``std`` and ``min``/``max``.
    """
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=n_jobs)
    result = {
        "fold_scores": [round(float(s), 4) for s in scores],
        "mean": round(float(scores.mean()), 4),
        "std": round(float(scores.std()), 4),
        "min": round(float(scores.min()), 4),
        "max": round(float(scores.max()), 4),
    }
    logger.info("CV accuracy: %s (mean=%.4f, std=%.4f)", scores, result["mean"], result["std"])
    return result


def _log_summary(metrics: dict, target_names: list[str]) -> None:
    """Print a compact human-readable summary."""
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision (macro): {metrics['precision_macro']:.4f}")
    print(f"Recall    (macro): {metrics['recall_macro']:.4f}")
    print(f"F1        (macro): {metrics['f1_macro']:.4f}")
    for key in ("roc_auc_micro", "roc_auc_macro"):
        if key in metrics:
            print(f"{key:>9}        : {metrics[key]:.4f}")
    if "top_features" in metrics:
        print(f"Model uses {metrics['top_features']['n_features']} features")
    print(f"Languages: {len(target_names)}")


def worst_classes(metrics: dict, top_k: int = 5) -> pd.DataFrame:
    """Return the classes with the lowest F1 scores for quick inspection."""
    report = pd.DataFrame(metrics["classification_report"]).T
    report = report[report.index.isin(metrics.get("_language_names", []))] if "_language_names" in metrics else report
    per_class = report[report.index != "accuracy"].drop(
        labels=["micro avg", "macro avg", "weighted avg", "accuracy"], errors="ignore"
    )
    per_class = per_class[pd.to_numeric(per_class["f1-score"], errors="coerce").notna()]
    per_class["f1-score"] = pd.to_numeric(per_class["f1-score"])
    return per_class.nsmallest(top_k, "f1-score")[["precision", "recall", "f1-score", "support"]]
