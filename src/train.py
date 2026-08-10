"""Training pipeline.

Workflow::

    load dataset
      -> preprocess + encode labels
      -> auto-select best feature strategy (logistic-regression proxy)
      -> compare classifiers with cross validation
      -> train the winning model on the full training split
      -> evaluate on the held-out test split
      -> persist model + vectorizer + label encoder + metadata

Everything is reproducible via ``RANDOM_STATE`` and the results land in
``reports/`` and ``models/``. Run with::

    python -m src.train
"""

from __future__ import annotations

import os
from pathlib import Path

# Bound native thread pools (XGBoost / LightGBM / OpenMP) so they never
# oversubscribe the CPU when combined with joblib parallelism.
try:
    _CPU_COUNT = len(os.sched_getaffinity(0))
except (AttributeError, OSError):  # pragma: no cover - non-Linux fallback
    _CPU_COUNT = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(min(_CPU_COUNT, 8)))
os.environ.setdefault("MKL_NUM_THREADS", str(min(_CPU_COUNT, 8)))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from src.config import (
    COMPARISON_SUBSAMPLE,
    CV_FOLDS,
    MAX_FEATURES,
    MODEL_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE,
)
from src.data_loader import load_dataset
from src.evaluate import evaluate_model
from src.feature_engineering import get_vectorizer, supported_strategies
from src.preprocessing import clean_series, deduplicate
from src.predict import save_pipeline
from src.utils import get_logger, save_json, setup_logging, timed

logger = get_logger("train")

# --------------------------------------------------------------------------- #
# Model registry
# --------------------------------------------------------------------------- #
#: XGBoost is *opt-in*: multiclass training on sparse features is very slow,
#: so it is skipped by default. Enable with ``ENABLE_XGBOOST=1 python -m src.train``.
ENABLE_XGBOOST: bool = os.environ.get("ENABLE_XGBOOST", "0") == "1"


def _build_models() -> dict[str, object]:
    """Instantiate all classifiers to compare.

    Returns:
        Ordered dict of ``name -> unfitted estimator``.
    """
    models: dict[str, object] = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(
            C=1.0, solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE
        ),
        "SGD (modified huber)": SGDClassifier(
            loss="modified_huber",
            max_iter=1000,
            tol=1e-4,
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(C=1.0, max_iter=2000, dual="auto", random_state=RANDOM_STATE),
    }

    try:  # Random Forest
        from sklearn.ensemble import RandomForestClassifier

        models["Random Forest"] = RandomForestClassifier(
            n_estimators=120,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Random Forest unavailable: %s", exc)

    if ENABLE_XGBOOST:  # opt-in (see module docstring)
        try:  # XGBoost (optional)
            from xgboost import XGBClassifier

            # Deliberately tiny: XGBoost builds one tree per class per round,
            # so multiclass training is expensive on sparse features.
            models["XGBoost"] = XGBClassifier(
                n_estimators=30,
                max_depth=6,
                learning_rate=0.3,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                tree_method="hist",
                eval_metric="mlogloss",
                verbosity=0,
                random_state=RANDOM_STATE,
            )
        except Exception as exc:  # noqa: BLE001
            logger.info("XGBoost not installed (%s); skipping.", exc)

    try:  # LightGBM (optional)
        from lightgbm import LGBMClassifier

        models["LightGBM"] = LGBMClassifier(
            n_estimators=120,
            learning_rate=0.15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            n_jobs=-1,
            verbose=-1,
            random_state=RANDOM_STATE,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("LightGBM not installed (%s); skipping.", exc)

    return models


def has_probabilities(model) -> bool:
    """Return True if ``model`` exposes ``predict_proba``."""
    return callable(getattr(model, "predict_proba", None))


# --------------------------------------------------------------------------- #
# Feature strategy auto-selection
# --------------------------------------------------------------------------- #
def select_best_feature_strategy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    strategies: tuple[str, ...] | None = None,
    max_features: int = 50_000,
    subsample: int = COMPARISON_SUBSAMPLE,
) -> tuple[str, float]:
    """Pick the feature strategy with the highest 5-fold CV accuracy.

    A logistic-regression proxy is used so the search stays fast.

    Args:
        X_train: Raw cleaned texts for training.
        y_train: Encoded labels.
        strategies: Feature strategies to try (defaults to all registered).
        max_features: Feature budget for the sparse representations.
        subsample: How many rows to use for the (fast) search.

    Returns:
        Tuple of ``(best_strategy, best_cv_accuracy)``.
    """
    strategies = strategies or supported_strategies()
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=min(subsample, len(X_train)), replace=False)
    X_s, y_s = X_train[idx], y_train[idx]

    results: list[tuple[str, float]] = []
    for strategy in strategies:
        logger.info("Feature search: evaluating strategy %r ...", strategy)
        try:
            vectorizer = get_vectorizer(strategy, max_features=max_features)
            X_s_mat = vectorizer.fit_transform(X_s)
        except Exception as exc:  # noqa: BLE001 - a strategy must never kill training
            logger.warning("Strategy %r failed (%s); skipping.", strategy, exc)
            continue

        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        scores = cross_val_score(model, X_s_mat, y_s, cv=skf, scoring="accuracy", n_jobs=-1)
        mean = float(scores.mean())
        results.append((strategy, mean))
        logger.info("  %-14s cv_accuracy=%.4f", strategy, mean)

    if not results:
        raise RuntimeError("No feature strategy could be evaluated successfully")

    results.sort(key=lambda kv: kv[1], reverse=True)
    best_strategy, best_score = results[0]
    logger.info("Best feature strategy: %s (%.4f)", best_strategy, best_score)
    save_json(
        {"ranking": [{"strategy": s, "cv_accuracy": round(a, 4)} for s, a in results],
         "best": best_strategy},
        REPORTS_DIR / "feature_strategy_selection.json",
    )
    return best_strategy, best_score


# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
def compare_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_strategy: str,
    max_features: int = 50_000,
    subsample: int = COMPARISON_SUBSAMPLE,
) -> pd.DataFrame:
    """Compare every classifier in the registry using 5-fold CV.

    All models are evaluated on the exact same feature matrix for a fair
    comparison. The matrix is built from a subsample to keep the comparison
    fast while remaining representative.

    Args:
        X_train: Raw cleaned training texts.
        y_train: Encoded training labels.
        feature_strategy: The chosen feature representation.
        max_features: Sparse feature budget for the comparison matrix.
        subsample: Rows used for comparison.

    Returns:
        DataFrame indexed by model name with CV results.
    """
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_train), size=min(subsample, len(X_train)), replace=False)
    X_s, y_s = X_train[idx], y_train[idx]

    vectorizer = get_vectorizer(feature_strategy, max_features=max_features)
    X_mat = vectorizer.fit_transform(X_s)
    logger.info("Comparison matrix: %s", X_mat.shape)

    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    rows: list[dict] = []
    for name, model in _build_models().items():
        logger.info("Comparing model %r ...", name)
        try:
            # n_jobs=1 for the folds: each model already parallelizes internally
            # (XGBoost / LightGBM / Random Forest). Nested parallelism would
            # oversubscribe the CPU and freeze the comparison.
            scores = cross_val_score(model, X_mat, y_s, cv=skf, scoring="accuracy", n_jobs=1)
            rows.append({
                "model": name,
                "cv_mean": round(float(scores.mean()), 4),
                "cv_std": round(float(scores.std()), 4),
                "cv_min": round(float(scores.min()), 4),
                "cv_max": round(float(scores.max()), 4),
                "has_proba": has_probabilities(model),
            })
            logger.info("  %-24s mean=%.4f std=%.4f", name, scores.mean(), scores.std())
        except Exception as exc:  # noqa: BLE001 - one bad model must not abort
            logger.warning("Model %r failed during comparison: %s", name, exc)
            rows.append({"model": name, "cv_mean": np.nan, "cv_std": np.nan,
                         "cv_min": np.nan, "cv_max": np.nan, "has_proba": has_probabilities(model)})

    results = pd.DataFrame(rows).set_index("model").sort_values("cv_mean", ascending=False)
    results.to_csv(REPORTS_DIR / "model_comparison.csv", encoding="utf-8")
    logger.info("Model comparison saved to reports/model_comparison.csv")
    return results


def select_final_model(comparison: pd.DataFrame) -> str:
    """Choose the model to deploy from the comparison table.

    Strategy: take the model with the highest mean CV accuracy. If that model
    cannot output probabilities, prefer the best *probabilistic* model when it
    is within a small tolerance, otherwise keep the winner (it will be
    calibrated later).

    Args:
        comparison: DataFrame produced by :func:`compare_models`.

    Returns:
        Name of the chosen model.
    """
    valid = comparison.dropna(subset=["cv_mean"])
    if valid.empty:
        raise RuntimeError("No model produced valid CV results")

    best_name = valid["cv_mean"].idxmax()
    best_acc = valid.loc[best_name, "cv_mean"]

    # Prefer a probabilistic model unless the non-probabilistic winner is
    # clearly better (>= 0.3 percentage points).
    tol = 0.003
    proba_candidates = valid[valid["has_proba"]]
    if best_acc < proba_candidates["cv_mean"].max() + tol and best_name not in proba_candidates.index:
        chosen = proba_candidates["cv_mean"].idxmax()
        logger.info("Chose probabilistic %r (%.4f) over %r (%.4f)",
                    chosen, proba_candidates.loc[chosen, "cv_mean"], best_name, best_acc)
        return chosen
    return best_name


# --------------------------------------------------------------------------- #
# Final training
# --------------------------------------------------------------------------- #
def final_training_config(model_name: str) -> tuple[int, int | None]:
    """Return ``(max_features, max_rows)`` for the final fit of ``model_name``.

    Tree/boosting models are expensive on very wide sparse matrices, so they
    get a reduced feature budget and an optional row cap.

    Args:
        model_name: Name of the chosen model.

    Returns:
        Tuple of ``max_features`` and ``max_rows`` (``None`` = use everything).
    """
    if model_name in {"Random Forest", "XGBoost"}:
        return 30_000, 20_000
    if model_name == "LightGBM":
        return 60_000, 30_000
    return MAX_FEATURES, None


@timed
def train_and_evaluate(
    force_download: bool = False,
    quick: bool = False,
) -> dict:
    """Run the complete training + evaluation pipeline.

    Args:
        force_download: Rebuild the dataset from source.
        quick: Use small subsamples everywhere (useful for CI).

    Returns:
        Metadata dict describing the trained artifacts.
    """
    # 1. Data
    df = load_dataset(force_download=force_download)
    df = deduplicate(df)
    logger.info("Training data ready: %s", df.shape)

    texts = clean_series(df["text"])
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df["language"])

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        texts.to_numpy(), y, test_size=TEST_SIZE, stratify=y,
        random_state=RANDOM_STATE,
    )
    logger.info("Train/test split: %d / %d", len(X_train), len(X_test))

    subsample = 4_000 if quick else COMPARISON_SUBSAMPLE

    # 3. Feature auto-selection
    feature_strategy, _ = select_best_feature_strategy(
        X_train, y_train, max_features=30_000 if quick else 50_000,
        subsample=subsample,
    )

    # 4. Model comparison + selection
    comparison = compare_models(
        X_train, y_train, feature_strategy,
        max_features=30_000 if quick else 50_000,
        subsample=subsample,
    )
    model_name = select_final_model(comparison)
    logger.info("Deploying model: %s", model_name)

    # 5. Final fit on the full training split
    max_features, max_rows = final_training_config(model_name)
    vectorizer = get_vectorizer(feature_strategy, max_features=max_features)
    X_train_final = vectorizer.fit_transform(X_train)
    X_test_mat = vectorizer.transform(X_test)

    rng = np.random.RandomState(RANDOM_STATE)
    if max_rows and len(X_train_final) > max_rows:
        keep = rng.choice(len(X_train_final), size=max_rows, replace=False)
        X_train_final, y_train_final = X_train_final[keep], y_train[keep]
    else:
        y_train_final = y_train

    model = _build_models()[model_name]
    logger.info("Fitting %s on %d samples with %d features ...",
                model_name, X_train_final.shape[0], X_train_final.shape[1])
    model.fit(X_train_final, y_train_final)

    # 6. Probability support for models without predict_proba
    if not has_probabilities(model):
        from sklearn.calibration import CalibratedClassifierCV

        logger.info("Calibrating %s for probability outputs ...", model_name)
        model = CalibratedClassifierCV(model, cv=3, method="sigmoid")
        model.fit(X_train_final, y_train_final)

    # 7. Evaluate on held-out test set
    metrics = evaluate_model(
        model, X_test_mat, y_test, label_encoder, REPORTS_DIR,
        prefix="final", X_train=X_train_final, y_train=y_train_final,
    )

    # 8. Persist everything
    meta = {
        "model": model_name,
        "feature_strategy": feature_strategy,
        "n_languages": int(label_encoder.classes_.size),
        "languages": label_encoder.classes_.tolist(),
        "train_samples": int(X_train.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "n_features": int(X_train_final.shape[1]),
        "accuracy": metrics["accuracy"],
        "f1_macro": metrics["f1_macro"],
        "roc_auc_macro": metrics.get("roc_auc_macro"),
        "model_comparison": comparison["cv_mean"].to_dict(),
        "random_state": RANDOM_STATE,
        "has_probabilities": has_probabilities(model),
    }
    save_pipeline(
        model=model,
        vectorizer=vectorizer,
        label_encoder=label_encoder,
        meta=meta,
        model_dir=MODEL_DIR,
    )
    logger.info("Artifacts saved to %s", MODEL_DIR)
    return meta


if __name__ == "__main__":
    import argparse

    setup_logging()
    parser = argparse.ArgumentParser(description="Train the language identification model.")
    parser.add_argument("--force-download", action="store_true",
                        help="Rebuild the dataset from source.")
    parser.add_argument("--quick", action="store_true",
                        help="Use small subsamples (CI mode).")
    args = parser.parse_args()
    train_and_evaluate(force_download=args.force_download, quick=args.quick)
