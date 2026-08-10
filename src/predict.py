"""Prediction pipeline and model persistence.

The central object is :class:`LanguagePredictor`, a thin wrapper around the
trained model / vectorizer / label encoder that exposes a friendly
``predict`` API returning the language, a confidence score, and a probability
distribution over the supported languages.

Run a quick demo with::

    python -m src.predict
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ENABLE_FALLBACK_DETECTOR,
    FALLBACK_THRESHOLD,
    LANGDETECT_TO_NAME,
    MODEL_DIR,
)
from src.preprocessing import clean_text
from src.utils import get_logger, load_json, save_json

logger = get_logger("predict")


class PredictionError(ValueError):
    """Raised when the input text cannot be classified."""


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_pipeline(
    model,
    vectorizer,
    label_encoder,
    meta: dict,
    model_dir: Path = MODEL_DIR,
) -> None:
    """Persist model + vectorizer + label encoder + metadata to disk.

    Args:
        model: Fitted classifier.
        vectorizer: Fitted text transformer.
        label_encoder: Fitted ``LabelEncoder``.
        meta: Arbitrary metadata dict (also written as JSON).
        model_dir: Output directory.
    """
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / "best_model.joblib"
    vectorizer_file = model_dir / "vectorizer.joblib"
    label_encoder_file = model_dir / "label_encoder.joblib"
    metadata_file = model_dir / "model_meta.json"

    joblib.dump(model, model_file)
    joblib.dump(vectorizer, vectorizer_file)
    joblib.dump(label_encoder, label_encoder_file)
    save_json(meta, metadata_file)
    logger.info("Pipeline saved: model, vectorizer, label encoder, metadata")


def load_pipeline(model_dir: Path = MODEL_DIR) -> "LanguagePredictor":
    """Load a persisted pipeline from ``model_dir``.

    Args:
        model_dir: Directory containing the four artifact files.

    Returns:
        A ready-to-use :class:`LanguagePredictor`.

    Raises:
        FileNotFoundError: If any artifact is missing.
    """
    model_dir = Path(model_dir)
    model_file = model_dir / "best_model.joblib"
    vectorizer_file = model_dir / "vectorizer.joblib"
    label_encoder_file = model_dir / "label_encoder.joblib"
    metadata_file = model_dir / "model_meta.json"

    required = (model_file, vectorizer_file, label_encoder_file)
    missing = [p for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model artifacts in {model_dir}: {[p.name for p in missing]}. "
            "Run `python -m src.train` first."
        )

    model = joblib.load(model_file)
    vectorizer = joblib.load(vectorizer_file)
    label_encoder = joblib.load(label_encoder_file)
    meta = load_json(metadata_file) if metadata_file.exists() else {}
    return LanguagePredictor(model, vectorizer, label_encoder, meta)


# --------------------------------------------------------------------------- #
# Predictor
# --------------------------------------------------------------------------- #
class LanguagePredictor:
    """Language identification wrapper with a friendly prediction API."""

    def __init__(self, model, vectorizer, label_encoder, meta: dict | None = None) -> None:
        """Create a predictor from fitted artifacts.

        Args:
            model: Classifier with ``predict`` (and ideally ``predict_proba``).
            vectorizer: Fitted text transformer.
            label_encoder: Fitted ``LabelEncoder``.
            meta: Optional metadata (languages list, model name, ...).
        """
        self.model = model
        self.vectorizer = vectorizer
        self.label_encoder = label_encoder
        self.meta = meta or {}
        self.languages: list[str] = list(self.label_encoder.classes_)

        # Display name -> ISO 639-1 code (when known) for the response.
        self._code_map: dict[str, str] = {
            name: code for code, name in LANGDETECT_TO_NAME.items()
        }

    # -- public API -------------------------------------------------------- #
    def predict(self, text: str, top_k: int = 5) -> dict:
        """Predict the language of a single text.

        Args:
            text: Raw input text.
            top_k: How many ranked predictions to return.

        Returns:
            Dict with keys: ``language``, ``language_code``, ``confidence``,
            ``probabilities``, ``top_k``, ``prediction_time_ms``,
            ``input_text`` and ``source`` (``"model"`` or ``"fallback"``).

        Raises:
            PredictionError: If the text is empty after cleaning.
        """
        start = time.perf_counter()
        cleaned = clean_text(text)
        if not cleaned:
            raise PredictionError("Text is empty after cleaning (spaces/emojis only).")

        vector = self.vectorizer.transform([cleaned])
        probs = self._probabilities(vector)
        ranked = self._ranked(probs, top_k)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        language, confidence = ranked[0]["language"], ranked[0]["confidence"]

        # Optional fallback for very short/ambiguous inputs.
        source = "model"
        if ENABLE_FALLBACK_DETECTOR and confidence < FALLBACK_THRESHOLD:
            fallback_lang = self._fallback_detect(cleaned)
            if fallback_lang and fallback_lang in self.languages:
                language = fallback_lang
                confidence = round(float(np.max(probs)), 4)  # keep model confidence
                source = "fallback"
                ranked = self._ranked(probs, top_k, force_first=fallback_lang)

        return {
            "language": language,
            "language_code": self._code_map.get(language, ""),
            "confidence": confidence,
            "probabilities": ranked,
            "top_k": ranked,
            "prediction_time_ms": round(elapsed_ms, 2),
            "input_text": text,
            "source": source,
        }

    def predict_many(self, texts: list[str], top_k: int = 5) -> list[dict]:
        """Predict several texts, returning a list of result dicts."""
        return [self.predict(t, top_k=top_k) for t in texts]

    def predict_from_csv(self, csv_path: str | Path, text_column: str = "text") -> pd.DataFrame:
        """Run batch prediction on a CSV file.

        Adds ``predicted_language`` and ``confidence`` columns.

        Args:
            csv_path: Path to a CSV containing a ``text`` column.
            text_column: Name of the text column.

        Returns:
            The original DataFrame augmented with predictions.
        """
        df = pd.read_csv(csv_path)
        if text_column not in df.columns:
            raise PredictionError(f"CSV must contain a '{text_column}' column, got {list(df.columns)}")

        results = df[text_column].map(lambda t: self.predict(str(t)) if str(t).strip() else None)
        df["predicted_language"] = [
            r["language"] if r else "" for r in results
        ]
        df["confidence"] = [
            r["confidence"] if r else 0.0 for r in results
        ]
        return df

    # -- internals --------------------------------------------------------- #
    def _probabilities(self, vector) -> np.ndarray:
        """Return class probabilities for one vectorized document (2D)."""
        if callable(getattr(self.model, "predict_proba", None)):
            return np.asarray(self.model.predict_proba(vector))
        # No native probabilities: softmax over decision scores.
        scores = np.asarray(self.model.decision_function(vector))
        if scores.ndim == 1:  # binary estimator
            scores = np.column_stack([-scores, scores])
        exp = np.exp(scores - scores.max(axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def _ranked(self, probs: np.ndarray, top_k: int, force_first: str | None = None) -> list[dict]:
        """Convert a probability vector into a ranked top-k list."""
        probs_1d = probs[0]
        order = np.argsort(probs_1d)[::-1][:top_k]
        ranked = [
            {
                "language": self.languages[i],
                "language_code": self._code_map.get(self.languages[i], ""),
                "confidence": round(float(probs_1d[i]), 6),
            }
            for i in order
        ]
        if force_first and force_first in self.languages:
            # Move the fallback language to the top, keep the rest ordered.
            ranked = [item for item in ranked if item["language"] == force_first] + [
                item for item in ranked if item["language"] != force_first
            ]
        return ranked

    def _fallback_detect(self, cleaned: str) -> str | None:
        """Best-effort fallback detection via ``langdetect``."""
        try:
            from langdetect import detect
            from langdetect.lang_detect_exception import LangDetectException

            code = detect(cleaned)
            return LANGDETECT_TO_NAME.get(code, "")
        except (ImportError, LangDetectException, ValueError):
            return None

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return (
            f"LanguagePredictor(model={self.meta.get('model', type(self.model).__name__)}, "
            f"languages={len(self.languages)}, accuracy={self.meta.get('accuracy')})"
        )


# --------------------------------------------------------------------------- #
# Module-level convenience
# --------------------------------------------------------------------------- #
_predictor_cache: LanguagePredictor | None = None


def get_predictor(model_dir: Path = MODEL_DIR) -> LanguagePredictor:
    """Return a cached (module-level) predictor for convenience.

    The pipeline is loaded once and reused on subsequent calls, which keeps
    prediction fast in long-running services (API / Streamlit / tests).

    Args:
        model_dir: Directory containing the saved artifacts.

    Returns:
        A :class:`LanguagePredictor` instance.
    """
    global _predictor_cache  # noqa: PLW0603
    if _predictor_cache is None:
        _predictor_cache = load_pipeline(model_dir)
    return _predictor_cache


def predict_language(text: str, top_k: int = 5, model_dir: Path = MODEL_DIR) -> dict:
    """Predict the language of ``text`` using the cached pipeline.

    Convenience wrapper around :meth:`LanguagePredictor.predict`.

    Args:
        text: Raw input text.
        top_k: Number of ranked predictions to return.
        model_dir: Directory containing the saved artifacts.

    Returns:
        The prediction dict (see :meth:`LanguagePredictor.predict`).
    """
    return get_predictor(model_dir).predict(text, top_k=top_k)


if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()
    predictor = load_pipeline()
    samples = [
        "Hello, how are you?",
        "Bonjour tout le monde",
        "नमस्ते आप कैसे हैं",
        "வணக்கம்",
        "ഹലോ സുഖമാണോ",
        "Guten Morgen, wie geht es dir?",
    ]
    for s in samples:
        result = predictor.predict(s)
        print(f"{s!r:45} -> {result['language']:12} confidence={result['confidence']:.2%}")
