"""Shared pytest fixtures and helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure the project root is importable regardless of the CWD used by pytest.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODEL_DIR  # noqa: E402


@pytest.fixture(scope="session")
def predictor():
    """Return the trained predictor, skipping tests if no model is available.

    This lets the suite run on a fresh checkout before ``python -m src.train``
    has been executed (CI trains a quick model first).
    """
    from src.predict import get_predictor

    if not (MODEL_DIR / "best_model.joblib").exists():
        pytest.skip("Trained model not found; run `python -m src.train` first.")
    return get_predictor(MODEL_DIR)


@pytest.fixture(scope="session")
def sample_texts() -> list[str]:
    """A few labelled texts across scripts for sanity checks."""
    return [
        ("Hello, how are you? I hope you are well.", "English"),
        ("Bonjour tout le monde, comment allez-vous ?", "French"),
        ("Guten Morgen, wie geht es dir heute?", "German"),
        ("Hola, ¿cómo estás hoy?", "Spanish"),
        ("नमस्ते आप कैसे हैं।", "Hindi"),
        ("வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்?", "Tamil"),
        ("ഹലോ, നിങ്ങൾക്ക് സുഖമാണോ?", "Malayalam"),
        ("こんにちは、今日はお元気ですか。", "Japanese"),
        ("안녕하세요, 오늘 기분이 어떠세요?", "Korean"),
        ("Привет, как у тебя дела сегодня?", "Russian"),
    ]
