"""Unit tests for loading and using the persisted model artifacts."""

from __future__ import annotations

import pytest

from src.config import (
    LABEL_ENCODER_FILE,
    METADATA_FILE,
    MODEL_DIR,
    MODEL_FILE,
    VECTORIZER_FILE,
)
from src.predict import load_pipeline


def test_all_artifacts_exist(predictor):
    for path in (MODEL_FILE, VECTORIZER_FILE, LABEL_ENCODER_FILE, METADATA_FILE):
        assert path.exists(), f"missing artifact: {path}"


def test_artifacts_are_non_trivial(predictor):
    assert MODEL_FILE.stat().st_size > 1_000
    assert VECTORIZER_FILE.stat().st_size > 1_000


def test_load_pipeline_roundtrip(predictor):
    loaded = load_pipeline(MODEL_DIR)
    result = loaded.predict("This is a roundtrip test")
    assert result["language"] == "English"
    assert result["confidence"] > 0.5


def test_metadata_json_valid(predictor):
    import json

    meta = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    assert meta["n_languages"] == len(predictor.languages)
    assert meta["accuracy"] > 0.5
    assert "languages" in meta


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path)
