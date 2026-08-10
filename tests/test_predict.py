"""Unit tests for the prediction pipeline and model persistence."""

from __future__ import annotations

import pytest

from src.predict import PredictionError, load_pipeline, predict_language


def test_predict_returns_expected_keys(predictor):
    result = predictor.predict("Hello, how are you?")
    for key in ("language", "language_code", "confidence", "probabilities",
                "top_k", "prediction_time_ms", "input_text", "source"):
        assert key in result, f"missing key: {key}"


def test_predict_english(predictor):
    result = predictor.predict("Hello, how are you? This is a test.")
    assert result["language"] == "English"


def test_predict_multilingual_samples(predictor, sample_texts):
    for text, expected in sample_texts:
        result = predictor.predict(text)
        assert result["language"] == expected, f"{text!r} -> {result['language']}"


def test_predict_short_text(predictor):
    # Even a 2-3 character input must not crash.
    result = predictor.predict("Hola")
    assert result["confidence"] > 0


def test_predict_handles_punctuation_and_caps(predictor):
    a = predictor.predict("HELLO, WORLD!!! How are you??")
    b = predictor.predict("hello world how are you")
    assert a["language"] == b["language"] == "English"


def test_predict_handles_urls_and_emojis(predictor):
    result = predictor.predict("Visit https://example.com now! \U0001F600\nThanks!")
    assert result["language"] == "English"


def test_predict_empty_text_raises(predictor):
    with pytest.raises(PredictionError):
        predictor.predict("")
    with pytest.raises(PredictionError):
        predictor.predict("   ")
    with pytest.raises(PredictionError):
        predictor.predict("\U0001F600\U0001F601")  # emoji-only


def test_predict_top_k_order(predictor):
    result = predictor.predict("Ich liebe die deutsche Sprache sehr")
    probs = [p["confidence"] for p in result["top_k"]]
    assert probs == sorted(probs, reverse=True)
    assert len(result["top_k"]) <= 5
    assert result["top_k"][0]["language"] == result["language"]


def test_predict_many(predictor):
    texts = ["Hello there", "Bonjour le monde", "नमस्ते"]
    results = predictor.predict_many(texts)
    assert len(results) == 3
    assert all(r["language"] for r in results)


def test_predict_language_module_level(predictor):
    result = predict_language("Ciao, come stai?")
    assert result["language"] == "Italian"


def test_predict_from_csv(tmp_path, predictor):
    import pandas as pd

    csv_path = tmp_path / "input.csv"
    pd.DataFrame({"text": ["Hello world", "Bonjour le monde"]}).to_csv(csv_path, index=False)
    out = predictor.predict_from_csv(csv_path)
    assert "predicted_language" in out.columns
    assert out.loc[0, "predicted_language"] == "English"
    assert out.loc[1, "predicted_language"] == "French"


def test_model_loading_produces_predictor(predictor):
    from src.config import MODEL_DIR

    loaded = load_pipeline(MODEL_DIR)
    assert loaded is not None
    assert len(loaded.languages) >= 20


def test_predictor_metadata(predictor):
    assert "model" in predictor.meta
    assert "feature_strategy" in predictor.meta
    assert predictor.meta["n_languages"] == len(predictor.languages)
