"""Unit tests for the feature-engineering module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from src.feature_engineering import (
    CharacterFrequencyVectorizer,
    UnicodeBlockVectorizer,
    get_vectorizer,
    supported_strategies,
)

CORPUS = [
    "This is a simple English sentence to test the feature builders.",
    "Das ist ein einfacher deutscher Satz, der die Merkmale testet.",
    "Esta es una frase sencilla en español para probar características.",
    "Questa è una frase semplice in italiano per testare le caratteristiche.",
    "C'est une phrase simple en français pour tester les caractéristiques.",
    "これは日本語のテスト文です。今日はとても良い天気ですね。",
    "これは難しい日本語の文章です。新しい機能を確認します。",
    "नमस्ते, यह एक हिंदी वाक्य है। मैं आपके बारे में सुनकर खुश हूं।",
    "வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்? நன்றி.",
    "ഹലോ, നിങ്ങൾക്ക് സുഖമാണോ? ഇന്ന് എന്തുണ്ട് വിശേഷം.",
    "안녕하세요, 오늘 기분이 어떠세요? 좋은 하루 되세요.",
    "Привет, как у тебя дела сегодня? У меня всё хорошо.",
    "Merhaba, bugün nasılsınız? Umarım iyi bir gün geçiriyorsunuzdur.",
    "مرحبا، كيف حالك اليوم؟ أتمنى أن يكون يومك جيدا.",
    "สวัสดีครับ วันนี้คุณเป็นอย่างไรบ้างครับ",
    "Xin chào, hôm nay bạn khỏe không? Rất vui được gặp bạn.",
]


@pytest.mark.parametrize("strategy", supported_strategies())
def test_vectorizer_fit_transform_shapes(strategy):
    vec = get_vectorizer(strategy, max_features=20_000)
    matrix = vec.fit_transform(CORPUS)
    assert matrix.shape[0] == len(CORPUS)
    assert matrix.shape[1] > 0
    # Re-application must work (fitted transformer).
    matrix2 = vec.transform(CORPUS)
    assert matrix2.shape == matrix.shape


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        get_vectorizer("not-a-strategy")


def test_tfidf_char_is_sparse():
    vec = get_vectorizer("tfidf_char", max_features=20_000)
    matrix = vec.fit_transform(CORPUS)
    assert sparse.issparse(matrix)


def test_unicode_blocks_dense_and_normalized():
    vec = UnicodeBlockVectorizer()
    matrix = vec.fit_transform(["hello", "مرحبا بالعالم", "こんにちは"])
    assert matrix.shape == (3, len(UnicodeBlockVectorizer().blocks))
    # Rows are length-normalized -> sum close to 1.0
    np.testing.assert_allclose(matrix.sum(axis=1), np.ones(3), atol=1e-6)


def test_character_frequency_vocab_learned():
    vec = CharacterFrequencyVectorizer(n_features=100, min_df=1)
    matrix = vec.fit_transform(CORPUS)
    assert matrix.shape[1] == 100
    assert len(vec.vocabulary_) == 100


def test_character_frequency_supports_repeat_transform():
    vec = CharacterFrequencyVectorizer(n_features=50, min_df=1)
    vec.fit(CORPUS)
    m1 = vec.transform(CORPUS)
    m2 = vec.transform(["a very different string"])
    assert m1.shape[1] == m2.shape[1] == 50


def test_feature_names_present():
    uvec = UnicodeBlockVectorizer().fit(CORPUS)
    names = uvec.get_feature_names_out()
    assert "tamil" in list(names) or "basic_latin" in list(names)
