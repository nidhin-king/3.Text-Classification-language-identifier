"""Unit tests for the preprocessing module."""

from __future__ import annotations

import pandas as pd

from src.preprocessing import (
    clean_series,
    clean_text,
    collapse_whitespace,
    deduplicate,
    normalize_unicode,
    remove_emoji,
    remove_urls,
)


def test_normalize_unicode_nfc():
    assert normalize_unicode("e\u0301") == "\u00e9"


def test_remove_urls():
    assert " " in remove_urls("see https://example.com/page now")
    assert "http://example.com" not in remove_urls("http://example.com")


def test_remove_emoji():
    cleaned = remove_emoji("hello \U0001F600 world")
    assert "\U0001F600" not in cleaned


def test_clean_text_lowercase_and_strip():
    assert clean_text("  HELLO   WORLD  ") == "hello world"


def test_clean_text_handles_none():
    assert clean_text(None) == ""
    assert clean_text("") == ""


def test_clean_text_removes_urls_and_emails():
    result = clean_text("Contact test@example.com or visit https://x.io")
    assert "@" not in result
    assert "https" not in result


def test_clean_text_keeps_numbers_by_default():
    assert clean_text("I have 42 apples") == "i have 42 apples"


def test_clean_text_can_remove_numbers():
    config = {"remove_numbers": True}
    assert "42" not in clean_text("I have 42 apples", config=config)


def test_clean_text_emoji_removed():
    assert clean_text("nice \U0001F60A") == "nice"


def test_collapse_whitespace():
    assert collapse_whitespace("a\t  b\n c") == "a b c"


def test_clean_series_vectorized():
    series = pd.Series(["  Hello  ", None, "Bonjour", ""])
    result = clean_series(series)
    assert result.tolist() == ["hello", "", "bonjour", ""]


def test_deduplicate_removes_duplicates_and_empties():
    df = pd.DataFrame({"text": ["a", "a", "", "  ", "b"], "language": ["x", "x", "y", "y", "z"]})
    out = deduplicate(df)
    assert out["text"].tolist() == ["a", "b"]
    assert len(out) == 2


def test_deduplicate_no_rows_is_safe():
    empty = pd.DataFrame({"text": [], "language": []})
    out = deduplicate(empty)
    assert out.empty
