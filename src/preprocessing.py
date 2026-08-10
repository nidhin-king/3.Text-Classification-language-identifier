"""Text preprocessing for language identification.

The pipeline is intentionally configurable (see :data:`PREPROCESS_CONFIG` in
:mod:`src.config`) and safe to apply to any string. Because language detection
is driven mainly by character statistics, the cleaners here are conservative:
we remove things that carry *no* language signal (URLs, emails, emojis,
control characters) and normalize casing/whitespace, while keeping
punctuation and digits unless explicitly configured otherwise.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from src.config import PREPROCESS_CONFIG
from src.utils import get_logger

logger = get_logger("preprocessing")

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_WHITESPACE_RE = re.compile(r"\s+")
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"  # misc symbols, pictographs, emoticons, flags
    "\U00002600-\U000027BF"  # misc symbols + dingbats
    "\U0001F1E6-\U0001F1FF"  # regional indicator symbols (flags)
    "\U0000FE0F"             # variation selector-16 (emoji presentation)
    "\U0000200D"             # zero-width joiner
    "\U00002B50\U00002764\U00002764\uFE0F"
    "]+",
    flags=re.UNICODE,
)
_NUMBER_RE = re.compile(r"\d+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def normalize_unicode(text: str) -> str:
    """Normalize text to NFC (canonical composition) form.

    NFC merges visually-identical composed characters (e.g. ``e`` + combining
    accent -> ``é``), which keeps the vocabulary of character n-grams smaller
    and more consistent.

    Args:
        text: Input string.

    Returns:
        Normalized string.
    """
    return unicodedata.normalize("NFC", text)


def remove_urls(text: str) -> str:
    """Strip HTTP(S) and bare ``www.`` URLs."""
    return _URL_RE.sub(" ", text)


def remove_emails(text: str) -> str:
    """Strip email addresses."""
    return _EMAIL_RE.sub(" ", text)


def remove_emoji(text: str) -> str:
    """Remove emoji / pictographic characters (language-neutral signal)."""
    return _EMOJI_RE.sub(" ", text)


def remove_numbers(text: str) -> str:
    """Replace digit runs with a single space."""
    return _NUMBER_RE.sub(" ", text)


def remove_control_chars(text: str) -> str:
    """Drop control characters that would otherwise pollute n-grams."""
    return _CTRL_RE.sub("", text)


def collapse_whitespace(text: str) -> str:
    """Replace any run of whitespace with a single ASCII space."""
    return _WHITESPACE_RE.sub(" ", text)


def strip_text(text: str) -> str:
    """Trim leading and trailing whitespace."""
    return text.strip()


def lower_text(text: str) -> str:
    """Lowercase the text.

    ``str.lower()`` is a no-op for caseless scripts (Devanagari, Arabic,
    Chinese, ...) and is the correct, safe choice for Latin/Cyrillic/Greek.
    """
    return text.lower()


def clean_text(text: str | None, config: dict[str, bool] | None = None) -> str:
    """Run the full cleaning pipeline on a single piece of text.

    Args:
        text: Raw input. ``None`` is treated as an empty string.
        config: Overrides for :data:`PREPROCESS_CONFIG`.

    Returns:
        Cleaned, whitespace-normalized string (never ``None``).
    """
    cfg = {**PREPROCESS_CONFIG, **(config or {})}
    if text is None:
        return ""
    try:  # NaN / pd.NA handling (e.g. from pandas with missing values)
        if bool(pd.isna(text)):
            return ""
    except (TypeError, ValueError):
        pass
    if not text:
        return ""

    result: str = str(text)

    if cfg.get("normalize_unicode", True):
        result = normalize_unicode(result)

    if cfg.get("remove_urls", True):
        result = remove_urls(result)
    if cfg.get("remove_emails", True):
        result = remove_emails(result)
    if cfg.get("remove_emoji", True):
        result = remove_emoji(result)
    if cfg.get("remove_control_chars", True):
        result = remove_control_chars(result)
    if cfg.get("remove_numbers", False):
        result = remove_numbers(result)

    if cfg.get("lowercase", True):
        result = lower_text(result)

    if cfg.get("collapse_whitespace", True):
        result = collapse_whitespace(result)
    if cfg.get("strip", True):
        result = strip_text(result)

    if cfg.get("remove_punctuation", False):
        result = remove_punctuation(result)

    return result


def remove_punctuation(text: str) -> str:
    """Remove punctuation characters, keeping letters, digits and spaces."""
    return "".join(
        ch
        for ch in text
        if ch.isalnum() or ch.isspace()
    )


def clean_series(series, config: dict[str, bool] | None = None):
    """Vectorized cleaning for a pandas Series of raw text.

    Args:
        series: pandas Series of strings (or missing values).
        config: Overrides for the cleaning configuration.

    Returns:
        A new Series with cleaned text (``None`` values become ``""``).
    """
    return series.map(lambda x: clean_text(x, config=config))


def deduplicate(df) -> "pd.DataFrame":
    """Remove exact duplicate rows and rows with empty text.

    Args:
        df: DataFrame with at least a ``text`` column.

    Returns:
        Deduplicated, index-reset DataFrame.
    """
    if df.empty:
        return df

    before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].str.strip().str.len() > 0]
    df = df.drop_duplicates(subset="text", keep="first")
    if len(df) < before:
        logger.info("Removed %d duplicate/empty rows", before - len(df))
    return df.reset_index(drop=True)


# Small smoke-test guard so the module can be run directly.
if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()
    samples = [
        "Hello  world!!!  https://example.com test@mail.com \U0001F600",
        "  Bonjour\tle monde   ",
        "नमस्ते आप कैसे हैं",
        None,
        "",
    ]
    for s in samples:
        print(f"{s!r:60} -> {clean_text(s)!r}")
