"""Dataset loading utilities.

Two sources are supported:

1. A fast-path CSV file (Kaggle-style ``Language Detection`` format) whose URL
   is configured in :mod:`src.config`. Used when it downloads successfully.
2. `Tatoeba <https://tatoeba.org>`_ per-language sentence exports, which are
   always publicly accessible and let us build a clean, balanced, multilingual
   dataset (30 languages in the default configuration).

The loader caches the final result to ``dataset/language_data.csv`` so later
runs are instant. Downloads use a small HTTP client from the standard library
to keep the dependency footprint minimal.
"""

from __future__ import annotations

import bz2
import io
import random
import time
import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    DATASET_CSV,
    FALLBACK_DATASET_URL,
    LANGS,
    RANDOM_STATE,
    SAMPLES_PER_LANGUAGE,
    TATOEBA_BASE_URL,
)
from src.utils import get_logger, timed

logger = get_logger("data_loader")

#: User agent that politely identifies the client to download servers.
_HEADERS = {"User-Agent": "language-identification-ml/1.0 (educational project)"}


class DatasetDownloadError(RuntimeError):
    """Raised when no dataset source could be reached."""


def _fetch(url: str, timeout: float = 60.0) -> bytes:
    """Download ``url`` and return the raw bytes."""
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def download_language_csv(url: str = FALLBACK_DATASET_URL, timeout: float = 60.0) -> pd.DataFrame:
    """Try to download the fast-path CSV dataset.

    Args:
        url: CSV download URL.
        timeout: Network timeout in seconds.

    Returns:
        DataFrame with columns ``text`` and ``language`` (names normalized).

    Raises:
        DatasetDownloadError: If the download or parse fails.
    """
    logger.info("Attempting fast-path CSV download from %s", url)
    raw = _fetch(url, timeout=timeout)
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:  # noqa: BLE001 - any parse problem becomes a clear error
        raise DatasetDownloadError(f"Could not parse CSV from {url}: {exc}") from exc

    df.columns = [str(c).strip().lower() for c in df.columns]
    if not {"text", "language"}.issubset(df.columns):
        raise DatasetDownloadError(f"CSV from {url} lacks 'text'/'language' columns: {list(df.columns)}")
    df = df[["text", "language"]].dropna()
    df["language"] = df["language"].astype(str).str.strip()
    logger.info("Fast-path dataset ready: %d rows, %d languages", len(df), df["language"].nunique())
    return df


def download_tatoeba_language(code: str, timeout: float = 120.0) -> list[str]:
    """Download one Tatoeba language export and return its sentences.

    Args:
        code: ISO 639-3 language code, e.g. ``"eng"``.
        timeout: Network timeout in seconds.

    Returns:
        List of raw sentences (without IDs / language tags).
    """
    url = TATOEBA_BASE_URL.format(code=code)
    raw = _fetch(url, timeout=timeout)
    sentences: list[str] = []
    with bz2.open(io.BytesIO(raw), "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 3 and parts[2].strip():
                sentences.append(parts[2].strip())
    return sentences


def build_from_tatoeba(
    langs: dict[str, str] | None = None,
    samples_per_language: int = SAMPLES_PER_LANGUAGE,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Build a balanced multilingual DataFrame from Tatoeba exports.

    Downloads each language file once, filters short/low-quality sentences and
    then deterministically samples ``samples_per_language`` rows per language.

    Args:
        langs: Mapping of ISO-639-3 code -> display name. Defaults to config.
        samples_per_language: How many sentences per language to keep.
        seed: Random seed for reproducible sampling.

    Returns:
        DataFrame with columns ``text`` and ``language``.
    """
    langs = langs or LANGS
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []

    for idx, (code, name) in enumerate(langs.items(), start=1):
        logger.info("[%d/%d] Downloading %s (%s)", idx, len(langs), name, code)
        try:
            sentences = download_tatoeba_language(code)
        except Exception as exc:  # noqa: BLE001 - keep building with other languages
            logger.warning("Skipping %s (%s): %s", name, code, exc)
            continue

        # Keep reasonably clean sentences (length between 8 and 400 chars).
        clean = [
            s
            for s in sentences
            if 8 <= len(s) <= 400 and any(c.isalpha() for c in s)
        ]
        if not clean:
            logger.warning("No usable sentences for %s (%s)", name, code)
            continue

        chosen = rng.sample(clean, k=min(samples_per_language, len(clean)))
        rows.extend((s, name) for s in chosen)
        logger.info("  kept %d sentences for %s", len(chosen), name)
        time.sleep(0.5)  # be polite to the download server

    if not rows:
        raise DatasetDownloadError("Tatoeba build produced no rows")

    df = pd.DataFrame(rows, columns=["text", "language"])
    logger.info("Tatoeba dataset ready: %d rows, %d languages", len(df), df["language"].nunique())
    return df


@timed
def load_dataset(force_download: bool = False) -> pd.DataFrame:
    """Return the training dataset, downloading/caching it if necessary.

    Order of preference:

    1. Return the cached ``dataset/language_data.csv`` if it exists.
    2. Try the fast-path CSV download.
    3. Fall back to building from Tatoeba.

    The chosen dataset is always cached to disk for future runs.

    Args:
        force_download: Ignore the cache and rebuild from source.

    Returns:
        DataFrame with ``text`` and ``language`` columns.
    """
    if not force_download and DATASET_CSV.exists():
        logger.info("Loading cached dataset from %s", DATASET_CSV)
        df = pd.read_csv(DATASET_CSV)
        df = df.drop_duplicates(subset="text", keep="first").dropna()
        logger.info("Cached dataset loaded: %d rows, %d languages", len(df), df["language"].nunique())
        return df

    logger.info("No usable cache found; building dataset...")
    try:
        df = download_language_csv()
    except Exception as exc:  # noqa: BLE001 - fall through to Tatoeba
        logger.info("Fast-path CSV unavailable (%s); using Tatoeba.", exc)
        df = build_from_tatoeba()

    df = df.drop_duplicates(subset="text", keep="first").dropna().reset_index(drop=True)
    DATASET_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATASET_CSV, index=False, encoding="utf-8")
    logger.info("Dataset cached to %s", DATASET_CSV)
    return df


if __name__ == "__main__":
    # CLI: python -m src.data_loader
    from src.utils import setup_logging

    setup_logging()
    frame = load_dataset(force_download=False)
    print(frame.head(10).to_string(index=False))
    print(f"\nTotal rows: {len(frame)}, languages: {frame['language'].nunique()}")
