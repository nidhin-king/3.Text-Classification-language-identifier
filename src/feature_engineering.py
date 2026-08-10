"""Feature engineering for language identification.

Implements and registers several text representations:

* ``bow``             - Bag-of-words (word 1-2 grams, raw counts)
* ``tfidf_word``      - TF-IDF word 1-2 grams
* ``word_ngrams``     - TF-IDF word 1-3 grams
* ``tfidf_char``      - TF-IDF character n-grams (``char_wb``) - usually the best
* ``char_freq``       - dense histogram over the most frequent characters
* ``unicode_blocks``  - dense counts of characters per Unicode block
* ``combined``        - TF-IDF char + word features concatenated

Every strategy is exposed through :func:`get_vectorizer` which returns a
fully-fitted scikit-learn transformer, so the training code can stay
completely agnostic of the underlying representation.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.config import (
    CHAR_NGRAM_RANGE,
    FEATURE_STRATEGIES,
    MAX_FEATURES,
    WORD_NGRAM_RANGE,
)
from src.utils import get_logger

logger = get_logger("feature_engineering")

# --------------------------------------------------------------------------- #
# Unicode block table (name, start_codepoint, end_codepoint) used by the
# UnicodeBlockVectorizer. These cover the scripts present in our 30 languages
# plus a handful of extra common ones for future extensibility.
# --------------------------------------------------------------------------- #
UNICODE_BLOCKS: tuple[tuple[str, int, int], ...] = (
    ("basic_latin", 0x0000, 0x007F),
    ("latin1_supplement", 0x0080, 0x00FF),
    ("latin_ext_a", 0x0100, 0x017F),
    ("latin_ext_b", 0x0180, 0x024F),
    ("latin_ext_additional", 0x1E00, 0x1EFF),
    ("greek", 0x0370, 0x03FF),
    ("cyrillic", 0x0400, 0x04FF),
    ("cyrillic_supp", 0x0500, 0x052F),
    ("armenian", 0x0530, 0x058F),
    ("hebrew", 0x0590, 0x05FF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic_supp", 0x0750, 0x077F),
    ("syriac", 0x0700, 0x074F),
    ("devanagari", 0x0900, 0x097F),
    ("bengali", 0x0980, 0x09FF),
    ("gurmukhi", 0x0A00, 0x0A7F),
    ("gujarati", 0x0A80, 0x0AFF),
    ("oriya", 0x0B00, 0x0B7F),
    ("tamil", 0x0B80, 0x0BFF),
    ("telugu", 0x0C00, 0x0C7F),
    ("kannada", 0x0C80, 0x0CFF),
    ("malayalam", 0x0D00, 0x0D7F),
    ("sinhala", 0x0D80, 0x0DFF),
    ("thai", 0x0E00, 0x0E7F),
    ("lao", 0x0E80, 0x0EFF),
    ("tibetan", 0x0F00, 0x0FFF),
    ("myanmar", 0x1000, 0x109F),
    ("georgian", 0x10A0, 0x10FF),
    ("hangul_jamo", 0x1100, 0x11FF),
    ("ethiopic", 0x1200, 0x137F),
    ("cherokee", 0x13A0, 0x13FF),
    ("canadian_aboriginal", 0x1400, 0x167F),
    ("khmer", 0x1780, 0x17FF),
    ("mongolian", 0x1800, 0x18AF),
    ("hiragana", 0x3040, 0x309F),
    ("katakana", 0x30A0, 0x30FF),
    ("bopomofo", 0x3100, 0x312F),
    ("cjk_unified", 0x4E00, 0x9FFF),
    ("cjk_ext_a", 0x3400, 0x4DBF),
    ("cjk_symbols_punct", 0x3000, 0x303F),
    ("hangul_syllables", 0xAC00, 0xD7AF),
    ("latin_ipa_ext", 0x0250, 0x02AF),
    ("spacing_modifiers", 0x02B0, 0x02FF),
    ("combining_marks", 0x0300, 0x036F),
    ("general_punct", 0x2000, 0x206F),
    ("currency_symbols", 0x20A0, 0x20CF),
    ("letterlike_symbols", 0x2100, 0x214F),
    ("number_forms", 0x2150, 0x218F),
    ("arrows", 0x2190, 0x21FF),
    ("math_operators", 0x2200, 0x22FF),
    ("misc_technical", 0x2300, 0x23FF),
    ("box_drawing", 0x2500, 0x257F),
    ("block_elements", 0x2580, 0x259F),
    ("geometric_shapes", 0x25A0, 0x25FF),
    ("misc_symbols", 0x2600, 0x26FF),
    ("dingbats", 0x2700, 0x27BF),
    ("misc_math_symbols_b", 0x2980, 0x29FF),
    ("cjk_radicals_supp", 0x2E80, 0x2EFF),
    ("kangxi_radicals", 0x2F00, 0x2FDF),
    ("emoticons", 0x1F600, 0x1F64F),
    ("transport_map", 0x1F680, 0x1F6FF),
    ("flags", 0x1F1E6, 0x1F1FF),
    ("private_use", 0xE000, 0xF8FF),
)


class UnicodeBlockVectorizer(BaseEstimator, TransformerMixin):
    """Dense features: share of characters belonging to each Unicode block.

    The resulting vector encodes the *script* of a text which is already a
    strong discriminator between many languages (e.g. Tamil vs Malayalam both
    use Indic scripts, but Russian vs English differ at the block level).
    """

    def __init__(self, blocks: tuple[tuple[str, int, int], ...] = UNICODE_BLOCKS) -> None:
        self.blocks = blocks

    def fit(self, X, y=None):  # noqa: N803 - sklearn interface
        """Stateless transformer: nothing to learn."""
        return self

    def transform(self, X):  # noqa: N803 - sklearn interface
        """Return an ``(n_samples, n_blocks)`` float array (length normalized)."""
        rows = []
        for doc in X:
            total = len(doc) or 1
            counts = np.zeros(len(self.blocks), dtype=np.float64)
            for ch in doc:
                code = ord(ch)
                for idx, (_, start, end) in enumerate(self.blocks):
                    if start <= code <= end:
                        counts[idx] += 1.0
                        break
            rows.append(counts / total)
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        return np.asarray([name for name, _, _ in self.blocks], dtype=object)


class CharacterFrequencyVectorizer(BaseEstimator, TransformerMixin):
    """Dense features: normalized frequency of the ``n_features`` most common
    characters observed in the training corpus.

    The vocabulary is learned at fit time (document-frequency ordering), which
    keeps the representation compact yet robust across scripts.
    """

    def __init__(self, n_features: int = 300, min_df: int = 5) -> None:
        self.n_features = n_features
        self.min_df = min_df

    def fit(self, X, y=None):  # noqa: N803 - sklearn interface
        """Learn the top-``n_features`` characters from ``X``."""
        counter: Counter[str] = Counter()
        for doc in X:
            counter.update(set(doc))  # document frequency
        common = [
            ch for ch, count in counter.most_common()
            if count >= self.min_df
        ][: self.n_features]
        if not common:
            raise ValueError("No characters found to build the frequency vocabulary")
        self.vocabulary_ = common
        self.vocab_index_ = {ch: i for i, ch in enumerate(common)}
        return self

    def transform(self, X):  # noqa: N803 - sklearn interface
        """Return an ``(n_samples, n_features)`` float array (length normalized)."""
        rows = []
        for doc in X:
            total = len(doc) or 1
            vec = np.zeros(len(self.vocabulary_), dtype=np.float64)
            for ch in doc:
                idx = self.vocab_index_.get(ch)
                if idx is not None:
                    vec[idx] += 1.0
            rows.append(vec / total)
        return np.asarray(rows, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):  # noqa: ANN001
        return np.asarray(self.vocabulary_, dtype=object)


# --------------------------------------------------------------------------- #
# Vectorizer registry
# --------------------------------------------------------------------------- #
def _tfidf_char(max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """TF-IDF character n-grams (whitespace-aware ``char_wb`` analyzer)."""
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=CHAR_NGRAM_RANGE,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        max_df=0.99,
        lowercase=False,  # preprocessing already handled casing
    )


def _tfidf_word(max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """TF-IDF word unigrams + bigrams."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=WORD_NGRAM_RANGE,
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        max_df=0.99,
        token_pattern=r"(?u)\b\w+\b",
    )


def _bow(max_features: int = MAX_FEATURES) -> CountVectorizer:
    """Plain bag-of-words (word 1-2 grams, raw counts)."""
    return CountVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        max_features=max_features,
        token_pattern=r"(?u)\b\w+\b",
    )


def _word_ngrams(max_features: int = MAX_FEATURES) -> TfidfVectorizer:
    """TF-IDF word 1-3 grams."""
    return TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2,
        max_df=0.99,
        token_pattern=r"(?u)\b\w+\b",
    )


def _combined(max_features: int = MAX_FEATURES) -> FeatureUnion:
    """Concatenation of character and word TF-IDF features."""
    char_max = int(max_features * 0.8)
    word_max = int(max_features * 0.2)
    return FeatureUnion(
        transformer_list=[
            ("tfidf_char", _tfidf_char(max_features=char_max)),
            ("tfidf_word", _tfidf_word(max_features=word_max)),
        ]
    )


def get_vectorizer(strategy: str, max_features: int = MAX_FEATURES):
    """Build (but do not fit) a vectorizer for ``strategy``.

    Args:
        strategy: One of :data:`FEATURE_STRATEGIES`.
        max_features: Feature budget for the sparse representations.

    Returns:
        An unfitted scikit-learn transformer.

    Raises:
        ValueError: If ``strategy`` is unknown.
    """
    if strategy == "tfidf_char":
        return _tfidf_char(max_features)
    if strategy == "tfidf_word":
        return _tfidf_word(max_features)
    if strategy == "bow":
        return _bow(max_features)
    if strategy == "word_ngrams":
        return _word_ngrams(max_features)
    if strategy == "combined":
        return _combined(max_features)
    if strategy == "char_freq":
        return CharacterFrequencyVectorizer()
    if strategy == "unicode_blocks":
        return UnicodeBlockVectorizer()
    raise ValueError(f"Unknown feature strategy: {strategy!r}")


def supported_strategies() -> tuple[str, ...]:
    """Return the tuple of registered feature strategy names."""
    return FEATURE_STRATEGIES


# --------------------------------------------------------------------------- #
# Smoke test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from src.utils import setup_logging

    setup_logging()
    corpus = [
        "This is a simple English sentence to test features.",
        "Das ist ein einfacher deutscher Satz für das Testen.",
        "これは日本語のテスト文です。",
        "Esta es una frase sencilla en español.",
    ]
    for strat in ("tfidf_char", "tfidf_word", "bow", "char_freq", "unicode_blocks", "combined"):
        vec = get_vectorizer(strat, max_features=10_000)
        mat = vec.fit_transform(corpus)
        shape = mat.shape
        print(f"{strat:15} -> shape={shape} type={type(mat).__name__}")
