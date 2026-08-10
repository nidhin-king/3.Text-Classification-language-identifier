"""Configuration and paths for the Language Identification project.

Central place for every tunable constant: dataset source, feature
representation, model hyper-parameters and output locations. Import the
``config`` module instead of hard-coding paths in other modules so that the
whole project stays consistent and easy to reconfigure.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
#: Project root (two levels up from this file: src/config.py -> project root)
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATASET_DIR: Path = PROJECT_ROOT / "dataset"
MODEL_DIR: Path = PROJECT_ROOT / "models"
NOTEBOOK_DIR: Path = PROJECT_ROOT / "notebooks"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
DATA_DIR: Path = PROJECT_ROOT / "data"
LOG_DIR: Path = PROJECT_ROOT / "logs"

# Create runtime directories eagerly so the rest of the code can rely on them.
for _dir in (DATASET_DIR, MODEL_DIR, REPORTS_DIR, DATA_DIR, LOG_DIR, NOTEBOOK_DIR):
    _dir.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
#: Cache file produced by :mod:`src.data_loader`.
DATASET_CSV: Path = DATASET_DIR / "language_data.csv"

#: Optional fast-path CSV (Kaggle-style "Language Detection" file). If present
#: it is used as the source; otherwise the loader falls back to Tatoeba.
FALLBACK_DATASET_URL: str = (
    "https://raw.githubusercontent.com/Suryaflame5/Language_Detection/master/datasets/train.csv"
)

#: Tatoeba download base for per-language sentence exports.
TATOEBA_BASE_URL: str = "https://downloads.tatoeba.org/exports/per_language/{code}/{code}_sentences.tsv.bz2"

#: Language code -> display name. Codes are ISO 639-3.
LANGS: dict[str, str] = {
    "eng": "English",
    "fra": "French",
    "spa": "Spanish",
    "deu": "German",
    "ita": "Italian",
    "por": "Portuguese",
    "nld": "Dutch",
    "rus": "Russian",
    "pol": "Polish",
    "tur": "Turkish",
    "ara": "Arabic",
    "hin": "Hindi",
    "tam": "Tamil",
    "mal": "Malayalam",
    "jpn": "Japanese",
    "kor": "Korean",
    "swe": "Swedish",
    "dan": "Danish",
    "fin": "Finnish",
    "ces": "Czech",
    "ell": "Greek",
    "heb": "Hebrew",
    "ukr": "Ukrainian",
    "hun": "Hungarian",
    "ron": "Romanian",
    "vie": "Vietnamese",
    "tha": "Thai",
    "ind": "Indonesian",
    "bul": "Bulgarian",
    "cat": "Catalan",
}

#: langdetect (ISO 639-1) code -> our display name, used only by the
#: optional fallback detector for very short/ambiguous inputs.
LANGDETECT_TO_NAME: dict[str, str] = {
    "en": "English", "fr": "French", "es": "Spanish", "de": "German",
    "it": "Italian", "pt": "Portuguese", "nl": "Dutch", "ru": "Russian",
    "pl": "Polish", "tr": "Turkish", "ar": "Arabic", "hi": "Hindi",
    "ta": "Tamil", "ml": "Malayalam", "ja": "Japanese", "ko": "Korean",
    "sv": "Swedish", "da": "Danish", "fi": "Finnish", "cs": "Czech",
    "el": "Greek", "he": "Hebrew", "uk": "Ukrainian", "hu": "Hungarian",
    "ro": "Romanian", "vi": "Vietnamese", "th": "Thai", "id": "Indonesian",
    "bg": "Bulgarian", "ca": "Catalan",
}

#: Display name -> flag emoji used in the web app / CLI.
FLAG_EMOJI: dict[str, str] = {
    "English": "\U0001F1EC\U0001F1E7",
    "French": "\U0001F1EB\U0001F1F7",
    "Spanish": "\U0001F1EA\U0001F1F8",
    "German": "\U0001F1E9\U0001F1EA",
    "Italian": "\U0001F1EE\U0001F1F9",
    "Portuguese": "\U0001F1F5\U0001F1F9",
    "Dutch": "\U0001F1F3\U0001F1F1",
    "Russian": "\U0001F1F7\U0001F1FA",
    "Polish": "\U0001F1F5\U0001F1F1",
    "Turkish": "\U0001F1F9\U0001F1F7",
    "Arabic": "\U0001F1E6\U0001F1FA",
    "Hindi": "\U0001F1EE\U0001F1F3",
    "Tamil": "\U0001F1EE\U0001F1F3",
    "Malayalam": "\U0001F1EE\U0001F1F3",
    "Japanese": "\U0001F1EF\U0001F1F5",
    "Korean": "\U0001F1F0\U0001F1F7",
    "Swedish": "\U0001F1F8\U0001F1EA",
    "Danish": "\U0001F1E9\U0001F1F0",
    "Finnish": "\U0001F1EB\U0001F1EE",
    "Czech": "\U0001F1E8\U0001F1FF",
    "Greek": "\U0001F1EC\U0001F1F7",
    "Hebrew": "\U0001F1EE\U0001F1F1",
    "Ukrainian": "\U0001F1FA\U0001F1E6",
    "Hungarian": "\U0001F1ED\U0001F1FA",
    "Romanian": "\U0001F1F7\U0001F1F4",
    "Vietnamese": "\U0001F1FB\U0001F1F3",
    "Thai": "\U0001F1F9\U0001F1ED",
    "Indonesian": "\U0001F1EE\U0001F1E9",
    "Bulgarian": "\U0001F1E7\U0001F1EC",
    "Catalan": "\U0001F1EA\U0001F1F8",
}

#: A few ready-to-try example texts shown in the web UI.
EXAMPLE_TEXTS: list[str] = [
    "Hello, how are you? I hope you are having a wonderful day.",
    "Bonjour tout le monde, comment allez-vous aujourd'hui ?",
    "नमस्ते आप कैसे हैं, मैं आपके बारे में सुनकर खुश हूं।",
    "வணக்கம், இன்று நீங்கள் எப்படி இருக்கிறீர்கள்?",
    "ഹലോ, നിങ്ങൾക്ക് സുഖമാണോ? ഇന്ന് എന്തുണ്ട് വിശേഷം?",
    "Guten Morgen, wie geht es dir heute?",
    "Hola, ¿cómo estás hoy? Espero que tengas un buen día.",
    "Buongiorno, come stai oggi?",
    "こんにちは、今日はお元気ですか。",
    "안녕하세요, 오늘 기분이 어떠세요?",
    "Привет, как у тебя дела сегодня?",
    "مرحبا، كيف حالك اليوم؟",
    "สวัสดีครับ วันนี้คุณเป็นอย่างไรบ้าง",
    "Xin chào, hôm nay bạn khỏe không?",
]

#: Number of sentences to sample per language when building from Tatoeba.
SAMPLES_PER_LANGUAGE: int = 2000

#: Seed for reproducible sampling / splits.
RANDOM_STATE: int = 42

# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
PREPROCESS_CONFIG: dict[str, bool] = {
    "lowercase": True,          # merge 'A'/'a' (no-op for caseless scripts)
    "strip": True,              # trim leading/trailing whitespace
    "collapse_whitespace": True,  # multiple spaces -> single space
    "remove_urls": True,
    "remove_emails": True,
    "remove_emoji": True,       # emojis carry no language signal
    "remove_numbers": False,    # keep numbers; they rarely hurt char n-grams
    "normalize_unicode": True,  # NFC normalization
    "remove_punctuation": False,  # keep punctuation (useful for char n-grams)
}

# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
#: Feature strategies available for comparison in train.py
FEATURE_STRATEGIES: tuple[str, ...] = (
    "tfidf_char",        # TF-IDF character n-grams  (usually the winner)
    "tfidf_word",        # TF-IDF word n-grams
    "bow",               # Bag of words (word 1-2 grams, raw counts)
    "word_ngrams",       # TF-IDF word 1-3 grams
    "char_freq",         # character frequency histogram (dense)
    "unicode_blocks",    # counts of characters per Unicode block (dense)
    "combined",          # tfidf_char + tfidf_word (FeatureUnion)
)

#: Max features for the sparse TF-IDF / BOW vectorizers.
MAX_FEATURES: int = 200_000
#: Character n-gram range for TF-IDF char vectorizer.
CHAR_NGRAM_RANGE: tuple[int, int] = (2, 5)
#: Word n-gram range for TF-IDF word vectorizer.
WORD_NGRAM_RANGE: tuple[int, int] = (1, 2)

# --------------------------------------------------------------------------- #
# Model selection
# --------------------------------------------------------------------------- #
#: Test set ratio for the final evaluation.
TEST_SIZE: float = 0.20
#: Cross-validation folds used during model comparison.
CV_FOLDS: int = 5
#: Number of samples used during the (fast) model comparison stage.
COMPARISON_SUBSAMPLE: int = 10_000

# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
MODEL_FILE: Path = MODEL_DIR / "best_model.joblib"
VECTORIZER_FILE: Path = MODEL_DIR / "vectorizer.joblib"
LABEL_ENCODER_FILE: Path = MODEL_DIR / "label_encoder.joblib"
METADATA_FILE: Path = MODEL_DIR / "model_meta.json"

#: Optional fallback detector used for very short/ambiguous inputs.
ENABLE_FALLBACK_DETECTOR: bool = True
#: Below this confidence the pipeline consults the fallback detector.
FALLBACK_THRESHOLD: float = 0.35

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_FILE: Path = LOG_DIR / "app.log"
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
