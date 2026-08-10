"""Shared utilities: logging, timing, file helpers, IO guards.

Everything in this module is intentionally dependency-light so that other
modules can import it without side effects.
"""

from __future__ import annotations

import functools
import json
import logging
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from src.config import LOG_FILE, LOG_LEVEL

_F = TypeVar("_F", bound=Callable[..., Any])

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
_LOGGER_NAME = "language_identification"


def setup_logging(level: str = LOG_LEVEL, log_file: Path = LOG_FILE) -> logging.Logger:
    """Configure and return the project-wide logger.

    The logger writes to both the console and an optional rotating log file.
    Calling this multiple times is idempotent.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
        log_file: Path to the log file, or ``None`` to disable file output.

    Returns:
        The configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:  # already configured
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    if log_file is not None:
        try:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(fmt)
            logger.addHandler(file_handler)
        except OSError:
            logger.warning("Could not create log file at %s", log_file)

    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a child logger of the project logger.

    Args:
        name: Optional child name, e.g. ``"train"`` -> ``language_identification.train``.
    """
    return logging.getLogger(_LOGGER_NAME if not name else f"{_LOGGER_NAME}.{name}")


# --------------------------------------------------------------------------- #
# Timing
# --------------------------------------------------------------------------- #
def timed(func: _F) -> _F:
    """Decorator that logs how long a function took to run."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        logger = get_logger("timing")
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            logger.info("%s took %.1f ms", func.__name__, elapsed_ms)

    return wrapper  # type: ignore[return-value]


def time_block(label: str) -> Callable[[], float]:
    """Simple context-manager-free timing helper.

    Usage::

        stop = time_block("train")
        # ... work ...
        print(f"{stop():.2f} s")
    """
    start = time.perf_counter()
    return lambda: time.perf_counter() - start


# --------------------------------------------------------------------------- #
# JSON helpers
# --------------------------------------------------------------------------- #
def save_json(data: dict[str, Any], path: Path, indent: int = 2) -> None:
    """Write ``data`` as pretty JSON to ``path`` (UTF-8)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=indent)


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON dict from ``path``."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def safe_filename(name: str) -> str:
    """Sanitize ``name`` so it can be used as a file name."""
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)


def ensure_directory(path: Path) -> Path:
    """Create ``path`` if needed and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
