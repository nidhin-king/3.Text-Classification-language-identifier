"""Exploratory Data Analysis for the language dataset.

Generates a set of summary statistics plus charts (bar / pie / histogram /
box plot / word-count analysis) into ``reports/`` and prints a text summary.
Run directly::

    python -m src.eda
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils import get_logger, save_json, setup_logging
from src.visualizer import (
    plot_bar_chart,
    plot_box_plot,
    plot_histogram,
    plot_pie_chart,
    plot_word_count_analysis,
)

logger = get_logger("eda")


def text_length_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-language DataFrame of text length statistics."""
    data = df.copy()
    data["_len"] = data["text"].str.len()
    stats = data.groupby("language")["_len"].agg(
        count="count",
        mean_len="mean",
        median_len="median",
        min_len="min",
        max_len="max",
    )
    stats["mean_len"] = stats["mean_len"].round(2)
    return stats


def word_count_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-language DataFrame of word-count statistics."""
    df = df.copy()
    df["n_words"] = df["text"].str.split().str.len()
    stats = df.groupby("language")["n_words"].agg(
        mean_words="mean",
        median_words="median",
        max_words="max",
    )
    stats["mean_words"] = stats["mean_words"].round(2)
    return stats


def compute_summary(df: pd.DataFrame) -> dict:
    """Compute all EDA summary statistics and write them to ``reports/``."""
    summary = {
        "shape": {"rows": int(df.shape[0]), "columns": int(df.shape[1])},
        "n_languages": int(df["language"].nunique()),
        "languages": sorted(df["language"].unique().tolist()),
        "class_distribution": df["language"].value_counts().to_dict(),
        "missing_text": int(df["text"].isna().sum()),
        "missing_language": int(df["language"].isna().sum()),
        "duplicate_text": int(df["text"].duplicated().sum()),
        "mean_text_length": float(df["text"].str.len().mean()),
        "median_text_length": float(df["text"].str.len().median()),
        "min_text_length": int(df["text"].str.len().min()),
        "max_text_length": int(df["text"].str.len().max()),
        "mean_word_count": float(df["text"].str.split().str.len().mean()),
        "language_text_length_stats": text_length_stats(df).to_dict(),
        "language_word_count_stats": word_count_stats(df).to_dict(),
    }
    return summary


def generate_charts(df: pd.DataFrame, reports_dir) -> None:
    """Render all EDA charts into ``reports_dir``."""
    from pathlib import Path

    reports_dir = Path(reports_dir)

    # 1. Class distribution (bar chart)
    plot_bar_chart(
        df["language"].value_counts().sort_index(),
        title="Language Frequency (samples per language)",
        xlabel="Language",
        ylabel="Number of samples",
        filename=str(reports_dir / "language_frequency_bar.png"),
        rotated=True,
    )

    # 2. Class distribution (pie chart)
    plot_pie_chart(
        df["language"].value_counts(),
        title="Language Share",
        filename=str(reports_dir / "language_share_pie.png"),
    )

    # 3. Text length histogram
    plot_histogram(
        df["text"].str.len(),
        title="Text Length Distribution (characters)",
        xlabel="Number of characters",
        ylabel="Frequency",
        filename=str(reports_dir / "text_length_histogram.png"),
    )

    # 4. Word count histogram
    plot_histogram(
        df["text"].str.split().str.len(),
        title="Word Count Distribution",
        xlabel="Number of words",
        ylabel="Frequency",
        filename=str(reports_dir / "word_count_histogram.png"),
    )

    # 5. Box plot of text length by language
    plot_box_plot(
        df,
        value_col="text",
        group_col="language",
        title="Text Length by Language (box plot)",
        ylabel="Number of characters",
        filename=str(reports_dir / "text_length_boxplot.png"),
    )

    # 6. Word count analysis (mean words per language bar chart)
    wc = word_count_stats(df)
    plot_word_count_analysis(
        wc,
        title="Mean Word Count per Language",
        filename=str(reports_dir / "word_count_analysis.png"),
    )

    logger.info("Charts written to %s", reports_dir)


def run_eda(df: pd.DataFrame, reports_dir) -> pd.DataFrame:
    """Run the full EDA pipeline.

    Args:
        df: Clean dataset with ``text`` and ``language`` columns.
        reports_dir: Directory where plots and the summary JSON are written.

    Returns:
        The same DataFrame (for chaining).
    """
    from pathlib import Path

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Dataset shape: %s", df.shape)
    summary = compute_summary(df)
    summary_path = reports_dir / "eda_summary.json"
    save_json(summary, summary_path)

    generate_charts(df, reports_dir)
    _print_report(df, summary)
    logger.info("EDA summary saved to %s", summary_path)
    return df


def _print_report(df: pd.DataFrame, summary: dict) -> None:
    """Pretty-print the EDA summary to stdout."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    print(f"Rows          : {summary['shape']['rows']}")
    print(f"Columns       : {summary['shape']['columns']}")
    print(f"Languages     : {summary['n_languages']}")
    print(f"Missing text  : {summary['missing_text']}")
    print(f"Missing lang  : {summary['missing_language']}")
    print(f"Duplicate text: {summary['duplicate_text']}")
    print(f"Mean length   : {summary['mean_text_length']:.1f} chars")
    print(f"Median length : {summary['median_text_length']:.1f} chars")
    print("\nTop 10 languages by frequency:")
    freq = pd.Series(summary["class_distribution"]).sort_values(ascending=False)
    print(freq.head(10).to_string())
    print("\nSample records:")
    print(df[["text", "language"]].head(5).to_string(index=False))


if __name__ == "__main__":
    setup_logging()
    from src.config import REPORTS_DIR
    from src.data_loader import load_dataset

    data = load_dataset()
    run_eda(data, REPORTS_DIR)
