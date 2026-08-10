"""Chart helpers used by the EDA module.

Each function writes a single matplotlib figure to ``filename`` and closes it
so long-running scripts do not accumulate open figures.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import get_logger

logger = get_logger("visualizer")

#: Base styling shared by every figure.
sns.set_theme(style="whitegrid", palette="muted")
_BASE_FIG_SIZE = (11, 6)
_DPI = 120


def _save(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(filename, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved chart -> %s", filename)


def plot_bar_chart(
    series: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    filename: str,
    rotated: bool = False,
) -> None:
    """Horizontal/vertical bar chart of a value-count series."""
    fig, ax = plt.subplots(figsize=_BASE_FIG_SIZE)
    series.plot(kind="barh" if rotated else "bar", ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel(ylabel if rotated else xlabel)
    ax.set_ylabel(xlabel if rotated else ylabel)
    if rotated:
        ax.invert_yaxis()
    _save(fig, filename)


def plot_pie_chart(series: pd.Series, title: str, filename: str) -> None:
    """Pie chart of the class distribution."""
    fig, ax = plt.subplots(figsize=_BASE_FIG_SIZE)
    ax.pie(series.values, labels=series.index, autopct="%1.1f%%", startangle=90)
    ax.set_title(title)
    ax.axis("equal")
    _save(fig, filename)


def plot_histogram(series: pd.Series, title: str, xlabel: str, ylabel: str, filename: str) -> None:
    """Histogram with a KDE overlay."""
    fig, ax = plt.subplots(figsize=_BASE_FIG_SIZE)
    sns.histplot(series, bins=50, kde=True, ax=ax, color="#55A868")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save(fig, filename)


def plot_box_plot(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    title: str,
    ylabel: str,
    filename: str,
) -> None:
    """Box plot of a numeric column grouped by a categorical column."""
    data = df.copy()
    data["_len"] = data[value_col].str.len()
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(x=group_col, y="_len", data=data, ax=ax, fliersize=2)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    _save(fig, filename)


def plot_word_count_analysis(wc: pd.DataFrame, title: str, filename: str) -> None:
    """Bar chart of mean word count per language."""
    fig, ax = plt.subplots(figsize=_BASE_FIG_SIZE)
    wc["mean_words"].sort_values().plot(kind="barh", ax=ax, color="#C44E52")
    ax.set_title(title)
    ax.set_xlabel("Mean number of words")
    ax.set_ylabel("Language")
    _save(fig, filename)


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    filename: str,
    title: str = "Confusion Matrix",
) -> None:
    """Heatmap of a confusion matrix with log-scaled colors."""
    fig, ax = plt.subplots(figsize=(14, 12))
    # Log-scale the counts to keep the heatmap readable.
    cm_log = np.log10(cm + 1)
    sns.heatmap(
        cm_log,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={"label": "Count (log-scaled color)"},
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    _save(fig, filename)


def plot_top5_probs(probs: dict[str, float], filename: str | None = None) -> None:
    """Horizontal bar chart of a language->probability mapping (top-5)."""
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
    labels = [k for k, _ in items]
    values = [v * 100 for _, v in items]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh(labels, values, color="#4C72B0")
    ax.invert_yaxis()
    ax.set_title("Prediction probabilities")
    ax.set_xlabel("Confidence (%)")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2, f"{val:.1f}%",
                va="center", fontsize=9)
    _save(fig, filename) if filename else plt.close(fig)
