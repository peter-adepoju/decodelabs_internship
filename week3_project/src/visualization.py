"""
src/visualization.py
────────────────────
Reusable plotting helpers used across notebooks.

Design principles:
  - Every function saves figures to both reports/figures/ and paper_or_report/figures/
    automatically so notebooks do not need to repeat save calls.
  - All plots use colorblind-safe palettes from seaborn.
  - Figures are 150 DPI PNG (configurable).

Usage (in a notebook):
  from src.visualization import save_fig, plot_review_distribution
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Default style for all figures
plt.rcParams.update({
    "figure.dpi":       100,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.size":        11,
    "axes.titlesize":   13,
    "axes.labelsize":   11,
})
PALETTE = "colorblind"


# ─── Save Helper ──────────────────────────────────────────────────────────────

def save_fig(
    fig: plt.Figure,
    filename: str,
    reports_dir: str | Path = "reports/figures",
    paper_dir: str | Path   = "paper_or_report/figures",
    dpi: int = 150,
    fmt: str = "png",
) -> list[Path]:
    """
    Save a matplotlib figure to both reports/ and paper_or_report/.

    Parameters
    ----------
    fig       : plt.Figure
    filename  : str  e.g. 'review_distribution' (no extension)
    reports_dir, paper_dir : str or Path
    dpi       : int
    fmt       : str  'png' or 'pdf'

    Returns
    -------
    list of saved Path objects
    """
    saved = []
    for out_dir in [reports_dir, paper_dir]:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        out_path = p / f"{filename}.{fmt}"
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        saved.append(out_path)
        print(f"  Saved: {out_path}")
    return saved


# ─── Specific Plot Functions ──────────────────────────────────────────────────

def plot_review_distribution(
    df: pd.DataFrame,
    col: str = "review_score",
    save: bool = True,
    **save_kwargs,
) -> plt.Figure:
    """
    Bar chart of review score distribution (1-5 stars).
    """
    counts = df[col].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts.index, counts.values,
           color=sns.color_palette(PALETTE, n_colors=5))
    ax.set_xlabel("Review score")
    ax.set_ylabel("Number of orders")
    ax.set_title("Distribution of review scores")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for i, v in zip(counts.index, counts.values):
        ax.text(i, v + counts.values.max() * 0.01, f"{v:,}", ha="center", fontsize=9)
    plt.tight_layout()
    if save:
        save_fig(fig, "review_distribution", **save_kwargs)
    return fig


def plot_monthly_revenue(
    master: pd.DataFrame,
    save: bool = True,
    **save_kwargs,
) -> plt.Figure:
    """
    Line chart of total monthly revenue.
    """
    df = master.copy()
    df["month"] = df["order_purchase_timestamp"].dt.to_period("M")
    monthly = df.groupby("month")["order_value"].sum().reset_index()
    monthly["month_dt"] = monthly["month"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly["month_dt"], monthly["order_value"] / 1e6,
            marker="o", markersize=4, linewidth=2,
            color=sns.color_palette(PALETTE)[0])
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (millions BRL)")
    ax.set_title("Monthly revenue trend")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:.1f}M"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    if save:
        save_fig(fig, "monthly_revenue", **save_kwargs)
    return fig


def plot_delivery_delay_hist(
    master: pd.DataFrame,
    save: bool = True,
    **save_kwargs,
) -> plt.Figure:
    """
    Histogram of delivery delay in days (positive = late, negative = early).
    """
    delays = master["delivery_delay_days"].dropna()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(delays.clip(-30, 60), bins=60,
            color=sns.color_palette(PALETTE)[2], edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="crimson", linestyle="--", linewidth=1.5, label="On time")
    ax.set_xlabel("Delivery delay (days; positive = late)")
    ax.set_ylabel("Number of orders")
    ax.set_title("Distribution of delivery delay")
    pct_late = (delays > 0).mean() * 100
    ax.set_title(f"Delivery delay distribution  ({pct_late:.1f}% of orders are late)")
    ax.legend()
    plt.tight_layout()
    if save:
        save_fig(fig, "delivery_delay_hist", **save_kwargs)
    return fig


def plot_rfm_scatter(
    rfm: pd.DataFrame,
    save: bool = True,
    **save_kwargs,
) -> plt.Figure:
    """
    Scatter plot of RFM: Recency vs Monetary, coloured by Frequency score.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        rfm["recency"].clip(upper=600),
        np.log1p(rfm["monetary"]),
        c=rfm["F"],
        cmap="viridis_r",
        alpha=0.4, s=8,
    )
    plt.colorbar(scatter, ax=ax, label="Frequency score (1=low, 5=high)")
    ax.set_xlabel("Recency (days since last purchase)")
    ax.set_ylabel("log(1 + Monetary value, BRL)")
    ax.set_title("RFM scatter: recency vs monetary value")
    plt.tight_layout()
    if save:
        save_fig(fig, "rfm_scatter", **save_kwargs)
    return fig


def plot_segment_counts(
    rfm: pd.DataFrame,
    save: bool = True,
    **save_kwargs,
) -> plt.Figure:
    """
    Horizontal bar chart of customer segment sizes.
    """
    counts = rfm["segment"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    pal = sns.color_palette(PALETTE, n_colors=len(counts))
    bars = ax.barh(counts.index, counts.values, color=pal)
    ax.set_xlabel("Number of customers")
    ax.set_title("Customer segment sizes")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    for bar, val in zip(bars, counts.values):
        ax.text(val + counts.values.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=9)
    plt.tight_layout()
    if save:
        save_fig(fig, "segment_counts", **save_kwargs)
    return fig


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    title: str = "Feature importances",
    top_n: int = 15,
    save: bool = True,
    save_name: str = "feature_importance",
    **save_kwargs,
) -> plt.Figure:
    """
    Horizontal bar chart of top-N feature importances.
    """
    df = pd.DataFrame({
        "feature":    feature_names,
        "importance": importances,
    }).nlargest(top_n, "importance").sort_values("importance")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(df["feature"], df["importance"],
            color=sns.color_palette(PALETTE)[0])
    ax.set_xlabel("Importance")
    ax.set_title(title)
    plt.tight_layout()
    if save:
        save_fig(fig, save_name, **save_kwargs)
    return fig


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list,
    title: str = "Confusion matrix",
    save: bool = True,
    save_name: str = "confusion_matrix",
    **save_kwargs,
) -> plt.Figure:
    """
    Seaborn heatmap of a confusion matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    if save:
        save_fig(fig, save_name, **save_kwargs)
    return fig


def plot_revenue_by_state(
    master: pd.DataFrame,
    top_n: int = 15,
    save: bool = True,
    **save_kwargs,
) -> plt.Figure:
    """
    Bar chart of total revenue by Brazilian state.
    """
    rev = (
        master.groupby("customer_state")["order_value"]
        .sum()
        .nlargest(top_n)
        .sort_values()
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(rev.index, rev.values / 1e6,
            color=sns.color_palette(PALETTE)[1])
    ax.set_xlabel("Total revenue (millions BRL)")
    ax.set_title(f"Top {top_n} states by total revenue")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"R${x:.1f}M"))
    plt.tight_layout()
    if save:
        save_fig(fig, "revenue_by_state", **save_kwargs)
    return fig
