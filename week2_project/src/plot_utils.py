"""
plot_utils.py — Reusable plotting helpers for the Diabetes Readmission project.

Each function is documented with what it does, what inputs it expects,
and what output it produces. I import these in notebooks to avoid
copy-pasting the same plotting code.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Style defaults ─────────────────────────────────────────────────────────────
COLORS = {"no_readmit": "#4878CF", "readmit": "#D65F5F",
          "neutral": "#6ACC65", "highlight": "#E8851A"}


def set_plot_style():
    """
    Apply a consistent, clean plot style across all notebooks.
    Call this once at the top of each notebook after importing.
    """
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    })
    sns.set_palette("colorblind")


def save_figure(fig, filename, figures_dir, dpi=150):
    """
    Save a matplotlib figure to the reports/figures/ directory.

    Args:
        fig: matplotlib Figure object.
        filename (str): e.g. "01_class_balance.png"
        figures_dir (str): path to the figures output directory.
        dpi (int): resolution. 150 for screen, 300 for print.

    Returns:
        str: full path where the figure was saved.
    """
    os.makedirs(figures_dir, exist_ok=True)
    path = os.path.join(figures_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Figure saved → {path}")
    return path


def save_table(df, filename, tables_dir):
    """
    Save a DataFrame as CSV to the reports/tables/ directory.

    Args:
        df (pd.DataFrame): table to save.
        filename (str): e.g. "model_comparison.csv"
        tables_dir (str): path to the tables output directory.

    Returns:
        str: full path where the table was saved.
    """
    os.makedirs(tables_dir, exist_ok=True)
    path = os.path.join(tables_dir, filename)
    df.to_csv(path)
    print(f"  Table  saved → {path}")
    return path


def plot_class_balance(y, title="Class Distribution", ax=None):
    """
    Bar chart showing the count and percentage of each class.

    Args:
        y (pd.Series): binary target series (0 / 1).
        title (str): plot title.
        ax: optional matplotlib Axes to draw on.

    Returns:
        matplotlib Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure

    counts = y.value_counts().sort_index()
    total  = len(y)
    labels = ["No Early\nReadmission (0)", "Early\nReadmission (1)"]
    colors = [COLORS["no_readmit"], COLORS["readmit"]]

    bars = ax.bar(labels, counts.values, color=colors, width=0.45, edgecolor="white")
    for bar, cnt in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + total * 0.005,
                f"{cnt:,}\n({cnt/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_title(title, pad=12)
    ax.set_ylabel("Count")
    ax.set_ylim(0, max(counts.values) * 1.18)
    return fig


def plot_confusion_matrix(cm, ax, title="Confusion Matrix"):
    """
    Annotated heatmap of a 2×2 confusion matrix.

    Args:
        cm (np.ndarray): 2×2 confusion matrix from sklearn.
        ax: matplotlib Axes to draw on.
        title (str): plot title.
    """
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred: No (<30)", "Pred: Yes (<30)"],
                yticklabels=["True: No (<30)", "True: Yes (<30)"],
                cbar=False, linewidths=0.5)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)


def plot_roc_curves(models_probs, y_true, ax=None):
    """
    Overlapping ROC curves for multiple models.

    Args:
        models_probs (dict): {model_name: y_prob_array}
        y_true (array): true binary labels.
        ax: optional matplotlib Axes.

    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
    palette = sns.color_palette("colorblind", len(models_probs))

    for (name, y_prob), ls, color in zip(models_probs.items(), line_styles, palette):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})",
                linestyle=ls, color=color, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random (AUC=0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title("ROC Curves — Test Set", pad=12)
    ax.legend(loc="lower right", fontsize=9)
    return fig


def plot_pr_curves(models_probs, y_true, ax=None):
    """
    Precision-Recall curves for multiple models.
    Preferred over ROC when class imbalance is present.

    Args:
        models_probs (dict): {model_name: y_prob_array}
        y_true (array): true binary labels.
        ax: optional matplotlib Axes.

    Returns:
        matplotlib Figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))
    else:
        fig = ax.figure

    baseline = y_true.mean()
    ax.axhline(baseline, color="gray", linestyle="--",
               label=f"Baseline (AP={baseline:.3f})", linewidth=1)

    line_styles = ["-", "--", "-.", ":"]
    palette = sns.color_palette("colorblind", len(models_probs))

    for (name, y_prob), ls, color in zip(models_probs.items(), line_styles, palette):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})",
                linestyle=ls, color=color, linewidth=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Test Set", pad=12)
    ax.legend(loc="upper right", fontsize=9)
    return fig
