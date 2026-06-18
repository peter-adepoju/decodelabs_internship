"""
src/visualization.py
====================
Publication-quality figure helpers for FedTB-Nigeria.

Why this is a separate module:
  - The same plotting style (colorblind-safe palette, 300 DPI, dual save to
    reports/ and paper_or_report/) is needed in notebooks 04, 11–15.
  - These are genuine repeats that would clutter notebooks without adding
    learning value — the important learning is interpreting the plots.

All plots use:
  - Seaborn colorblind-safe palette ("colorblind")
  - Matplotlib rcParams tuned for publication quality
  - Automatic dual save: reports/figures/ AND paper_or_report/figures/
  - Both PNG (300 DPI) and PDF (vector) formats
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix


# ──────────────────────────────────────────────
# Global style
# ──────────────────────────────────────────────

def set_publication_style() -> None:
    """
    Apply a clean, publication-ready matplotlib style.

    Call once at the top of any notebook before plotting.
    """
    plt.rcParams.update({
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "font.family":       "sans-serif",
        "font.size":         12,
        "axes.titlesize":    13,
        "axes.labelsize":    12,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
        "legend.fontsize":   10,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.figsize":    (7, 5),
    })
    sns.set_palette("colorblind")


def save_figure(
    fig: plt.Figure,
    name: str,
    figures_dir: str | Path = "reports/figures",
    paper_figures_dir: str | Path = "paper_or_report/figures",
    dpi: int = 300,
) -> None:
    """
    Save a figure to both reports/figures/ and paper_or_report/figures/.

    Saves PNG (raster) and PDF (vector) formats.

    Parameters
    ----------
    fig : plt.Figure
    name : str
        Filename without extension (e.g., 'roc_curve_comparison').
    figures_dir : str or Path
    paper_figures_dir : str or Path
    dpi : int
    """
    for directory in [figures_dir, paper_figures_dir]:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{name}.png", dpi=dpi, bbox_inches="tight")
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")

    print(f"Figure saved: '{name}.png' and '{name}.pdf'")


# ──────────────────────────────────────────────
# Diagnostic curves
# ──────────────────────────────────────────────

def plot_roc_curves(
    models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "ROC Curves",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot ROC curves for one or more models on the same axes.

    Parameters
    ----------
    models_dict : dict
        Keys: model name (str).
        Values: (y_true, y_prob) tuple of numpy arrays.
    title : str
    ax : plt.Axes or None

    Returns
    -------
    plt.Figure

    Example
    -------
    >>> fig = plot_roc_curves({
    ...     "Centralised": (y_true, probs_central),
    ...     "Federated (ε=8)": (y_true, probs_federated),
    ... })
    """
    from sklearn.metrics import roc_auc_score

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    palette = sns.color_palette("colorblind", n_colors=len(models_dict))

    for (name, (y_true, y_prob)), color in zip(models_dict.items(), palette):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=color, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random chance (AUC = 0.50)")
    ax.set_xlabel("False Positive Rate (1 − Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    return fig


def plot_prc_curves(
    models_dict: dict[str, tuple[np.ndarray, np.ndarray]],
    title: str = "Precision–Recall Curves",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot precision–recall curves for one or more models.

    Parameters
    ----------
    models_dict : dict
        Keys: model name. Values: (y_true, y_prob).

    Returns
    -------
    plt.Figure
    """
    from sklearn.metrics import average_precision_score

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.get_figure()

    palette = sns.color_palette("colorblind", n_colors=len(models_dict))

    for (name, (y_true, y_prob)), color in zip(models_dict.items(), palette):
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.3f})", color=color, lw=2)

    # Baseline = prevalence
    prevalence = None
    for _, (y_true, _) in models_dict.items():
        prevalence = y_true.mean()
        break
    if prevalence is not None:
        ax.axhline(prevalence, color="k", linestyle="--", lw=1,
                   label=f"Prevalence baseline ({prevalence:.2f})")

    ax.set_xlabel("Recall (Sensitivity)")
    ax.set_ylabel("Precision (PPV)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.3)

    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
    title: str = "Confusion Matrix",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot a normalised confusion matrix with raw counts.

    Parameters
    ----------
    y_true : np.ndarray
    y_pred : np.ndarray
    class_names : list of str or None
    title : str
    ax : plt.Axes or None

    Returns
    -------
    plt.Figure
    """
    if class_names is None:
        class_names = ["TB Negative", "TB Positive"]

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    sns.heatmap(
        cm_norm, annot=False, fmt=".0%", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax, cbar=True, vmin=0, vmax=1,
    )

    # Overlay raw counts
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]}\n({cm_norm[i, j]:.1%})",
                ha="center", va="center", fontsize=10,
                color="white" if cm_norm[i, j] > 0.6 else "black",
            )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)

    return fig


# ──────────────────────────────────────────────
# FL training curves
# ──────────────────────────────────────────────

def plot_fl_training_curves(
    history: dict[str, list[float]],
    title: str = "Federated Learning Training History",
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """
    Plot FL global model metrics across communication rounds.

    Parameters
    ----------
    history : dict
        Keys: metric name. Values: list of per-round values.
        E.g., {'loss': [0.9, 0.7, ...], 'auc_roc': [0.6, 0.75, ...]}
    title : str
    ax : plt.Axes or None

    Returns
    -------
    plt.Figure
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    palette = sns.color_palette("colorblind", n_colors=len(history))
    rounds  = None

    for (metric_name, values), color in zip(history.items(), palette):
        if rounds is None:
            rounds = list(range(1, len(values) + 1))
        ax.plot(rounds, values, label=metric_name, color=color, lw=2, marker="o", markersize=3)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    return fig


def plot_site_class_distribution(
    site_labels: dict[str, list[int]],
    title: str = "Class Distribution Across Simulated Hospital Sites",
) -> plt.Figure:
    """
    Bar chart showing TB positive/negative proportions per site.

    Parameters
    ----------
    site_labels : dict
        Keys: site name. Values: list of integer labels.

    Returns
    -------
    plt.Figure
    """
    rows = []
    for site_name, labels in site_labels.items():
        labels_arr = np.array(labels)
        n_pos = int(labels_arr.sum())
        n_neg = int((1 - labels_arr).sum())
        rows.append({"Site": site_name, "TB Positive": n_pos, "TB Negative": n_neg})

    df = pd.DataFrame(rows)
    df = df.set_index("Site")

    fig, ax = plt.subplots(figsize=(9, 4))
    df.plot(kind="bar", ax=ax, color=sns.color_palette("colorblind", 2), edgecolor="white")
    ax.set_xlabel("Hospital Site")
    ax.set_ylabel("Number of Images")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Label")
    ax.grid(axis="y", alpha=0.3)

    return fig


# ──────────────────────────────────────────────
# Privacy–utility trade-off
# ──────────────────────────────────────────────

def plot_privacy_utility_tradeoff(
    epsilon_values: list[float],
    auc_values: list[float],
    auc_ci_lower: list[float],
    auc_ci_upper: list[float],
    centralised_auc: float | None = None,
    title: str = "Privacy–Utility Trade-off (AUC vs ε)",
) -> plt.Figure:
    """
    Plot AUC-ROC versus differential privacy budget ε.

    Parameters
    ----------
    epsilon_values : list of float
        ε values on x-axis. Can include np.inf for 'No DP'.
    auc_values : list of float
        Corresponding AUC-ROC point estimates.
    auc_ci_lower / auc_ci_upper : list of float
        95% bootstrap CI bounds.
    centralised_auc : float or None
        Centralised baseline AUC to draw as horizontal reference.

    Returns
    -------
    plt.Figure
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    x_pos = list(range(len(epsilon_values)))
    x_labels = [f"ε={e:.1f}" if np.isfinite(e) else "No DP" for e in epsilon_values]

    ax.errorbar(
        x_pos, auc_values,
        yerr=[
            np.array(auc_values) - np.array(auc_ci_lower),
            np.array(auc_ci_upper) - np.array(auc_values),
        ],
        fmt="o-", color=sns.color_palette("colorblind")[0],
        capsize=5, lw=2, markersize=7, label="Federated + DP",
    )

    if centralised_auc is not None:
        ax.axhline(
            centralised_auc, color=sns.color_palette("colorblind")[1],
            linestyle="--", lw=2, label=f"Centralised baseline (AUC = {centralised_auc:.3f})",
        )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Differential Privacy Budget (ε)  [lower = more private]")
    ax.set_ylabel("AUC-ROC (95% CI)")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    return fig
