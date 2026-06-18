"""
src/metrics.py
───────────────
Model evaluation helpers: bootstrap confidence intervals and permutation tests.

Why this exists:
  A single accuracy number on the test set is not sufficient — it has no
  uncertainty estimate.  These functions add bootstrap CIs and permutation
  p-values so we can make statistical claims about model performance.

All functions are intentionally written with explicit loops so a beginner
can read and adapt them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from typing import Callable, Optional


# ─── Bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_metric(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    metric_fn: Callable = accuracy_score,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    **metric_kwargs,
) -> dict:
    """
    Estimate a confidence interval for a metric using bootstrapping.

    The bootstrap procedure:
      1. Resample (y_true, y_pred) with replacement n_bootstrap times.
      2. Compute the metric on each resample.
      3. Take the ci-width percentile interval as the CI.

    Parameters
    ----------
    y_true       : array-like   True labels.
    y_pred       : array-like   Predicted labels or scores.
    metric_fn    : callable     Any sklearn metric function.
    n_bootstrap  : int          Number of resamples.
    ci           : float        Confidence level (0.95 = 95%).
    seed         : int          Random seed for reproducibility.
    **metric_kwargs             Passed to metric_fn.

    Returns
    -------
    dict with keys: 'point', 'lower', 'upper', 'ci', 'n_bootstrap'
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    point_estimate = metric_fn(y_true, y_pred, **metric_kwargs)

    boot_scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        score = metric_fn(y_true[idx], y_pred[idx], **metric_kwargs)
        boot_scores.append(score)

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_scores, alpha * 100))
    upper = float(np.percentile(boot_scores, (1 - alpha) * 100))

    return {
        "point":       float(point_estimate),
        "lower":       lower,
        "upper":       upper,
        "ci":          ci,
        "n_bootstrap": n_bootstrap,
    }


# ─── Permutation Test ─────────────────────────────────────────────────────────

def permutation_test(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    metric_fn: Callable = accuracy_score,
    n_permutations: int = 500,
    seed: int = 42,
    **metric_kwargs,
) -> dict:
    """
    Test whether model performance exceeds chance using label permutation.

    The null hypothesis: the model is no better than random label assignment.

    Procedure:
      1. Compute the observed metric on (y_true, y_pred).
      2. Repeat n_permutations times: shuffle y_true randomly, compute metric.
      3. p-value = proportion of permuted scores >= observed score.

    A small p-value (< 0.05) suggests the model performs better than chance.

    Parameters
    ----------
    y_true, y_pred  : array-like
    metric_fn       : callable
    n_permutations  : int
    seed            : int
    **metric_kwargs : passed to metric_fn

    Returns
    -------
    dict with keys: 'observed', 'null_mean', 'null_std', 'p_value', 'n_permutations'
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    observed = metric_fn(y_true, y_pred, **metric_kwargs)

    null_scores = []
    for _ in range(n_permutations):
        y_shuffled = rng.permutation(y_true)
        score = metric_fn(y_shuffled, y_pred, **metric_kwargs)
        null_scores.append(score)

    null_scores = np.array(null_scores)
    p_value = float(np.mean(null_scores >= observed))

    return {
        "observed":       float(observed),
        "null_mean":      float(null_scores.mean()),
        "null_std":       float(null_scores.std()),
        "p_value":        p_value,
        "n_permutations": n_permutations,
    }


# ─── Summary Table ────────────────────────────────────────────────────────────

def model_summary_table(
    results: dict[str, dict],
) -> pd.DataFrame:
    """
    Build a comparison table from a dict of {model_name: bootstrap_result_dict}.

    Parameters
    ----------
    results : dict
        e.g. {'Dummy': bootstrap_metric(...), 'LogReg': bootstrap_metric(...)}

    Returns
    -------
    pd.DataFrame  with columns: Model, Accuracy, Lower_CI, Upper_CI
    """
    rows = []
    for name, r in results.items():
        rows.append({
            "Model":     name,
            "Accuracy":  round(r["point"], 4),
            "Lower_CI":  round(r["lower"], 4),
            "Upper_CI":  round(r["upper"], 4),
            "CI_Width":  round(r["upper"] - r["lower"], 4),
        })
    return pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)


# ─── Macro F1 Helper ──────────────────────────────────────────────────────────

def macro_f1(y_true, y_pred, **kwargs):
    """Wrapper for macro-averaged F1 score, suitable as metric_fn."""
    return f1_score(y_true, y_pred, average="macro", zero_division=0, **kwargs)


def weighted_f1(y_true, y_pred, **kwargs):
    """Wrapper for weighted F1 score."""
    return f1_score(y_true, y_pred, average="weighted", zero_division=0, **kwargs)


# ─── Effect Size ─────────────────────────────────────────────────────────────

def cohen_h(p1: float, p2: float) -> float:
    """
    Cohen's h effect size for two proportions.

    Parameters
    ----------
    p1, p2 : float  Two proportions to compare.

    Returns
    -------
    float  Effect size h.  |h| < 0.2 = small, 0.2-0.5 = medium, > 0.8 = large.
    """
    phi1 = 2 * np.arcsin(np.sqrt(p1))
    phi2 = 2 * np.arcsin(np.sqrt(p2))
    return float(phi1 - phi2)
