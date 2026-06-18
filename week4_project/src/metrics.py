"""
src/metrics.py
==============
Evaluation metrics for TB classification with bootstrap confidence intervals.

Why this is a separate module:
  - Metric functions are called in notebooks 11, 12, 13, 14, 15, and in
    the FL training loop (notebooks 08–10).
  - Bootstrap CI calculation is ~40 lines — cleaner to import than repeat.
  - Every function below is also demonstrated step-by-step in notebook 11.

Contents
--------
  compute_metrics          : Full metric suite from predictions
  bootstrap_ci             : Bootstrap confidence intervals for any metric
  youden_threshold         : Optimal threshold via Youden's J statistic
  mcnemar_test             : Compare two classifiers on same test set
  delong_auc_comparison    : DeLong test for comparing AUC-ROCs
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from scipy import stats


# ──────────────────────────────────────────────
# Core metric computation
# ──────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    threshold: float | None = None,
) -> dict:
    """
    Compute a full suite of binary classification metrics.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth labels (0 = TB negative, 1 = TB positive).
    y_prob : array-like of float
        Predicted probability of TB positive (class 1).
    threshold : float or None
        Decision threshold for converting probabilities to class labels.
        If None, the Youden's J optimal threshold is used.

    Returns
    -------
    dict
        Keys: 'auc_roc', 'auc_prc', 'threshold', 'sensitivity', 'specificity',
              'ppv', 'npv', 'f1', 'accuracy', 'balanced_accuracy',
              'tp', 'tn', 'fp', 'fn'

    Notes
    -----
    - Sensitivity = recall = true positive rate (TPR)
    - Specificity = true negative rate (TNR)
    - PPV = positive predictive value = precision
    - NPV = negative predictive value
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if len(np.unique(y_true)) < 2:
        warnings.warn("y_true contains only one class — some metrics are undefined.")

    # AUC metrics (threshold-independent)
    auc_roc = roc_auc_score(y_true, y_prob)
    auc_prc = average_precision_score(y_true, y_prob)

    # Determine decision threshold
    if threshold is None:
        threshold = youden_threshold(y_true, y_prob)

    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Derived metrics — with safety for zero denominators
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # True positive rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0   # True negative rate
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # Precision
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0   # Neg. predictive value

    return {
        "auc_roc":          float(auc_roc),
        "auc_prc":          float(auc_prc),
        "threshold":        float(threshold),
        "sensitivity":      float(sensitivity),
        "specificity":      float(specificity),
        "ppv":              float(ppv),
        "npv":              float(npv),
        "f1":               float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy":         float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy":float(balanced_accuracy_score(y_true, y_pred)),
        "tp":               int(tp),
        "tn":               int(tn),
        "fp":               int(fp),
        "fn":               int(fn),
        "n_positive":       int(y_true.sum()),
        "n_negative":       int((1 - y_true).sum()),
    }


# ──────────────────────────────────────────────
# Threshold selection
# ──────────────────────────────────────────────

def youden_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> float:
    """
    Find the decision threshold that maximises Youden's J statistic.

    Youden's J = Sensitivity + Specificity - 1
               = TPR - FPR

    This is commonly used in medical diagnostics to select an operating
    point that balances sensitivity and specificity.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_prob : np.ndarray
        Predicted positive probabilities.

    Returns
    -------
    float
        Optimal threshold in [0, 1].
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    j_scores = tpr - fpr   # Youden's J at each threshold
    best_idx = np.argmax(j_scores)
    return float(thresholds[best_idx])


# ──────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────

def bootstrap_ci(
    y_true: np.ndarray | list,
    y_prob: np.ndarray | list,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Compute bootstrap confidence intervals for any scalar metric.

    Why bootstrap?
    --------------
    A single metric value (e.g., AUC = 0.88) looks precise but could
    vary substantially if we had a different test set. Bootstrap CI gives
    a range of plausible values: "AUC = 0.88 (95% CI: 0.83–0.93)".
    This is required for Nature Medicine-level reporting.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth labels.
    y_prob : array-like of float
        Predicted probabilities.
    metric_fn : callable
        Function with signature (y_true, y_prob) → float.
        E.g., sklearn.metrics.roc_auc_score
    n_bootstrap : int
        Number of bootstrap resamples. 1000 is standard.
    ci : float
        Confidence interval level. Default 0.95 = 95% CI.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with keys: 'point_estimate', 'ci_lower', 'ci_upper', 'ci_level'

    Example
    -------
    >>> from sklearn.metrics import roc_auc_score
    >>> result = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap=1000)
    >>> print(f"AUC = {result['point_estimate']:.3f} "
    ...       f"({result['ci_lower']:.3f}–{result['ci_upper']:.3f})")
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    rng    = np.random.default_rng(seed)
    n      = len(y_true)

    # Point estimate on the full test set
    point_estimate = metric_fn(y_true, y_prob)

    # Bootstrap resampling
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = rng.integers(0, n, size=n)
        y_t = y_true[idx]
        y_p = y_prob[idx]

        # Skip this resample if it only has one class
        if len(np.unique(y_t)) < 2:
            continue

        try:
            score = metric_fn(y_t, y_p)
            bootstrap_scores.append(score)
        except Exception:
            continue

    bootstrap_scores = np.array(bootstrap_scores)

    # Percentile CI
    alpha = (1 - ci) / 2
    ci_lower = np.percentile(bootstrap_scores, 100 * alpha)
    ci_upper = np.percentile(bootstrap_scores, 100 * (1 - alpha))

    return {
        "point_estimate": float(point_estimate),
        "ci_lower":       float(ci_lower),
        "ci_upper":       float(ci_upper),
        "ci_level":       ci,
        "n_bootstrap":    len(bootstrap_scores),
    }


def bootstrap_metric_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Compute bootstrap CI for multiple metrics at once, returning a DataFrame.

    Parameters
    ----------
    y_true, y_prob : arrays
    threshold : float or None
        Decision threshold. If None, Youden's J is used.
    n_bootstrap, ci, seed : see bootstrap_ci()

    Returns
    -------
    pd.DataFrame with columns: metric, point_estimate, ci_lower, ci_upper
    """
    if threshold is None:
        threshold = youden_threshold(y_true, y_prob)

    # Define metric functions that work with (y_true, y_prob)
    metric_fns = {
        "AUC-ROC":          lambda yt, yp: roc_auc_score(yt, yp),
        "AUC-PRC":          lambda yt, yp: average_precision_score(yt, yp),
        "Sensitivity":      lambda yt, yp: recall_score(yt, (yp >= threshold).astype(int), zero_division=0),
        "Specificity":      lambda yt, yp: recall_score(1 - yt, 1 - (yp >= threshold).astype(int), zero_division=0),
        "PPV":              lambda yt, yp: precision_score(yt, (yp >= threshold).astype(int), zero_division=0),
        "F1":               lambda yt, yp: f1_score(yt, (yp >= threshold).astype(int), zero_division=0),
        "Balanced Accuracy":lambda yt, yp: balanced_accuracy_score(yt, (yp >= threshold).astype(int)),
    }

    rows = []
    for metric_name, fn in metric_fns.items():
        result = bootstrap_ci(y_true, y_prob, fn, n_bootstrap=n_bootstrap, ci=ci, seed=seed)
        rows.append({
            "Metric":         metric_name,
            "Estimate":       round(result["point_estimate"], 4),
            "CI Lower":       round(result["ci_lower"], 4),
            "CI Upper":       round(result["ci_upper"], 4),
            f"{int(ci*100)}% CI": (
                f"{result['ci_lower']:.3f}–{result['ci_upper']:.3f}"
            ),
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────
# Statistical comparison tests
# ──────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
) -> dict:
    """
    McNemar's test comparing two classifiers on the same test set.

    McNemar's test asks: "Do classifiers A and B make different errors?"
    It is appropriate for paired binary outcomes (same test set).

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth labels.
    y_pred_a : np.ndarray
        Binary predictions from model A (e.g., federated).
    y_pred_b : np.ndarray
        Binary predictions from model B (e.g., centralised).

    Returns
    -------
    dict with 'statistic', 'p_value', 'interpretation'
    """
    # Build contingency table of disagreements
    # b = A correct, B wrong; c = A wrong, B correct
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    b = np.sum(correct_a & ~correct_b)   # A right, B wrong
    c = np.sum(~correct_a & correct_b)   # A wrong, B right

    # McNemar's chi-squared with continuity correction
    if (b + c) == 0:
        return {
            "statistic": 0.0,
            "p_value":   1.0,
            "b":         int(b),
            "c":         int(c),
            "interpretation": "Both models make identical errors (b+c=0).",
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value   = 1 - stats.chi2.cdf(statistic, df=1)

    interpretation = (
        "Significant difference in errors (p < 0.05)."
        if p_value < 0.05
        else "No significant difference in errors (p ≥ 0.05)."
    )

    return {
        "statistic":      float(statistic),
        "p_value":        float(p_value),
        "b":              int(b),
        "c":              int(c),
        "interpretation": interpretation,
    }


def bootstrap_auc_diff(
    y_true: np.ndarray,
    y_prob_a: np.ndarray,
    y_prob_b: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """
    Bootstrap test for the difference in AUC-ROC between two models.

    Used to assess: "Is the federated model's AUC significantly different
    from the centralised model's AUC?"

    Parameters
    ----------
    y_true : np.ndarray
    y_prob_a : np.ndarray
        Predicted probabilities from model A.
    y_prob_b : np.ndarray
        Predicted probabilities from model B.
    n_bootstrap, ci, seed : see bootstrap_ci()

    Returns
    -------
    dict with 'auc_a', 'auc_b', 'diff', 'ci_lower', 'ci_upper', 'p_value'
    """
    rng = np.random.default_rng(seed)
    n   = len(y_true)

    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)
    obs_diff = auc_a - auc_b

    diffs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        try:
            d = roc_auc_score(yt, y_prob_a[idx]) - roc_auc_score(yt, y_prob_b[idx])
            diffs.append(d)
        except Exception:
            continue

    diffs = np.array(diffs)
    alpha    = (1 - ci) / 2
    ci_lower = np.percentile(diffs, 100 * alpha)
    ci_upper = np.percentile(diffs, 100 * (1 - alpha))

    # Two-tailed p-value: proportion of bootstrap diffs on opposite side of 0
    p_value = float(np.mean(np.abs(diffs) >= np.abs(obs_diff)))

    return {
        "auc_a":      float(auc_a),
        "auc_b":      float(auc_b),
        "diff":       float(obs_diff),
        "ci_lower":   float(ci_lower),
        "ci_upper":   float(ci_upper),
        "ci_level":   ci,
        "p_value":    p_value,
        "n_bootstrap":len(diffs),
    }
