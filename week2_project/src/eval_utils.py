"""
eval_utils.py — Model evaluation helpers for the Diabetes Readmission project.

This file contains functions for computing, comparing, and bootstrapping
model performance metrics. I use these in notebooks 08, 09, and 10.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
)


def compute_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    """
    Compute a comprehensive set of binary classification metrics.

    Args:
        y_true (array): true binary labels.
        y_pred (array): predicted binary labels.
        y_prob (array or None): predicted probabilities for class 1.
        model_name (str): label for display.

    Returns:
        dict: all computed metrics.
    """
    metrics = {
        "Model":            model_name,
        "Accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "Balanced Acc.":    round(balanced_accuracy_score(y_true, y_pred), 4),
        "Precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-Score":         round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
        metrics["Avg. Precision"] = round(average_precision_score(y_true, y_prob), 4)
    else:
        metrics["ROC-AUC"] = "N/A"
        metrics["Avg. Precision"] = "N/A"

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics["TP"] = int(tp)
    metrics["TN"] = int(tn)
    metrics["FP"] = int(fp)
    metrics["FN"] = int(fn)
    return metrics


def bootstrap_metric(y_true, y_prob, metric_fn, n_bootstrap=1000, seed=42):
    """
    Bootstrap confidence interval for a probability-based metric.

    I use bootstrap resampling to get a 95% confidence interval around
    a metric like ROC-AUC. This tells me how reliable the point estimate is.

    Method: resample with replacement n_bootstrap times, compute the metric
    each time, and report the 2.5th and 97.5th percentiles.

    Args:
        y_true (array): true binary labels.
        y_prob (array): predicted probabilities.
        metric_fn (callable): function(y_true, y_prob) → float.
        n_bootstrap (int): number of bootstrap samples.
        seed (int): random seed for reproducibility.

    Returns:
        tuple: (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)   # resample with replacement
        y_t = np.array(y_true)[idx]
        y_p = np.array(y_prob)[idx]

        # Skip samples where only one class appears (metric undefined)
        if len(np.unique(y_t)) < 2:
            continue
        boot_scores.append(metric_fn(y_t, y_p))

    point_est = metric_fn(y_true, y_prob)
    ci_lower  = np.percentile(boot_scores, 2.5)
    ci_upper  = np.percentile(boot_scores, 97.5)
    return point_est, ci_lower, ci_upper


def compare_models_table(results_list):
    """
    Build a clean DataFrame comparing multiple models' metrics.

    Args:
        results_list (list of dict): each dict from compute_metrics().

    Returns:
        pd.DataFrame: sorted by ROC-AUC descending.
    """
    df = pd.DataFrame(results_list)
    df = df.set_index("Model")

    # Sort by ROC-AUC if it's numeric
    try:
        df = df.sort_values("ROC-AUC", ascending=False)
    except Exception:
        pass
    return df


def subgroup_metrics(df, y_col, pred_col, prob_col, group_col):
    """
    Compute ROC-AUC, F1, and Recall for each subgroup of a categorical column.

    I use this in the fairness analysis notebook to check whether model
    performance differs across demographic groups (age, race, gender).

    Args:
        df (pd.DataFrame): test set with predictions and probabilities.
        y_col (str): true label column name.
        pred_col (str): predicted label column name.
        prob_col (str): predicted probability column name.
        group_col (str): grouping column name (e.g. "race", "age_group").

    Returns:
        pd.DataFrame: one row per subgroup with key metrics.
    """
    rows = []
    for group_val, sub in df.groupby(group_col):
        if len(sub) < 10:
            continue
        y_t = sub[y_col]
        y_p = sub[pred_col]
        y_pr = sub[prob_col]

        row = {
            group_col: group_val,
            "n_samples": len(sub),
            "prevalence_%": round(y_t.mean() * 100, 1),
            "F1": round(f1_score(y_t, y_p, zero_division=0), 3),
            "Recall": round(recall_score(y_t, y_p, zero_division=0), 3),
            "Precision": round(precision_score(y_t, y_p, zero_division=0), 3),
        }
        try:
            row["ROC-AUC"] = round(roc_auc_score(y_t, y_pr), 3)
        except Exception:
            row["ROC-AUC"] = "N/A"
        rows.append(row)

    return pd.DataFrame(rows).set_index(group_col)
