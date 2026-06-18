"""
tests/test_metrics.py
──────────────────────
Tests for src/metrics.py

Uses only numpy arrays — no Olist data needed.
"""

import numpy as np
import pytest

from src.metrics import (
    bootstrap_metric,
    permutation_test,
    model_summary_table,
    macro_f1,
    cohen_h,
)
from sklearn.metrics import accuracy_score


# ─── bootstrap_metric ─────────────────────────────────────────────────────────

def test_bootstrap_returns_expected_keys():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0])
    result = bootstrap_metric(y_true, y_pred, n_bootstrap=100)
    assert {"point", "lower", "upper", "ci", "n_bootstrap"}.issubset(result)

def test_bootstrap_lower_le_point_le_upper():
    y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 1, 1, 0, 0, 0, 1, 1])
    r = bootstrap_metric(y_true, y_pred, n_bootstrap=200)
    assert r["lower"] <= r["point"] <= r["upper"]

def test_bootstrap_perfect_classifier():
    y = np.array([0, 1, 1, 0, 1])
    r = bootstrap_metric(y, y, n_bootstrap=100)
    assert r["point"] == pytest.approx(1.0)

def test_bootstrap_ci_level():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0] * 5)
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0] * 5)
    r = bootstrap_metric(y_true, y_pred, ci=0.90, n_bootstrap=200)
    assert r["ci"] == 0.90

def test_bootstrap_reproducible():
    y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0] * 10)
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 0] * 10)
    r1 = bootstrap_metric(y_true, y_pred, seed=42, n_bootstrap=100)
    r2 = bootstrap_metric(y_true, y_pred, seed=42, n_bootstrap=100)
    assert r1["lower"] == r2["lower"]
    assert r1["upper"] == r2["upper"]


# ─── permutation_test ─────────────────────────────────────────────────────────

def test_permutation_returns_expected_keys():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    result = permutation_test(y_true, y_pred, n_permutations=50)
    assert {"observed", "null_mean", "null_std", "p_value", "n_permutations"}.issubset(result)

def test_permutation_p_value_in_range():
    y_true = np.array([0, 1, 1, 0, 1, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 1, 0, 0, 0])
    r = permutation_test(y_true, y_pred, n_permutations=50)
    assert 0.0 <= r["p_value"] <= 1.0

def test_permutation_perfect_classifier_low_pvalue():
    """A perfect classifier should have p-value very close to 0."""
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=50)
    r = permutation_test(y, y, n_permutations=200, seed=42)
    # p-value should be 0 or very small
    assert r["p_value"] < 0.05


# ─── model_summary_table ──────────────────────────────────────────────────────

def test_summary_table_shape():
    results = {
        "Dummy":  {"point": 0.50, "lower": 0.45, "upper": 0.55},
        "LogReg": {"point": 0.65, "lower": 0.60, "upper": 0.70},
    }
    df = model_summary_table(results)
    assert df.shape[0] == 2
    assert "Model" in df.columns

def test_summary_table_sorted_by_accuracy():
    results = {
        "Dummy":  {"point": 0.50, "lower": 0.45, "upper": 0.55},
        "RF":     {"point": 0.72, "lower": 0.68, "upper": 0.76},
        "LogReg": {"point": 0.65, "lower": 0.60, "upper": 0.70},
    }
    df = model_summary_table(results)
    assert df.iloc[0]["Model"] == "RF"


# ─── cohen_h ──────────────────────────────────────────────────────────────────

def test_cohen_h_identical_proportions():
    assert cohen_h(0.5, 0.5) == pytest.approx(0.0, abs=1e-9)

def test_cohen_h_nonzero_for_different_proportions():
    h = cohen_h(0.7, 0.3)
    assert abs(h) > 0.5  # medium-to-large effect


# ─── macro_f1 helper ──────────────────────────────────────────────────────────

def test_macro_f1_perfect():
    y = np.array([0, 1, 2, 0, 1, 2])
    assert macro_f1(y, y) == pytest.approx(1.0)

def test_macro_f1_between_0_and_1():
    y_true = np.array([0, 1, 1, 0, 2, 2])
    y_pred = np.array([0, 0, 1, 1, 2, 1])
    score = macro_f1(y_true, y_pred)
    assert 0.0 <= score <= 1.0
