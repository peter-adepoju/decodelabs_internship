"""
tests/test_metrics.py
=====================
Unit tests for src/metrics.py — AUC, sensitivity, bootstrap CI,
McNemar's test, and bootstrap AUC difference.

Run with:
  pytest tests/test_metrics.py -v
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.metrics import (
    compute_metrics,
    bootstrap_ci,
    bootstrap_metric_table,
    youden_threshold,
    mcnemar_test,
    bootstrap_auc_diff,
)
from sklearn.metrics import roc_auc_score


# ── Fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.1, 0.1, 0.9, 0.9, 0.9])
    return y_true, y_prob

@pytest.fixture
def random_predictions():
    rng    = np.random.default_rng(42)
    y_true = rng.choice([0, 1], size=80)
    y_prob = rng.uniform(0, 1, size=80)
    return y_true, y_prob

@pytest.fixture
def imbalanced_predictions():
    """Highly imbalanced: 90% negative."""
    rng    = np.random.default_rng(0)
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = rng.uniform(0, 1, size=100)
    return y_true, y_prob


# ── compute_metrics ──────────────────────────────────────────────────

class TestComputeMetrics:

    def test_returns_dict(self, random_predictions):
        y_true, y_prob = random_predictions
        result = compute_metrics(y_true, y_prob)
        assert isinstance(result, dict)

    def test_required_keys_present(self, random_predictions):
        y_true, y_prob = random_predictions
        result = compute_metrics(y_true, y_prob)
        for key in ["auc_roc", "auc_prc", "sensitivity", "specificity",
                    "ppv", "npv", "f1", "accuracy", "balanced_accuracy",
                    "tp", "tn", "fp", "fn", "threshold"]:
            assert key in result, f"Missing key: {key}"

    def test_perfect_predictions_give_auc_one(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        result = compute_metrics(y_true, y_prob)
        assert abs(result["auc_roc"] - 1.0) < 1e-6

    def test_auc_between_zero_and_one(self, random_predictions):
        y_true, y_prob = random_predictions
        result = compute_metrics(y_true, y_prob)
        assert 0.0 <= result["auc_roc"] <= 1.0

    def test_sensitivity_between_zero_and_one(self, random_predictions):
        y_true, y_prob = random_predictions
        result = compute_metrics(y_true, y_prob)
        assert 0.0 <= result["sensitivity"] <= 1.0

    def test_tp_plus_fn_equals_total_positives(self, random_predictions):
        y_true, y_prob = random_predictions
        result = compute_metrics(y_true, y_prob)
        total_pos = int(y_true.sum())
        assert result["tp"] + result["fn"] == total_pos

    def test_custom_threshold_applied(self, random_predictions):
        y_true, y_prob = random_predictions
        # With threshold=0.0, everything is predicted positive
        result = compute_metrics(y_true, y_prob, threshold=0.0)
        assert result["fn"] == 0   # No false negatives
        assert result["threshold"] == 0.0


# ── youden_threshold ─────────────────────────────────────────────────

class TestYoudenThreshold:

    def test_returns_float_in_unit_interval(self, random_predictions):
        y_true, y_prob = random_predictions
        thr = youden_threshold(y_true, y_prob)
        assert 0.0 <= thr <= 1.0

    def test_perfect_threshold_near_midpoint(self, perfect_predictions):
        y_true, y_prob = perfect_predictions
        thr = youden_threshold(y_true, y_prob)
        # Perfect predictions: threshold should be between 0.1 and 0.9
        assert 0.05 < thr < 0.95


# ── bootstrap_ci ─────────────────────────────────────────────────────

class TestBootstrapCI:

    def test_returns_dict_with_required_keys(self, random_predictions):
        y_true, y_prob = random_predictions
        result = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap=50)
        for key in ["point_estimate", "ci_lower", "ci_upper", "ci_level"]:
            assert key in result

    def test_ci_lower_leq_point_leq_ci_upper(self, random_predictions):
        y_true, y_prob = random_predictions
        result = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap=100)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_perfect_predictions_tight_ci(self, perfect_predictions):
        """Perfect predictions should give CI near [1.0, 1.0]."""
        y_true, y_prob = perfect_predictions
        result = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap=100)
        assert result["ci_lower"] > 0.80   # Should be near 1.0


# ── bootstrap_metric_table ───────────────────────────────────────────

class TestBootstrapMetricTable:

    def test_returns_dataframe(self, random_predictions):
        import pandas as pd
        y_true, y_prob = random_predictions
        df = bootstrap_metric_table(y_true, y_prob, n_bootstrap=50)
        assert hasattr(df, "columns")

    def test_has_metric_column(self, random_predictions):
        y_true, y_prob = random_predictions
        df = bootstrap_metric_table(y_true, y_prob, n_bootstrap=50)
        assert "Metric" in df.columns


# ── mcnemar_test ─────────────────────────────────────────────────────

class TestMcNemarTest:

    def test_identical_predictions_pvalue_one(self):
        y_true  = np.array([0, 1, 0, 1, 0, 1])
        y_pred  = np.array([0, 1, 0, 1, 0, 1])
        result  = mcnemar_test(y_true, y_pred, y_pred)
        assert result["p_value"] == 1.0

    def test_returns_required_keys(self):
        y_true  = np.array([0, 1, 0, 1, 0, 1])
        y_pred_a = np.array([0, 1, 0, 0, 0, 1])
        y_pred_b = np.array([0, 1, 1, 1, 0, 0])
        result   = mcnemar_test(y_true, y_pred_a, y_pred_b)
        for key in ["statistic", "p_value", "b", "c", "interpretation"]:
            assert key in result

    def test_pvalue_in_unit_interval(self):
        rng = np.random.default_rng(0)
        y_true   = rng.choice([0, 1], 50)
        y_pred_a = rng.choice([0, 1], 50)
        y_pred_b = rng.choice([0, 1], 50)
        result   = mcnemar_test(y_true, y_pred_a, y_pred_b)
        assert 0.0 <= result["p_value"] <= 1.0


# ── bootstrap_auc_diff ───────────────────────────────────────────────

class TestBootstrapAUCDiff:

    def test_returns_required_keys(self, random_predictions):
        y_true, y_prob_a = random_predictions
        rng = np.random.default_rng(99)
        y_prob_b = rng.uniform(0, 1, len(y_true))
        result = bootstrap_auc_diff(y_true, y_prob_a, y_prob_b, n_bootstrap=50)
        for key in ["auc_a", "auc_b", "diff", "ci_lower", "ci_upper", "p_value"]:
            assert key in result

    def test_same_predictions_diff_near_zero(self, random_predictions):
        y_true, y_prob = random_predictions
        result = bootstrap_auc_diff(y_true, y_prob, y_prob, n_bootstrap=50)
        assert abs(result["diff"]) < 1e-10

    def test_diff_equals_auc_a_minus_auc_b(self, random_predictions):
        y_true, y_prob_a = random_predictions
        rng = np.random.default_rng(7)
        y_prob_b = rng.uniform(0, 1, len(y_true))
        result = bootstrap_auc_diff(y_true, y_prob_a, y_prob_b, n_bootstrap=50)
        expected_diff = roc_auc_score(y_true, y_prob_a) - roc_auc_score(y_true, y_prob_b)
        assert abs(result["diff"] - expected_diff) < 1e-10
