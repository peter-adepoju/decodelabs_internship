"""
tests/test_rfm.py
──────────────────
Tests for src/rfm.py

All data is SYNTHETIC FIXTURES from conftest.py.
"""

import pandas as pd
import pytest

from src.rfm import compute_rfm, score_rfm, assign_segments, full_rfm_pipeline


# ─── compute_rfm ──────────────────────────────────────────────────────────────

def test_compute_rfm_one_row_per_customer(tiny_master):
    rfm = compute_rfm(tiny_master)
    n_unique = tiny_master["customer_unique_id"].nunique()
    assert rfm.shape[0] == n_unique

def test_compute_rfm_recency_positive(tiny_master):
    rfm = compute_rfm(tiny_master)
    assert (rfm["recency"] > 0).all()

def test_compute_rfm_frequency_positive(tiny_master):
    rfm = compute_rfm(tiny_master)
    assert (rfm["frequency"] >= 1).all()

def test_compute_rfm_monetary_positive(tiny_master):
    rfm = compute_rfm(tiny_master)
    assert (rfm["monetary"] > 0).all()

def test_compute_rfm_customer_with_two_orders(tiny_master):
    """Customer u1 placed orders o0 and o2 — frequency should be 2."""
    rfm = compute_rfm(tiny_master)
    u1_row = rfm[rfm["customer_unique_id"] == "u1"]
    assert len(u1_row) == 1
    assert u1_row.iloc[0]["frequency"] == 2

def test_compute_rfm_raises_on_missing_columns():
    bad_df = pd.DataFrame({"order_id": ["x"], "order_value": [10.0]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_rfm(bad_df)


# ─── score_rfm ────────────────────────────────────────────────────────────────

def test_score_rfm_adds_r_f_m_columns(tiny_master):
    rfm = compute_rfm(tiny_master)
    scored = score_rfm(rfm)
    assert {"R", "F", "M", "RFM_Score"}.issubset(scored.columns)

def test_score_rfm_values_in_range(tiny_master):
    rfm = score_rfm(compute_rfm(tiny_master))
    for col in ["R", "F", "M"]:
        assert rfm[col].between(1, 5).all(), f"{col} has values outside [1,5]"

def test_rfm_score_string_length(tiny_master):
    rfm = score_rfm(compute_rfm(tiny_master))
    assert (rfm["RFM_Score"].str.len() == 3).all()


# ─── assign_segments ──────────────────────────────────────────────────────────

def test_assign_segments_no_nulls(tiny_master):
    rfm = assign_segments(score_rfm(compute_rfm(tiny_master)))
    assert rfm["segment"].notna().all()

def test_assign_segments_raises_without_rfm_score(tiny_master):
    rfm = compute_rfm(tiny_master)  # not scored yet
    with pytest.raises(ValueError, match="RFM_Score"):
        assign_segments(rfm)


# ─── full_rfm_pipeline ────────────────────────────────────────────────────────

def test_full_pipeline_returns_all_columns(tiny_master):
    result = full_rfm_pipeline(tiny_master)
    expected = ["customer_unique_id", "recency", "frequency", "monetary",
                "R", "F", "M", "RFM_Score", "segment"]
    for col in expected:
        assert col in result.columns, f"Missing column: {col}"

def test_full_pipeline_row_count(tiny_master):
    result = full_rfm_pipeline(tiny_master)
    assert result.shape[0] == tiny_master["customer_unique_id"].nunique()
