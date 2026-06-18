"""
tests/test_preprocessing.py
────────────────────────────
Tests for src/preprocessing.py

All tests use the SYNTHETIC FIXTURES from conftest.py.
No real Olist data is required to run these tests.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import (
    parse_dates,
    compute_delivery_features,
    aggregate_order_items,
    aggregate_payments,
    aggregate_reviews,
    build_master,
    check_schema,
    EXPECTED_SCHEMAS,
)


# ─── parse_dates ──────────────────────────────────────────────────────────────

def test_parse_dates_converts_columns(tiny_orders):
    df = tiny_orders.copy()
    # Reset to strings first
    df["order_purchase_timestamp"] = df["order_purchase_timestamp"].astype(str)
    result = parse_dates(df, ["order_purchase_timestamp"])
    assert pd.api.types.is_datetime64_any_dtype(result["order_purchase_timestamp"])


def test_parse_dates_missing_column_does_not_raise(tiny_orders):
    # Should print a warning but not raise
    result = parse_dates(tiny_orders.copy(), ["nonexistent_column"])
    assert "nonexistent_column" not in result.columns


# ─── compute_delivery_features ────────────────────────────────────────────────

def test_delivery_delay_days(tiny_orders):
    result = compute_delivery_features(tiny_orders)
    # o0: delivered 2017-01-10, estimated 2017-01-12 → delay = -2
    row = result[result["order_id"] == "o0"].iloc[0]
    assert row["delivery_delay_days"] == -2

def test_is_late_positive_delay(tiny_orders):
    result = compute_delivery_features(tiny_orders)
    # o1: delivered 2017-03-22, estimated 2017-03-20 → delay = 2 → is_late = 1
    row = result[result["order_id"] == "o1"].iloc[0]
    assert row["is_late"] == 1

def test_is_late_negative_delay(tiny_orders):
    result = compute_delivery_features(tiny_orders)
    # o0: delay = -2 → is_late = 0
    row = result[result["order_id"] == "o0"].iloc[0]
    assert row["is_late"] == 0

def test_canceled_order_has_nan_delay(tiny_orders):
    result = compute_delivery_features(tiny_orders)
    row = result[result["order_id"] == "o5"].iloc[0]
    assert pd.isna(row["delivery_delay_days"])


# ─── aggregate_order_items ────────────────────────────────────────────────────

def test_aggregate_items_shape(tiny_items):
    result = aggregate_order_items(tiny_items)
    assert result.shape[0] == tiny_items["order_id"].nunique()

def test_aggregate_items_columns(tiny_items):
    result = aggregate_order_items(tiny_items)
    assert {"order_id", "order_value", "freight_value", "n_items"}.issubset(result.columns)

def test_aggregate_items_value(tiny_items):
    result = aggregate_order_items(tiny_items)
    val = result[result["order_id"] == "o0"]["order_value"].iloc[0]
    assert val == pytest.approx(100.0)

def test_freight_ratio_is_fraction(tiny_items):
    result = aggregate_order_items(tiny_items)
    ratios = result["freight_ratio"].dropna()
    assert (ratios >= 0).all()
    assert (ratios <= 5).all()  # freight should not exceed 5x item price


# ─── aggregate_payments ───────────────────────────────────────────────────────

def test_aggregate_payments_one_row_per_order(tiny_payments):
    result = aggregate_payments(tiny_payments)
    assert result.shape[0] == tiny_payments["order_id"].nunique()

def test_aggregate_payments_columns(tiny_payments):
    result = aggregate_payments(tiny_payments)
    assert {"order_id", "payment_type", "payment_value"}.issubset(result.columns)


# ─── aggregate_reviews ────────────────────────────────────────────────────────

def test_aggregate_reviews_one_row_per_order(tiny_reviews):
    result = aggregate_reviews(tiny_reviews)
    assert result["order_id"].nunique() == result.shape[0]

def test_review_score_in_valid_range(tiny_reviews):
    result = aggregate_reviews(tiny_reviews)
    assert result["review_score"].between(1, 5).all()


# ─── build_master ─────────────────────────────────────────────────────────────

def test_master_only_delivered(tiny_master):
    """Canceled order o5 should be excluded from master."""
    assert "o5" not in tiny_master["order_id"].values

def test_master_has_expected_columns(tiny_master):
    expected = [
        "order_id", "customer_unique_id", "order_value",
        "delivery_delay_days", "is_late", "review_score",
        "product_category_english", "customer_state",
    ]
    for col in expected:
        assert col in tiny_master.columns, f"Missing: {col}"

def test_master_no_duplicate_order_ids(tiny_master):
    assert tiny_master["order_id"].nunique() == len(tiny_master)

def test_master_order_values_positive(tiny_master):
    assert (tiny_master["order_value"] > 0).all()


# ─── check_schema ─────────────────────────────────────────────────────────────

def test_check_schema_passes_with_correct_columns(tiny_orders):
    missing = check_schema(tiny_orders, "orders")
    assert missing == []

def test_check_schema_detects_missing_column(tiny_orders):
    df = tiny_orders.drop(columns=["order_status"])
    missing = check_schema(df, "orders")
    assert "order_status" in missing

def test_check_schema_unknown_table_returns_empty():
    df = pd.DataFrame({"col": [1]})
    missing = check_schema(df, "nonexistent_table")
    assert missing == []
