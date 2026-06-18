"""
tests/conftest.py
─────────────────
Shared pytest fixtures.

IMPORTANT: All data here is SYNTHETIC TEST FIXTURE DATA.
It is NOT used in any analysis.  It exists only to let the tests run
without needing the full 100 MB Olist dataset.

The shapes and column names mirror the real Olist schema so that the
preprocessing and RFM functions can be tested correctly.
"""

import numpy as np
import pandas as pd
import pytest


# ─── Minimal Olist-shaped fixtures ────────────────────────────────────────────

@pytest.fixture
def tiny_orders():
    """6-row fixture — same columns as olist_orders_dataset.csv."""
    return pd.DataFrame({
        "order_id": [f"o{i}" for i in range(6)],
        "customer_id": [f"c{i}" for i in range(6)],
        "order_status": ["delivered"] * 5 + ["canceled"],
        "order_purchase_timestamp": pd.to_datetime([
            "2017-01-01", "2017-03-15", "2017-06-01",
            "2018-01-10", "2018-06-20", "2018-07-01",
        ]),
        "order_approved_at": pd.to_datetime([
            "2017-01-02", "2017-03-16", "2017-06-02",
            "2018-01-11", "2018-06-21", "2018-07-02",
        ]),
        "order_delivered_carrier_date": pd.to_datetime([
            "2017-01-05", "2017-03-18", "2017-06-07",
            "2018-01-13", "2018-06-25", np.nan,
        ]),
        "order_delivered_customer_date": pd.to_datetime([
            "2017-01-10", "2017-03-22", "2017-06-12",
            "2018-01-20", "2018-07-01", np.nan,
        ]),
        "order_estimated_delivery_date": pd.to_datetime([
            "2017-01-12", "2017-03-20", "2017-06-10",
            "2018-01-25", "2018-06-28", "2018-07-15",
        ]),
    })


@pytest.fixture
def tiny_items():
    """6-row fixture — same columns as olist_order_items_dataset.csv."""
    return pd.DataFrame({
        "order_id":     [f"o{i}" for i in range(6)],
        "order_item_id": [1] * 6,
        "product_id":   ["p1", "p2", "p1", "p3", "p2", "p1"],
        "seller_id":    ["s1", "s1", "s2", "s2", "s1", "s2"],
        "shipping_limit_date": pd.to_datetime(["2017-01-03"] * 6),
        "price":        [100.0, 250.0, 50.0, 400.0, 75.0, 120.0],
        "freight_value":[15.0,  20.0,  10.0, 30.0,  12.0, 18.0],
    })


@pytest.fixture
def tiny_payments():
    """6-row fixture."""
    return pd.DataFrame({
        "order_id":              [f"o{i}" for i in range(6)],
        "payment_sequential":    [1] * 6,
        "payment_type":          ["credit_card", "boleto", "credit_card",
                                  "credit_card", "voucher", "credit_card"],
        "payment_installments":  [3, 1, 1, 6, 1, 2],
        "payment_value":         [115.0, 270.0, 60.0, 430.0, 87.0, 138.0],
    })


@pytest.fixture
def tiny_reviews():
    """5-row fixture (canceled order o5 has no review)."""
    return pd.DataFrame({
        "review_id":    [f"r{i}" for i in range(5)],
        "order_id":     [f"o{i}" for i in range(5)],
        "review_score": [5, 4, 3, 5, 2],
    })


@pytest.fixture
def tiny_customers():
    return pd.DataFrame({
        "customer_id":           [f"c{i}" for i in range(6)],
        "customer_unique_id":    ["u1", "u2", "u1", "u3", "u4", "u5"],
        "customer_zip_code_prefix": ["01310"] * 6,
        "customer_city":         ["Sao Paulo"] * 6,
        "customer_state":        ["SP", "RJ", "SP", "MG", "BA", "PR"],
    })


@pytest.fixture
def tiny_products():
    return pd.DataFrame({
        "product_id":              ["p1", "p2", "p3"],
        "product_category_name":   ["cama_mesa_banho", "esporte_lazer", "informatica_acessorios"],
        "product_name_lenght":     [30, 25, 40],
        "product_description_lenght": [200, 150, 300],
        "product_photos_qty":      [3, 2, 5],
        "product_weight_g":        [500.0, 800.0, 1200.0],
        "product_length_cm":       [30.0, 20.0, 40.0],
        "product_height_cm":       [10.0, 15.0, 20.0],
        "product_width_cm":        [25.0, 18.0, 35.0],
    })


@pytest.fixture
def tiny_sellers():
    return pd.DataFrame({
        "seller_id":              ["s1", "s2"],
        "seller_zip_code_prefix": ["01310", "04038"],
        "seller_city":            ["Sao Paulo", "Sao Paulo"],
        "seller_state":           ["SP", "SP"],
    })


@pytest.fixture
def tiny_translation():
    return pd.DataFrame({
        "product_category_name":         ["cama_mesa_banho", "esporte_lazer", "informatica_acessorios"],
        "product_category_name_english": ["bed_bath_table",  "sports_leisure", "computers_accessories"],
    })


@pytest.fixture
def tiny_master(tiny_orders, tiny_items, tiny_payments, tiny_reviews,
                tiny_customers, tiny_products, tiny_sellers, tiny_translation):
    """
    A minimal master dataframe built from the tiny fixtures.
    Uses the real build_master() function to ensure integration tests are valid.
    """
    from src.preprocessing import build_master, parse_dates

    date_cols = [
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    orders = parse_dates(tiny_orders.copy(), date_cols)

    master = build_master(
        orders, tiny_items, tiny_payments, tiny_reviews,
        tiny_customers, tiny_products, tiny_sellers, tiny_translation,
    )
    return master
