"""
src/preprocessing.py
─────────────────────
Data cleaning, merging, and feature engineering for the Olist dataset.

Why this exists:
  The cleaning steps are long but deterministic.  Keeping them here means
  every notebook that needs the master dataset calls one function instead of
  repeating 80 lines of merge logic.

Each public function is explained with a docstring and a short rationale so
a notebook can display it with help().
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional


# ─── Date Parsing ─────────────────────────────────────────────────────────────

def parse_dates(df: pd.DataFrame, date_columns: list[str]) -> pd.DataFrame:
    """
    Convert specified columns to datetime in-place and return the dataframe.

    Why: Pandas reads date strings as object dtype; we need datetime for
    arithmetic (e.g. delivery delay).

    Parameters
    ----------
    df : pd.DataFrame
    date_columns : list of str

    Returns
    -------
    pd.DataFrame  (same object, modified in-place)
    """
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            print(f"  [parse_dates] Column not found, skipping: {col}")
    return df


# ─── Delivery Features ────────────────────────────────────────────────────────

def compute_delivery_features(orders: pd.DataFrame) -> pd.DataFrame:
    """
    Add delivery-related derived columns to the orders dataframe.

    New columns
    -----------
    delivery_delay_days : int
        Actual delivery date minus estimated delivery date (days).
        Positive = late.  Negative = early.
    is_late : int (0 or 1)
        1 if delivery_delay_days > 0, else 0.
    days_to_deliver : float
        Days from purchase to actual delivery.
    days_to_approve : float
        Days from purchase to order approval.

    Why: These are the core operational KPIs we analyse and model.
    """
    df = orders.copy()

    delivered_mask = df["order_status"] == "delivered"

    df["delivery_delay_days"] = np.where(
        delivered_mask,
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"])
        .dt.days,
        np.nan,
    )
    df["is_late"] = np.where(
        delivered_mask & df["delivery_delay_days"].notna(),
        (df["delivery_delay_days"] > 0).astype(int),
        np.nan,
    )
    df["days_to_deliver"] = (
        df["order_delivered_customer_date"] - df["order_purchase_timestamp"]
    ).dt.days
    df["days_to_approve"] = (
        df["order_approved_at"] - df["order_purchase_timestamp"]
    ).dt.days

    return df


# ─── Order-level Aggregates ───────────────────────────────────────────────────

def aggregate_order_items(items: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate order-items to one row per order.

    Returns columns: order_id, order_value, freight_value, n_items
    """
    agg = (
        items.groupby("order_id")
        .agg(
            order_value=("price", "sum"),
            freight_value=("freight_value", "sum"),
            n_items=("order_item_id", "count"),
        )
        .reset_index()
    )
    agg["freight_ratio"] = np.where(
        agg["order_value"] > 0,
        agg["freight_value"] / agg["order_value"],
        np.nan,
    )
    return agg


def aggregate_payments(payments: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate payments to one row per order.

    Returns: order_id, payment_value, payment_installments, payment_type (most common)
    """
    # Most common payment type per order
    ptype = (
        payments.groupby("order_id")["payment_type"]
        .agg(lambda x: x.mode().iloc[0] if len(x) else "unknown")
        .reset_index()
    )
    pval = (
        payments.groupby("order_id")
        .agg(
            payment_value=("payment_value", "sum"),
            payment_installments=("payment_installments", "max"),
        )
        .reset_index()
    )
    return ptype.merge(pval, on="order_id", how="left")


def aggregate_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one review per order (most recent if multiple exist).

    Returns: order_id, review_score
    """
    reviews = reviews.copy()
    if "review_creation_date" in reviews.columns:
        reviews["review_creation_date"] = pd.to_datetime(
            reviews["review_creation_date"], errors="coerce"
        )
        reviews = reviews.sort_values("review_creation_date", ascending=False)

    return (
        reviews.drop_duplicates(subset="order_id", keep="first")[
            ["order_id", "review_score"]
        ]
        .copy()
        .reset_index(drop=True)
    )


# ─── Product Features ─────────────────────────────────────────────────────────

def merge_product_features(
    items: pd.DataFrame,
    products: pd.DataFrame,
    translation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return a per-order product feature dataframe.

    Uses the first item's product for category/weight when multiple items exist.
    This is a simplifying assumption — document it.

    Returns: order_id, product_category_english, product_weight_g, product_photos_qty
    """
    prod = products.merge(
        translation, on="product_category_name", how="left"
    )
    # Fill missing translations
    prod["product_category_english"] = prod[
        "product_category_name_english"
    ].fillna(prod["product_category_name"]).fillna("unknown")

    # Grab first item per order
    first_item = items.sort_values("order_item_id").drop_duplicates(
        subset="order_id", keep="first"
    )[["order_id", "product_id"]]

    merged = first_item.merge(prod, on="product_id", how="left")
    return merged[
        [
            "order_id",
            "product_category_english",
            "product_weight_g",
            "product_photos_qty",
        ]
    ].copy()


# ─── Master Merge ─────────────────────────────────────────────────────────────

def build_master(
    orders: pd.DataFrame,
    items: pd.DataFrame,
    payments: pd.DataFrame,
    reviews: pd.DataFrame,
    customers: pd.DataFrame,
    products: pd.DataFrame,
    sellers: pd.DataFrame,
    translation: pd.DataFrame,
    cfg: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Merge all Olist tables into a single flat master dataframe.

    Only delivered orders are kept (configurable via cfg).

    Steps
    -----
    1. Filter to delivered orders only.
    2. Compute delivery features.
    3. Aggregate items, payments, reviews.
    4. Merge product and seller features.
    5. Merge customer geo features.
    6. Add calendar features.
    7. Encode categoricals.

    Parameters
    ----------
    orders, items, payments, reviews, customers, products,
    sellers, translation : pd.DataFrame
        Raw Olist dataframes.
    cfg : dict, optional
        Project config.  Used for delivered_status and min_order_value.

    Returns
    -------
    pd.DataFrame
        One row per delivered order with all analysis-ready features.
    """
    delivered_status = "delivered"
    min_order_value = 0.01
    if cfg:
        delivered_status = cfg["preprocessing"].get("delivered_status", "delivered")
        min_order_value = cfg["preprocessing"].get("min_order_value", 0.01)

    # 1. Filter
    delivered = orders[orders["order_status"] == delivered_status].copy()
    print(f"  Delivered orders: {len(delivered):,} / {len(orders):,} total")

    # 2. Delivery features
    delivered = compute_delivery_features(delivered)

    # 3. Aggregate items, payments, reviews
    items_agg   = aggregate_order_items(items)
    pay_agg     = aggregate_payments(payments)
    review_agg  = aggregate_reviews(reviews)
    prod_feats  = merge_product_features(items, products, translation)

    # 4. Seller state (first seller per order)
    first_seller = (
        items.sort_values("order_item_id")
        .drop_duplicates(subset="order_id", keep="first")[["order_id", "seller_id"]]
        .merge(sellers[["seller_id", "seller_state"]], on="seller_id", how="left")
    )

    # 5. Merge everything
    master = (
        delivered
        .merge(items_agg,   on="order_id",   how="left")
        .merge(pay_agg,     on="order_id",   how="left")
        .merge(review_agg,  on="order_id",   how="left")
        .merge(prod_feats,  on="order_id",   how="left")
        .merge(first_seller[["order_id", "seller_state"]], on="order_id", how="left")
        .merge(
            customers[["customer_id", "customer_unique_id", "customer_state"]],
            on="customer_id", how="left",
        )
    )

    # 6. Filter out suspicious order values
    master = master[master["order_value"].fillna(0) >= min_order_value].copy()

    # 7. Calendar features (from purchase timestamp)
    master["purchase_month"]      = master["order_purchase_timestamp"].dt.month
    master["purchase_dayofweek"]  = master["order_purchase_timestamp"].dt.dayofweek
    master["purchase_year"]       = master["order_purchase_timestamp"].dt.year

    # 8. Encode high-cardinality categoricals with integer codes
    for col, new_col in [
        ("customer_state",          "customer_state_encoded"),
        ("seller_state",            "seller_state_encoded"),
        ("product_category_english","product_category_encoded"),
        ("payment_type",            "payment_type_encoded"),
    ]:
        if col in master.columns:
            master[new_col] = master[col].astype("category").cat.codes

    print(f"  Master shape: {master.shape}")
    print(f"  Date range: {master['order_purchase_timestamp'].min().date()} "
          f"→ {master['order_purchase_timestamp'].max().date()}")
    return master.reset_index(drop=True)


# ─── Schema Checks ────────────────────────────────────────────────────────────

EXPECTED_SCHEMAS = {
    "orders": [
        "order_id", "customer_id", "order_status",
        "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "order_items": [
        "order_id", "order_item_id", "product_id", "seller_id",
        "shipping_limit_date", "price", "freight_value",
    ],
    "order_payments": [
        "order_id", "payment_sequential", "payment_type",
        "payment_installments", "payment_value",
    ],
    "order_reviews": [
        "review_id", "order_id", "review_score",
    ],
    "customers": [
        "customer_id", "customer_unique_id",
        "customer_zip_code_prefix", "customer_city", "customer_state",
    ],
    "products": [
        "product_id", "product_category_name",
        "product_weight_g",
    ],
    "sellers": [
        "seller_id", "seller_zip_code_prefix", "seller_city", "seller_state",
    ],
}


def check_schema(df: pd.DataFrame, table_name: str) -> list[str]:
    """
    Check that a dataframe contains all expected columns for a given table.

    Returns
    -------
    list[str]
        List of missing column names (empty if all present).
    """
    expected = EXPECTED_SCHEMAS.get(table_name, [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        print(f"  [schema] MISSING columns in '{table_name}': {missing}")
    else:
        print(f"  [schema] '{table_name}' OK — all {len(expected)} expected columns present")
    return missing
