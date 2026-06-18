"""
src/sql_queries.py
───────────────────
SQL analysis using DuckDB (in-process — no server required).

Why DuckDB:
  It runs entirely inside Python, reads parquet and pandas dataframes
  directly, and supports full SQL syntax.  No PostgreSQL installation needed,
  making the project easier to run on any machine.

Why this module exists:
  SQL queries are long strings.  Keeping them here (and in .sql files) means
  notebooks stay readable.  Each function returns a pandas DataFrame.

Usage (in a notebook):
  import duckdb
  from src.sql_queries import get_revenue_by_state, get_top_categories
  con = duckdb.connect()
  con.register('master', master_df)   # register pandas df as a SQL table
  rev = get_revenue_by_state(con)
  print(rev)
"""

from __future__ import annotations

import pandas as pd


# ─── Revenue Analysis ─────────────────────────────────────────────────────────

def get_revenue_by_state(con) -> pd.DataFrame:
    """
    Total revenue and order count by customer state, ranked by revenue.
    Table required: master
    """
    return con.execute("""
        SELECT
            customer_state,
            ROUND(SUM(order_value), 2)                  AS total_revenue_brl,
            COUNT(*)                                     AS num_orders,
            ROUND(AVG(order_value), 2)                  AS avg_order_value,
            ROUND(SUM(order_value) * 100.0
                  / SUM(SUM(order_value)) OVER (), 2)   AS revenue_share_pct
        FROM master
        GROUP BY customer_state
        ORDER BY total_revenue_brl DESC
    """).df()


def get_top_categories(con, top_n: int = 20) -> pd.DataFrame:
    """
    Top product categories by revenue, order count, and average review.
    Table required: master
    """
    return con.execute(f"""
        SELECT
            product_category_english,
            ROUND(SUM(order_value), 2)      AS total_revenue_brl,
            COUNT(*)                         AS num_orders,
            ROUND(AVG(order_value), 2)       AS avg_order_value,
            ROUND(AVG(review_score), 3)      AS avg_review_score,
            ROUND(AVG(delivery_delay_days), 2) AS avg_delivery_delay_days
        FROM master
        WHERE product_category_english IS NOT NULL
        GROUP BY product_category_english
        ORDER BY total_revenue_brl DESC
        LIMIT {top_n}
    """).df()


def get_monthly_revenue(con) -> pd.DataFrame:
    """
    Monthly revenue and order volume trend.
    Table required: master
    """
    return con.execute("""
        SELECT
            DATE_TRUNC('month', order_purchase_timestamp)   AS month,
            ROUND(SUM(order_value), 2)                      AS total_revenue_brl,
            COUNT(*)                                         AS num_orders,
            ROUND(AVG(order_value), 2)                      AS avg_order_value
        FROM master
        GROUP BY DATE_TRUNC('month', order_purchase_timestamp)
        ORDER BY month
    """).df()


# ─── Delivery Performance ─────────────────────────────────────────────────────

def get_delivery_performance_by_state(con) -> pd.DataFrame:
    """
    Average delivery delay and late-delivery rate by customer state.
    Table required: master
    """
    return con.execute("""
        SELECT
            customer_state,
            COUNT(*)                                    AS num_orders,
            ROUND(AVG(delivery_delay_days), 2)          AS avg_delay_days,
            ROUND(AVG(is_late) * 100, 2)                AS late_rate_pct,
            ROUND(AVG(days_to_deliver), 2)              AS avg_days_to_deliver
        FROM master
        WHERE delivery_delay_days IS NOT NULL
        GROUP BY customer_state
        ORDER BY late_rate_pct DESC
    """).df()


def get_delivery_performance_by_seller_state(con) -> pd.DataFrame:
    """
    Delivery metrics grouped by seller state.
    Table required: master
    """
    return con.execute("""
        SELECT
            seller_state,
            COUNT(*)                            AS num_orders,
            ROUND(AVG(delivery_delay_days), 2)  AS avg_delay_days,
            ROUND(AVG(is_late) * 100, 2)        AS late_rate_pct
        FROM master
        WHERE delivery_delay_days IS NOT NULL
          AND seller_state IS NOT NULL
        GROUP BY seller_state
        ORDER BY num_orders DESC
    """).df()


def get_late_orders_by_category(con) -> pd.DataFrame:
    """
    Late delivery rate per product category.
    Table required: master
    """
    return con.execute("""
        SELECT
            product_category_english,
            COUNT(*)                            AS num_orders,
            ROUND(AVG(is_late) * 100, 2)        AS late_rate_pct,
            ROUND(AVG(delivery_delay_days), 2)  AS avg_delay_days
        FROM master
        WHERE is_late IS NOT NULL
          AND product_category_english IS NOT NULL
        GROUP BY product_category_english
        HAVING COUNT(*) >= 50
        ORDER BY late_rate_pct DESC
        LIMIT 20
    """).df()


# ─── Customer Analysis ────────────────────────────────────────────────────────

def get_repeat_purchase_rate(con) -> pd.DataFrame:
    """
    Proportion of customers with more than one order.
    Table required: master
    """
    return con.execute("""
        WITH customer_orders AS (
            SELECT
                customer_unique_id,
                COUNT(DISTINCT order_id) AS n_orders
            FROM master
            GROUP BY customer_unique_id
        )
        SELECT
            COUNT(*)                                   AS total_customers,
            SUM(CASE WHEN n_orders > 1 THEN 1 ELSE 0 END) AS repeat_customers,
            ROUND(AVG(CASE WHEN n_orders > 1 THEN 1.0 ELSE 0 END) * 100, 2)
                                                       AS repeat_rate_pct,
            ROUND(AVG(n_orders), 3)                    AS avg_orders_per_customer
        FROM customer_orders
    """).df()


def get_top_sellers(con, top_n: int = 20) -> pd.DataFrame:
    """
    Top sellers by revenue and their average review score.
    Table required: master
    """
    return con.execute(f"""
        SELECT
            seller_state,
            COUNT(DISTINCT order_id)            AS num_orders,
            ROUND(SUM(order_value), 2)          AS total_revenue_brl,
            ROUND(AVG(review_score), 3)         AS avg_review_score,
            ROUND(AVG(is_late) * 100, 2)        AS late_rate_pct
        FROM master
        WHERE seller_state IS NOT NULL
        GROUP BY seller_state
        ORDER BY total_revenue_brl DESC
        LIMIT {top_n}
    """).df()


def get_payment_type_breakdown(con) -> pd.DataFrame:
    """
    Revenue and order share by payment type.
    Table required: master
    """
    return con.execute("""
        SELECT
            payment_type,
            COUNT(*)                                        AS num_orders,
            ROUND(SUM(order_value), 2)                      AS total_revenue_brl,
            ROUND(AVG(order_value), 2)                      AS avg_order_value,
            ROUND(AVG(payment_installments), 2)             AS avg_installments,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS order_share_pct
        FROM master
        WHERE payment_type IS NOT NULL
        GROUP BY payment_type
        ORDER BY num_orders DESC
    """).df()


# ─── Review Analysis ──────────────────────────────────────────────────────────

def get_delay_vs_review(con) -> pd.DataFrame:
    """
    Average review score grouped by delivery delay bucket.
    Table required: master
    """
    return con.execute("""
        SELECT
            CASE
                WHEN delivery_delay_days < -7  THEN 'Very early (< -7d)'
                WHEN delivery_delay_days < 0   THEN 'Early (-7 to -1d)'
                WHEN delivery_delay_days = 0   THEN 'On time (0d)'
                WHEN delivery_delay_days <= 7  THEN 'Late (1-7d)'
                WHEN delivery_delay_days <= 14 THEN 'Late (8-14d)'
                ELSE 'Very late (>14d)'
            END                                         AS delay_bucket,
            COUNT(*)                                    AS num_orders,
            ROUND(AVG(review_score), 3)                 AS avg_review_score,
            ROUND(STDDEV(review_score), 3)              AS std_review_score
        FROM master
        WHERE delivery_delay_days IS NOT NULL
          AND review_score IS NOT NULL
        GROUP BY delay_bucket
        ORDER BY MIN(delivery_delay_days)
    """).df()
