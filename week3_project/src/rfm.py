"""
src/rfm.py
──────────
RFM (Recency, Frequency, Monetary) analysis and customer segmentation.

RFM is a proven, interpretable framework for ranking customers by:
  - Recency   : how recently they purchased (lower = better)
  - Frequency : how often they purchase    (higher = better)
  - Monetary  : how much they spend        (higher = better)

Each dimension is scored 1-5, giving a three-digit RFM code (e.g. "555" = best customer).
Customers are then grouped into named segments.

Why this module exists:
  The RFM calculation is straightforward but has several non-obvious details
  (tie-breaking, reverse scoring for recency, label assignment).  Centralising
  it here keeps notebooks clean and makes the logic testable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


# ─── RFM Computation ──────────────────────────────────────────────────────────

def compute_rfm(
    master: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Compute raw RFM values per unique customer.

    Parameters
    ----------
    master : pd.DataFrame
        Master dataframe with columns:
        customer_unique_id, order_purchase_timestamp, order_id, order_value
    reference_date : pd.Timestamp, optional
        The anchor date for recency calculation.
        Defaults to max(order_purchase_timestamp) + 1 day.

    Returns
    -------
    pd.DataFrame
        One row per customer with columns:
        customer_unique_id, last_purchase, recency, frequency, monetary
    """
    required = {"customer_unique_id", "order_purchase_timestamp",
                "order_id", "order_value"}
    missing = required - set(master.columns)
    if missing:
        raise ValueError(f"master is missing columns: {missing}")

    if reference_date is None:
        reference_date = master["order_purchase_timestamp"].max() + pd.Timedelta(days=1)

    rfm = (
        master.groupby("customer_unique_id")
        .agg(
            last_purchase  = ("order_purchase_timestamp", "max"),
            frequency      = ("order_id",                 "count"),
            monetary       = ("order_value",               "sum"),
        )
        .reset_index()
    )
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days
    rfm["reference_date"] = reference_date

    return rfm


# ─── RFM Scoring ──────────────────────────────────────────────────────────────

def _safe_qcut(series: pd.Series, n: int, reverse: bool = False) -> pd.Series:
    """
    Assign 1-n quantile scores to a series.

    Parameters
    ----------
    series  : pd.Series
    n       : int  number of bins (5 = quintiles)
    reverse : bool If True, lowest raw value gets highest score (used for recency).

    Returns
    -------
    pd.Series of int (1 to n)
    """
    labels = list(range(n, 0, -1)) if reverse else list(range(1, n + 1))
    # rank(method='first') breaks ties deterministically
    ranked = series.rank(method="first", ascending=True)
    scores = pd.qcut(ranked, n, labels=labels)
    return scores.astype(int)


def score_rfm(rfm: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Add R, F, M score columns (each 1-n_bins) and an RFM_Score string.

    Parameters
    ----------
    rfm    : pd.DataFrame  Output of compute_rfm().
    n_bins : int           Number of quantile bins (default 5).

    Returns
    -------
    pd.DataFrame with extra columns: R, F, M, RFM_Score
    """
    rfm = rfm.copy()
    rfm["R"] = _safe_qcut(rfm["recency"],  n_bins, reverse=True)  # low recency → high score
    rfm["F"] = _safe_qcut(rfm["frequency"], n_bins, reverse=False)
    rfm["M"] = _safe_qcut(rfm["monetary"],  n_bins, reverse=False)
    rfm["RFM_Score"] = (
        rfm["R"].astype(str) + rfm["F"].astype(str) + rfm["M"].astype(str)
    )
    return rfm


# ─── Segment Assignment ───────────────────────────────────────────────────────

SEGMENT_MAP = {
    # Champions: bought recently, buy often, spend most
    frozenset(["555","554","544","545","454","455","445","444"]): "Champions",
    # Loyal
    frozenset(["354","355","344","345","343","342","341",
               "453","452","451","443","442","441"]): "Loyal",
    # Potential Loyalists
    frozenset(["512","511","522","521","531","532","541","542",
               "421","422","431","432"]): "Potential Loyalist",
    # Recent customers
    frozenset(["511","412","411","311"]): "New Customer",
    # Promising
    frozenset(["525","524","523","413","414","415",
               "315","314","313"]): "Promising",
    # Need Attention
    frozenset(["535","534","443","434","343","334","325","324"]): "Need Attention",
    # About to Sleep
    frozenset(["331","321","312","221","213"]): "About to Sleep",
    # At Risk
    frozenset(["255","254","245","244","253","252","243","242",
               "235","234","225","224","153","152","145","143",
               "142","135","134","125","124"]): "At Risk",
    # Cant Lose Them
    frozenset(["155","154","144","214","215","115","114","113"]): "Cant Lose Them",
    # Hibernating
    frozenset(["332","322","231","241","251","233","232",
               "223","222","132","123","122","212","211"]): "Hibernating",
    # Lost
    frozenset(["111","112","121","131","141","151"]): "Lost",
}


def _build_score_to_segment() -> dict[str, str]:
    """Build a flat lookup dict from RFM score string to segment name."""
    lookup = {}
    for score_set, segment in SEGMENT_MAP.items():
        for score in score_set:
            lookup[score] = segment
    return lookup


_SCORE_TO_SEGMENT = _build_score_to_segment()


def assign_segments(rfm: pd.DataFrame) -> pd.DataFrame:
    """
    Map each customer's RFM_Score to a business segment label.

    Any score not found in the map defaults to "Other".

    Parameters
    ----------
    rfm : pd.DataFrame  Output of score_rfm().

    Returns
    -------
    pd.DataFrame with extra column: segment
    """
    if "RFM_Score" not in rfm.columns:
        raise ValueError("rfm must have an 'RFM_Score' column. Run score_rfm() first.")
    rfm = rfm.copy()
    rfm["segment"] = rfm["RFM_Score"].map(_SCORE_TO_SEGMENT).fillna("Other")
    return rfm


# ─── Convenience Wrapper ──────────────────────────────────────────────────────

def full_rfm_pipeline(
    master: pd.DataFrame,
    reference_date: Optional[pd.Timestamp] = None,
    n_bins: int = 5,
) -> pd.DataFrame:
    """
    Run compute_rfm → score_rfm → assign_segments in one call.

    Returns
    -------
    pd.DataFrame
        One row per customer with all RFM columns plus segment.
    """
    rfm = compute_rfm(master, reference_date=reference_date)
    rfm = score_rfm(rfm, n_bins=n_bins)
    rfm = assign_segments(rfm)
    print(f"  RFM pipeline complete: {len(rfm):,} customers")
    print(rfm["segment"].value_counts().to_string())
    return rfm
