"""
src/utils.py
─────────────
General utilities: artifact saving, table export, seed setting.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Reproducibility ──────────────────────────────────────────────────────────

def set_all_seeds(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and (if available) PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    except (ImportError, OSError):
        # Torch is unavailable or failed to initialize.
        pass

    print(f"Random seed set to {seed}")


# ─── Table Saving ─────────────────────────────────────────────────────────────

def save_table(
    df: pd.DataFrame,
    filename: str,
    reports_dir: str | Path = "reports/tables",
    paper_dir: str | Path   = "paper_or_report/tables",
    fmt: str = "csv",
    index: bool = False,
) -> list[Path]:
    """
    Save a dataframe as CSV (and optionally XLSX) to both output directories.

    Parameters
    ----------
    df         : pd.DataFrame
    filename   : str   e.g. 'model_comparison' (no extension)
    reports_dir, paper_dir : output directories
    fmt        : 'csv' or 'xlsx'
    index      : whether to include the dataframe index

    Returns
    -------
    list of saved Path objects
    """
    saved = []
    for out_dir in [reports_dir, paper_dir]:
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        out_path = p / f"{filename}.{fmt}"
        if fmt == "csv":
            df.to_csv(out_path, index=index)
        elif fmt == "xlsx":
            df.to_excel(out_path, index=index)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        saved.append(out_path)
        print(f"  Saved table: {out_path}")
    return saved


# ─── Metrics Persistence ──────────────────────────────────────────────────────

def save_metrics(metrics: dict, filepath: str | Path) -> Path:
    """
    Save a metrics dictionary as JSON.

    Parameters
    ----------
    metrics  : dict
    filepath : path to the output .json file

    Returns
    -------
    Path
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved metrics: {filepath}")
    return filepath


def load_metrics(filepath: str | Path) -> dict:
    """Load a metrics JSON file saved by save_metrics()."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── Data Summary Helpers ─────────────────────────────────────────────────────

def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary of missing values per column.

    Returns
    -------
    pd.DataFrame with columns: column, n_missing, pct_missing
    """
    n = len(df)
    missing = df.isnull().sum()
    pct = (missing / n * 100).round(2)
    summary = pd.DataFrame({
        "column":     missing.index,
        "n_missing":  missing.values,
        "pct_missing":pct.values,
    })
    return summary[summary["n_missing"] > 0].sort_values("pct_missing", ascending=False)


def duplicate_summary(df: pd.DataFrame, subset: list[str] | None = None) -> dict:
    """
    Report duplicate rows.

    Parameters
    ----------
    df     : pd.DataFrame
    subset : columns to check for duplicates (None = all columns)

    Returns
    -------
    dict with n_total, n_duplicates, pct_duplicates
    """
    n_total = len(df)
    n_dupes = df.duplicated(subset=subset).sum()
    return {
        "n_total":       n_total,
        "n_duplicates":  int(n_dupes),
        "pct_duplicates":round(n_dupes / n_total * 100, 3),
    }


def cardinality_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique value counts per column.

    Useful for deciding which columns need encoding vs which are IDs.
    """
    rows = []
    for col in df.columns:
        rows.append({
            "column":      col,
            "dtype":       str(df[col].dtype),
            "n_unique":    df[col].nunique(),
            "pct_unique":  round(df[col].nunique() / len(df) * 100, 2),
            "sample":      str(df[col].dropna().unique()[:3].tolist()),
        })
    return pd.DataFrame(rows)
