#!/usr/bin/env python3
"""
scripts/generate_mock_data.py
==============================
Generate the synthetic mock dataset and all processed split files
required for running tests and notebooks in mock mode.

This script creates:
  - data/mock/           : 60 tiny synthetic images + manifest.csv
  - data/interim/        : manifest.csv and manifest_clean.csv
  - data/processed/      : train.csv, val.csv, test.csv, site_0..4_train.csv

Usage:
  python scripts/generate_mock_data.py

No internet connection required. Completes in seconds on CPU.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import load_config
from src.paths import get_paths
from src.data_utils import make_mock_dataset, dirichlet_split

def main():
    cfg   = load_config()
    paths = get_paths()

    SEED    = cfg["project"]["random_seed"]
    N_SITES = cfg["federated"]["num_clients"]
    ALPHA   = cfg["federated"]["dirichlet_alpha"]
    SITE_NAMES = cfg["federated"]["site_names"]

    print("=" * 55)
    print("FedTB-Nigeria — Mock Dataset Generator")
    print("=" * 55)
    print()

    # ── 1. Generate mock images ──────────────────────────────────────
    print("Step 1: Generating mock images...")
    mock_df = make_mock_dataset(
        save_dir=paths["mock"],
        num_images=60,
        image_size=64,
        seed=SEED,
    )
    print(f"  Generated {len(mock_df)} mock images in {paths['mock']}")
    print()

    # ── 2. Save as interim manifest ──────────────────────────────────
    print("Step 2: Saving interim manifests...")
    manifest_path = paths["interim"] / "manifest.csv"
    mock_df.to_csv(manifest_path, index=False)

    clean_path = paths["interim"] / "manifest_clean.csv"
    mock_df.to_csv(clean_path, index=False)
    print(f"  Saved: {manifest_path}")
    print(f"  Saved: {clean_path}")
    print()

    # ── 3. Train/val/test split ──────────────────────────────────────
    print("Step 3: Creating train/val/test splits...")
    train_df, temp_df = train_test_split(
        mock_df, test_size=0.30,
        stratify=mock_df["label"], random_state=SEED,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50,
        stratify=temp_df["label"], random_state=SEED,
    )

    train_df.to_csv(paths["processed"] / "train.csv",  index=False)
    val_df.to_csv(  paths["processed"] / "val.csv",    index=False)
    test_df.to_csv( paths["processed"] / "test.csv",   index=False)

    print(f"  Train: {len(train_df)}  |  Val: {len(val_df)}  |  Test: {len(test_df)}")
    print()

    # ── 4. Dirichlet site split ──────────────────────────────────────
    print(f"Step 4: Partitioning training data into {N_SITES} sites (alpha={ALPHA})...")
    client_indices = dirichlet_split(
        labels=train_df["label"].values,
        num_clients=N_SITES,
        alpha=max(ALPHA, 1.0),
        min_samples=1,
        seed=SEED,
    )

    for i, indices in enumerate(client_indices):
        site_df = train_df.iloc[indices].copy().reset_index(drop=True)
        site_df["site"] = SITE_NAMES[i]
        fname = paths["processed"] / f"site_{i}_train.csv"
        site_df.to_csv(fname, index=False)
        pos_rate = 100 * (site_df["label"] == 1).mean()
        print(f"  Site {i} ({SITE_NAMES[i][:28]}...): "
              f"n={len(site_df)}, TB+: {pos_rate:.1f}%")

    print()
    print("=" * 55)
    print("Mock data generation complete.")
    print("You can now run notebooks in USE_MOCK_DATA=True mode.")
    print("Run tests with: pytest tests/ -v")
    print("=" * 55)


if __name__ == "__main__":
    main()
