#!/usr/bin/env python3
"""
scripts/download_data.py
─────────────────────────
Downloads the Olist Brazilian E-Commerce dataset from Kaggle.

Requirements
────────────
1. A Kaggle account (free) at https://www.kaggle.com
2. A Kaggle API token: Account → Settings → Create New Token → kaggle.json
3. Place kaggle.json in ~/.kaggle/kaggle.json  (Linux/Mac)
   or  C:\\Users\\<username>\\.kaggle\\kaggle.json  (Windows)
4. pip install kaggle

Dataset
───────
Source  : https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
Size    : ~100 MB uncompressed (9 CSV files)
License : CC BY-NC-SA 4.0

Files expected after download (in data/raw/):
  olist_orders_dataset.csv
  olist_order_items_dataset.csv
  olist_order_payments_dataset.csv
  olist_order_reviews_dataset.csv
  olist_customers_dataset.csv
  olist_products_dataset.csv
  olist_sellers_dataset.csv
  olist_geolocation_dataset.csv
  product_category_name_translation.csv

Usage
─────
  python scripts/download_data.py
  # or via Makefile:
  make download
"""

import subprocess
import sys
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

KAGGLE_DATASET = "olistbr/brazilian-ecommerce"

EXPECTED_FILES = [
    "olist_orders_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_customers_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "product_category_name_translation.csv",
]


def check_already_downloaded() -> bool:
    """Return True if all expected files are already present."""
    missing = [f for f in EXPECTED_FILES if not (RAW_DATA_DIR / f).exists()]
    if not missing:
        print("All Olist CSV files are already present in data/raw/.")
        return True
    print(f"Missing files: {missing}")
    return False


def download_with_kaggle_api() -> bool:
    """Download the dataset using the kaggle CLI."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("ERROR: kaggle package not installed.  Run:  pip install kaggle")
        return False

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading '{KAGGLE_DATASET}' to {RAW_DATA_DIR} ...")
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "kaggle", "datasets", "download",
                "-d", KAGGLE_DATASET,
                "--unzip",
                "-p", str(RAW_DATA_DIR),
            ],
            check=True,
            capture_output=False,
        )
        print("Download complete.")
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Kaggle download failed: {e}")
        return False


def verify_download() -> bool:
    """Check all expected files exist and print their sizes."""
    all_ok = True
    for fname in EXPECTED_FILES:
        fpath = RAW_DATA_DIR / fname
        if fpath.exists():
            size_mb = fpath.stat().st_size / (1024 ** 2)
            print(f"  ✓  {fname}  ({size_mb:.1f} MB)")
        else:
            print(f"  ✗  MISSING: {fname}")
            all_ok = False
    return all_ok


def print_manual_instructions():
    print(
        "\n── Manual download instructions ──────────────────────────────────────"
        "\n1. Go to https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce"
        "\n2. Click 'Download' (you must be logged in)"
        "\n3. Unzip the archive"
        f"\n4. Move all CSV files to:  {RAW_DATA_DIR}"
        "\n──────────────────────────────────────────────────────────────────────\n"
    )


def main():
    print("=" * 60)
    print("Olist Dataset Downloader")
    print("=" * 60)

    if check_already_downloaded():
        verify_download()
        return

    success = download_with_kaggle_api()

    if success:
        print("\nVerifying downloaded files...")
        ok = verify_download()
        if ok:
            print("\nAll files downloaded and verified. You are ready to run notebooks.")
        else:
            print("\nSome files are missing. See manual instructions below.")
            print_manual_instructions()
    else:
        print("\nAutomatic download failed. Please download manually:")
        print_manual_instructions()


if __name__ == "__main__":
    main()
