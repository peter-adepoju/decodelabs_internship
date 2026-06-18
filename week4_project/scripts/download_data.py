#!/usr/bin/env python3
"""
scripts/download_data.py
========================
Interactive downloader for the Montgomery County and Shenzhen CXR datasets.

Asks the user before each large download. Checks if files already exist
to avoid re-downloading unnecessarily.

Usage:
  python scripts/download_data.py [--auto]

Options:
  --auto   Download all datasets without asking (for CI/pipeline use).
"""

import sys
import argparse
import requests
import zipfile
import io
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import load_config
from src.paths import get_paths

DATASETS = [
    {
        "name":    "Montgomery County CXR Set",
        "url":     "https://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip",
        "dest":    "montgomery",
        "size_mb": 115,
        "license": "Public domain",
        "source":  "US National Library of Medicine",
    },
    {
        "name":    "Shenzhen Hospital CXR Set",
        "url":     "https://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip",
        "dest":    "shenzhen",
        "size_mb": 1200,
        "license": "Public domain",
        "source":  "Guanganmen Hospital / NLM",
    },
]


def prompt_yes_no(question: str, auto: bool = False) -> bool:
    """Ask a yes/no question. Return True automatically if auto=True."""
    if auto:
        return True
    answer = input(f"{question} [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


def download_and_extract(url: str, dest_dir: Path, dataset_name: str) -> bool:
    """Download a ZIP file from url and extract it to dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading {dataset_name}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest_dir}")

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  ERROR: Download failed — {e}")
        return False

    total_size  = int(response.headers.get("content-length", 0))
    downloaded  = 0
    chunks      = []

    for chunk in response.iter_content(chunk_size=32768):
        chunks.append(chunk)
        downloaded += len(chunk)
        if total_size > 0:
            pct = 100.0 * downloaded / total_size
            mb  = downloaded / 1_000_000
            print(f"  {pct:.1f}%  ({mb:.1f} MB)", end="\r")

    print(f"\n  Downloaded {downloaded / 1_000_000:.1f} MB")

    print("  Extracting ZIP archive...")
    try:
        zf = zipfile.ZipFile(io.BytesIO(b"".join(chunks)))
        zf.extractall(dest_dir)
        print(f"  Extracted to: {dest_dir}")
    except zipfile.BadZipFile as e:
        print(f"  ERROR: Could not extract ZIP — {e}")
        return False

    return True


def count_images(directory: Path) -> int:
    """Recursively count PNG/JPG images in a directory."""
    count = 0
    for ext in ["*.png", "*.PNG", "*.jpg", "*.JPG", "*.jpeg"]:
        count += len(list(directory.rglob(ext)))
    return count


def main():
    parser = argparse.ArgumentParser(description="Download FedTB-Nigeria datasets")
    parser.add_argument("--auto", action="store_true",
                        help="Download all datasets without confirmation prompts")
    args = parser.parse_args()

    cfg   = load_config()
    paths = get_paths()

    print("=" * 60)
    print("FedTB-Nigeria Dataset Downloader")
    print("=" * 60)
    print()
    print("This script will download publicly available chest X-ray")
    print("datasets for TB classification research.")
    print()
    print("Datasets:")
    for ds in DATASETS:
        print(f"  - {ds['name']}")
        print(f"      Source : {ds['source']}")
        print(f"      License: {ds['license']}")
        print(f"      Size   : ~{ds['size_mb']} MB")
        print()

    print("NOTE: You are responsible for complying with each dataset's")
    print("      terms of use. All datasets listed are public domain.")
    print()

    results = {}
    for ds in DATASETS:
        dest_dir = paths["raw"] / ds["dest"]

        # Check if already downloaded
        if dest_dir.exists() and count_images(dest_dir) > 10:
            print(f"[SKIP] {ds['name']}: already downloaded "
                  f"({count_images(dest_dir)} images found in {dest_dir})")
            results[ds["name"]] = "already_exists"
            continue

        question = (f"Download {ds['name']} (~{ds['size_mb']} MB)? "
                    f"This may take several minutes.")
        if not prompt_yes_no(question, auto=args.auto):
            print(f"[SKIP] {ds['name']}: skipped by user.")
            results[ds["name"]] = "skipped"
            continue

        success = download_and_extract(ds["url"], dest_dir, ds["name"])
        if success:
            n = count_images(dest_dir)
            print(f"  Done. Found {n} images in {dest_dir}")
            results[ds["name"]] = "downloaded"
        else:
            results[ds["name"]] = "failed"

    print()
    print("=" * 60)
    print("Download Summary:")
    for name, status in results.items():
        icon = {"downloaded": "OK", "already_exists": "OK",
                "skipped": "--", "failed": "FAIL"}.get(status, "??")
        print(f"  [{icon}] {name}: {status}")

    print()
    print("Next steps:")
    print("  1. If downloads succeeded, run: jupyter lab notebooks/")
    print("     and open notebooks in order starting from 02.")
    print("  2. To run with mock data only: python scripts/generate_mock_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
