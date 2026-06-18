#!/usr/bin/env python3
"""
scripts/run_all.py
───────────────────
Execute all project notebooks in order using nbconvert.

Usage:
    python scripts/run_all.py             # run all notebooks
    python scripts/run_all.py --nb 02 05  # run only notebooks 02 and 05
    python scripts/run_all.py --dry-run   # list notebooks without running

This script:
  1. Discovers notebooks/ in sorted order.
  2. Runs each with jupyter nbconvert --execute (in-place).
  3. Logs success / failure for each notebook.
  4. Stops on first failure unless --continue-on-error is set.

Expected runtime (with data downloaded):
  ~15-25 minutes total on a standard laptop (CPU only).
  Notebooks 09-13 (models, evaluation) are the slowest (~5-10 min each).

Requirements:
  pip install nbconvert jupyter
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def discover_notebooks(filter_ids: list[str] | None = None) -> list[Path]:
    """Return sorted list of .ipynb files, optionally filtered by ID prefix."""
    nbs = sorted(NOTEBOOKS_DIR.glob("*.ipynb"))
    if filter_ids:
        nbs = [nb for nb in nbs if any(nb.name.startswith(fid) for fid in filter_ids)]
    return nbs


def run_notebook(nb_path: Path, timeout: int = 3600) -> tuple[bool, float]:
    """
    Execute a notebook in-place with nbconvert.

    Returns
    -------
    (success: bool, elapsed_seconds: float)
    """
    start = time.time()
    cmd = [
        sys.executable, "-m", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--inplace",
        f"--ExecutePreprocessor.timeout={timeout}",
        "--ExecutePreprocessor.kernel_name=python3",
        str(nb_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"  STDERR: {result.stderr[-500:]}")
            return False, elapsed
        return True, elapsed
    except Exception as e:
        return False, time.time() - start


def main():
    parser = argparse.ArgumentParser(description="Run all project notebooks in order.")
    parser.add_argument("--nb", nargs="+", metavar="ID",
                        help="Run only notebooks whose name starts with these IDs "
                             "(e.g. --nb 02 05)")
    parser.add_argument("--dry-run", action="store_true",
                        help="List notebooks without executing them")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Do not stop on first notebook failure")
    parser.add_argument("--timeout", type=int, default=3600,
                        help="Per-notebook timeout in seconds (default: 3600)")
    args = parser.parse_args()

    notebooks = discover_notebooks(args.nb)

    if not notebooks:
        print("No notebooks found. Check that notebooks/ exists and contains .ipynb files.")
        sys.exit(1)

    print(f"Found {len(notebooks)} notebook(s):")
    for nb in notebooks:
        print(f"  {nb.name}")

    if args.dry_run:
        print("\n[Dry run] No notebooks were executed.")
        return

    print("\nStarting notebook execution...")
    print("=" * 60)

    results = []
    for nb_path in notebooks:
        print(f"\n→ Running: {nb_path.name}")
        success, elapsed = run_notebook(nb_path, timeout=args.timeout)
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}  ({elapsed:.1f}s)")
        results.append((nb_path.name, success, elapsed))

        if not success and not args.continue_on_error:
            print("\nStopping due to notebook failure.")
            print("Re-run with --continue-on-error to skip failures.")
            break

    # Summary
    print("\n" + "=" * 60)
    print("Execution summary:")
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_fail = len(results) - n_pass
    for name, ok, elapsed in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {name:50s} {elapsed:6.1f}s")
    print(f"\n  Passed: {n_pass}/{len(results)}   Failed: {n_fail}/{len(results)}")

    if n_fail > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
