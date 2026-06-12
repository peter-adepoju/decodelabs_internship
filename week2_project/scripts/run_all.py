"""
run_all.py — Execute all 13 notebooks in order using nbconvert.

Usage:
    python scripts/run_all.py               # run all notebooks
    python scripts/run_all.py --from 04     # start from notebook 04
    python scripts/run_all.py --only 09     # run only notebook 09

Requirements:
    pip install nbconvert

Each notebook is executed in-place (outputs saved back into the .ipynb file).
Execution stops at the first error so you can diagnose the problem.

Expected total runtime: 60–120 minutes depending on hardware.
SHAP computation (NB11) is the slowest step (~15–30 minutes on CPU).
"""

import subprocess
import sys
import os
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NB_DIR = os.path.join(PROJECT_ROOT, "notebooks")

ALL_NOTEBOOKS = [
    "00_project_overview.ipynb",
    "01_data_collection.ipynb",
    "02_data_cleaning.ipynb",
    "03_feature_engineering.ipynb",
    "04_eda_univariate.ipynb",
    "05_eda_bivariate.ipynb",
    "06_leakage_and_splits.ipynb",
    "07_baseline_models.ipynb",
    "08_advanced_models.ipynb",
    "09_model_evaluation.ipynb",
    "10_visualizations.ipynb",
    "11_error_analysis_shap.ipynb",
    "12_fairness_analysis.ipynb",
]

RUNTIMES = {
    "00": "< 1 min",
    "01": "1–3 min (includes download)",
    "02": "1–2 min",
    "03": "1–2 min",
    "04": "2–4 min",
    "05": "2–4 min",
    "06": "3–5 min",
    "07": "3–8 min",
    "08": "10–25 min",
    "09": "5–15 min",
    "10": "5–10 min",
    "11": "15–35 min (SHAP is slow on CPU)",
    "12": "5–10 min",
}


def run_notebook(nb_path, timeout=1800):
    """
    Execute a single notebook using nbconvert.

    Args:
        nb_path (str): full path to the .ipynb file.
        timeout (int): max seconds before giving up (default 30 min).

    Returns:
        bool: True if succeeded, False if failed.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--inplace",
            f"--ExecutePreprocessor.timeout={timeout}",
            "--ExecutePreprocessor.kernel_name=python3",
            nb_path,
        ],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, result.stderr


def main():
    parser = argparse.ArgumentParser(
        description="Run all Diabetes Readmission notebooks in order."
    )
    parser.add_argument("--from", dest="start_nb", default=None,
                        help="Start from this notebook number, e.g. --from 04")
    parser.add_argument("--only", dest="only_nb", default=None,
                        help="Run only one notebook, e.g. --only 09")
    args = parser.parse_args()

    # Filter notebook list
    if args.only_nb:
        notebooks = [n for n in ALL_NOTEBOOKS if n.startswith(args.only_nb)]
    elif args.start_nb:
        notebooks = [n for n in ALL_NOTEBOOKS if n[:2] >= args.start_nb]
    else:
        notebooks = ALL_NOTEBOOKS

    print("=" * 65)
    print("  Diabetes Readmission — Full Notebook Execution")
    print("=" * 65)
    print(f"  Notebooks to run : {len(notebooks)}")
    print(f"  Project root     : {PROJECT_ROOT}")
    print()

    total_start = time.time()
    completed = []
    failed = []

    for nb_name in notebooks:
        nb_path  = os.path.join(NB_DIR, nb_name)
        nb_id    = nb_name[:2]
        est_time = RUNTIMES.get(nb_id, "?")

        if not os.path.exists(nb_path):
            print(f"  ⚠  Not found: {nb_name}  (run build_all_notebooks.py first)")
            continue

        print(f"  ▶  Running {nb_name}  (estimated: {est_time}) ...")
        t0 = time.time()

        success, stderr = run_notebook(nb_path)
        elapsed = time.time() - t0

        if success:
            print(f"     ✓  Done in {elapsed:.0f}s")
            completed.append(nb_name)
        else:
            print(f"     ✗  FAILED after {elapsed:.0f}s")
            print("     Last error output:")
            # Show last 500 chars of stderr
            print("     " + stderr[-500:].replace("\n", "\n     "))
            failed.append(nb_name)
            print()
            print("  Stopping — fix the error above, then re-run with:")
            print(f"    python scripts/run_all.py --from {nb_id}")
            sys.exit(1)

    total_elapsed = time.time() - total_start
    print()
    print("=" * 65)
    print(f"  Completed: {len(completed)} / {len(notebooks)} notebooks")
    print(f"  Total time: {total_elapsed/60:.1f} minutes")
    print("=" * 65)
    print()
    print(f"  Figures saved to : {os.path.join(PROJECT_ROOT, 'reports', 'figures')}")
    print(f"  Tables  saved to : {os.path.join(PROJECT_ROOT, 'reports', 'tables')}")
    print()
    print("  Next steps:")
    print("  1. Review reports/figures/ for all saved plots")
    print("  2. Fill in results placeholders in paper_or_report/report.md")
    print("  3. Push to GitHub!")


if __name__ == "__main__":
    main()
