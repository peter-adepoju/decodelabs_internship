#!/usr/bin/env python3
"""
scripts/run_all.py
==================
End-to-end pipeline: mock data → centralised → federated → evaluation.

This script orchestrates the full FedTB-Nigeria pipeline in sequence.
It is equivalent to running all notebooks 00–16 in order, but without
the interactive explanations.

Usage:
  python scripts/run_all.py                 # Full run (requires GPU + real data)
  python scripts/run_all.py --mock          # Fast mock run on CPU (~5 min)
  python scripts/run_all.py --mock --quick  # Minimal runs (2 epochs, 2 rounds)
  python scripts/run_all.py --skip-fl       # Only centralised baseline
  python scripts/run_all.py --skip-dp       # FL but no DP

Outputs:
  All model checkpoints, metric tables, and figures from the full pipeline.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_step(description: str, cmd: list[str]) -> bool:
    """Run a subprocess command and report success/failure."""
    print()
    print(f"{'='*55}")
    print(f"STEP: {description}")
    print(f"{'='*55}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(project_root))
    elapsed = time.time() - start
    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"  [{status}] {description} ({elapsed:.1f}s)")
    return result.returncode == 0


def parse_args():
    p = argparse.ArgumentParser(description="Run full FedTB-Nigeria pipeline")
    p.add_argument("--mock",     action="store_true", help="Use mock data")
    p.add_argument("--quick",    action="store_true", help="Minimal epochs/rounds")
    p.add_argument("--skip-fl",  action="store_true", dest="skip_fl",
                   help="Skip federated training")
    p.add_argument("--skip-dp",  action="store_true", dest="skip_dp",
                   help="Skip DP federated training")
    return p.parse_args()


def main():
    args = parse_args()
    python = sys.executable   # Use the same Python that launched this script

    print("=" * 55)
    print("FedTB-Nigeria — Full Pipeline")
    print("=" * 55)
    print(f"Mock mode  : {args.mock}")
    print(f"Quick mode : {args.quick}")
    print(f"Skip FL    : {args.skip_fl}")
    print(f"Skip DP    : {args.skip_dp}")

    results = {}
    start_total = time.time()

    # ── Step 1: Generate mock data ───────────────────────────────────
    if args.mock:
        ok = run_step(
            "Generate mock data",
            [python, "scripts/generate_mock_data.py"],
        )
        results["mock_data"] = ok

    # ── Step 2: Centralised baseline ────────────────────────────────
    central_cmd = [python, "scripts/run_centralised.py"]
    if args.mock:
        central_cmd.append("--mock")
    if args.quick:
        central_cmd += ["--epochs", "3"]

    ok = run_step("Centralised baseline training", central_cmd)
    results["centralised"] = ok

    # ── Step 3: Federated (no DP) ────────────────────────────────────
    if not args.skip_fl:
        fl_cmd = [python, "scripts/run_federated.py", "--no-dp"]
        if args.mock:
            fl_cmd.append("--mock")
        if args.quick:
            fl_cmd += ["--rounds", "2"]
        ok = run_step("Federated training (no DP)", fl_cmd)
        results["fl_no_dp"] = ok

    # ── Step 4: Federated + DP ───────────────────────────────────────
    if not args.skip_fl and not args.skip_dp:
        fl_dp_cmd = [python, "scripts/run_federated.py"]
        if args.mock:
            fl_dp_cmd.append("--mock")
        if args.quick:
            fl_dp_cmd += ["--rounds", "2"]
        ok = run_step("Federated training (with DP)", fl_dp_cmd)
        results["fl_dp"] = ok

    # ── Final summary ────────────────────────────────────────────────
    total_time = time.time() - start_total
    print()
    print("=" * 55)
    print(f"Pipeline complete in {total_time:.1f}s")
    print("=" * 55)
    print("Step results:")
    all_ok = True
    for step, ok in results.items():
        icon = "OK  " if ok else "FAIL"
        print(f"  [{icon}] {step}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("All steps completed successfully.")
        print("Next: open notebooks/11_model_evaluation_and_statistical_comparison.ipynb")
        print("to compute bootstrap CIs and statistical tests on the saved predictions.")
    else:
        print("Some steps failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
