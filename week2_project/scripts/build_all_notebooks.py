"""
build_all_notebooks.py — Master script that generates all 13 notebooks.
Run this once from the project root:
    python scripts/build_all_notebooks.py
"""
import subprocess, sys, os

SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)

parts = [
    "build_notebooks_part1.py",
    "build_notebooks_part2.py",
    "build_notebooks_part3.py",
    "build_notebooks_part4.py",
    "build_notebooks_part5.py",
]

print("=" * 60)
print("  Diabetes Readmission — Notebook Builder")
print("=" * 60)

for part in parts:
    script_path = os.path.join(SCRIPTS_DIR, part)
    print(f"\nRunning {part}...")
    result = subprocess.run([sys.executable, script_path],
                            capture_output=True, text=True)
    print(result.stdout)
    if result.returncode != 0:
        print("ERROR:", result.stderr[-600:])
        sys.exit(1)

# Validate
nb_dir = os.path.join(PROJECT_ROOT, "notebooks")
notebooks = sorted([f for f in os.listdir(nb_dir) if f.endswith(".ipynb")])
print(f"\n{'='*60}")
print(f"  All notebooks created: {len(notebooks)}")
print(f"{'='*60}")
for nb in notebooks:
    print(f"  ✓ {nb}")
