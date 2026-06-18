"""
src/paths.py
============
Centralised path management for FedTB-Nigeria.

Why this is a separate module:
  - Hardcoding file paths in notebooks causes errors when notebooks move.
  - This module gives every notebook a reliable way to find data, models,
    and output directories, regardless of where the notebook is run from.

Usage:
  from src.paths import get_paths
  p = get_paths()
  print(p["figures"])   # → /absolute/path/to/reports/figures
"""

from pathlib import Path
from src.config import load_config


def get_paths(config_path: str | None = None) -> dict[str, Path]:
    """
    Return a dictionary of absolute Paths for all project directories.

    Parameters
    ----------
    config_path : str or None
        Optional path to a custom config.yaml.

    Returns
    -------
    dict[str, Path]
        Keys: descriptive names. Values: absolute Path objects.
        All directories are created if they do not exist.

    Example
    -------
    >>> from src.paths import get_paths
    >>> p = get_paths()
    >>> p["processed"]
    PosixPath('/home/.../data/processed')
    """
    cfg = load_config(config_path)
    root = Path(cfg["_project_root"])

    paths = {
        # Data directories
        "raw":          root / cfg["data"]["raw_dir"],
        "interim":      root / cfg["data"]["interim_dir"],
        "processed":    root / cfg["data"]["processed_dir"],
        "mock":         root / cfg["data"]["mock_dir"],

        # Model directories
        "models":       root / cfg["paths"]["models_dir"],
        "centralised_model_dir": root / "models" / "centralised",
        "federated_model_dir":   root / "models" / "federated",

        # Report directories
        "reports":      root / cfg["paths"]["reports_dir"],
        "figures":      root / cfg["paths"]["figures_dir"],
        "tables":       root / cfg["paths"]["tables_dir"],
        "paper":        root / cfg["paths"]["paper_dir"],
        "paper_figures": root / cfg["paths"]["paper_figures_dir"],
        "paper_tables":  root / cfg["paths"]["paper_tables_dir"],
    }

    # Create all directories if they don't exist yet
    for name, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)

    return paths
