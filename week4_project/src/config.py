"""
src/config.py
=============
Load and validate the YAML configuration file.

Why this is a separate module:
  - Many notebooks need the same config values (paths, seeds, hyperparameters).
  - Rather than hard-coding values in each notebook, we load them from one
    central file: configs/config.yaml.
  - This means you only change one file to run a different experiment.

Usage (in any notebook):
  from src.config import load_config
  cfg = load_config()
  print(cfg["model"]["architecture"])   # → "resnet18"
"""

import yaml
from pathlib import Path


def load_config(config_path: str | None = None) -> dict:
    """
    Load the project configuration from YAML.

    Parameters
    ----------
    config_path : str or None
        Path to the YAML config file.
        If None, uses 'configs/config.yaml' relative to the project root.

    Returns
    -------
    dict
        Nested dictionary of configuration values.

    Example
    -------
    >>> cfg = load_config()
    >>> cfg["project"]["random_seed"]
    42
    """
    # Find the project root — walk up from this file's location
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent  # src/ → project root

    if config_path is None:
        config_path = project_root / "configs" / "config.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            "Make sure you are running from the project root directory."
        )

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Attach project root for path resolution
    cfg["_project_root"] = str(project_root)

    return cfg


def get_project_root() -> Path:
    """
    Return the absolute path to the project root directory.

    Returns
    -------
    Path
        Absolute path to the project root.
    """
    return Path(__file__).resolve().parent.parent
