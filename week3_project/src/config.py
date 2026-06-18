"""
src/config.py
─────────────
Loads the project configuration from configs/config.yaml.

Why this exists:
  Every notebook needs the same set of paths, parameters, and settings.
  Instead of copy-pasting config values, we centralise them here.

Usage (in a notebook):
  import sys; sys.path.insert(0, '..')
  from src.config import load_config
  cfg = load_config()
  seed = cfg['project']['random_seed']
"""

from pathlib import Path
import yaml


def load_config(config_path: str | Path | None = None) -> dict:
    """
    Load the project config from configs/config.yaml.

    Parameters
    ----------
    config_path : str or Path, optional
        Path to the YAML file.  Defaults to <project_root>/configs/config.yaml,
        resolved relative to this file's location.

    Returns
    -------
    dict
        Nested dictionary of all config values.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist at the resolved path.
    """
    if config_path is None:
        # src/ is one level below the project root
        project_root = Path(__file__).resolve().parent.parent
        config_path = project_root / "configs" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found at: {config_path}\n"
            "Make sure you are running from the project root or pass an explicit path."
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    return cfg


def get_random_seed(cfg: dict | None = None) -> int:
    """Convenience wrapper to get the project random seed."""
    if cfg is None:
        cfg = load_config()
    return cfg["project"]["random_seed"]
