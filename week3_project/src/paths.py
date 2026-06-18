"""
src/paths.py
────────────
Centralised path management.

Why this exists:
  Notebooks live inside notebooks/ but they reference files in data/, reports/,
  and models/.  Hard-coding relative paths like '../../data/raw/' is fragile.
  This module resolves all paths from the project root so they work regardless
  of where a script is called from.

Usage (in a notebook):
  from src.config import load_config
  from src.paths import Paths
  cfg = load_config()
  paths = Paths(cfg)
  orders = pd.read_csv(paths.raw / 'olist_orders_dataset.csv')
"""

from pathlib import Path
from src.config import load_config


class Paths:
    """
    Resolves and stores all important project paths.

    Attributes
    ----------
    root        : project root directory
    raw         : data/raw
    interim     : data/interim
    processed   : data/processed
    external    : data/external
    mock        : data/mock  (test fixtures only)
    models      : models/
    reports_figs: reports/figures
    reports_tabs: reports/tables
    paper_figs  : paper_or_report/figures
    paper_tabs  : paper_or_report/tables
    """

    def __init__(self, cfg: dict | None = None):
        if cfg is None:
            cfg = load_config()

        # Project root = parent of src/
        self.root = Path(__file__).resolve().parent.parent

        d = cfg["data"]
        self.raw       = self.root / d["raw_dir"]
        self.interim   = self.root / d["interim_dir"]
        self.processed = self.root / d["processed_dir"]
        self.external  = self.root / d["external_dir"]
        self.mock      = self.root / d["mock_dir"]

        self.models = self.root / cfg["models"]["save_dir"]

        self.reports_figs = self.root / cfg["figures"]["reports_dir"]
        self.reports_tabs = self.root / cfg["tables"]["reports_dir"]
        self.paper_figs   = self.root / cfg["figures"]["paper_dir"]
        self.paper_tabs   = self.root / cfg["tables"]["paper_dir"]

    def ensure_dirs(self):
        """Create all directories if they do not already exist."""
        for attr, path in self.__dict__.items():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)

    def olist_file(self, key: str, cfg: dict | None = None) -> Path:
        """
        Return the full path for an Olist CSV file by config key.

        Parameters
        ----------
        key : str
            Key in cfg['data']['olist_files'], e.g. 'orders', 'customers'.

        Returns
        -------
        Path
        """
        if cfg is None:
            cfg = load_config()
        filename = cfg["data"]["olist_files"][key]
        return self.raw / filename

    def __repr__(self):
        lines = [f"Paths(root={self.root})"]
        for attr, val in self.__dict__.items():
            if isinstance(val, Path):
                lines.append(f"  {attr}: {val}")
        return "\n".join(lines)
