"""
tests/test_config.py
─────────────────────
Tests for src/config.py and src/paths.py
"""

import pytest
from pathlib import Path

from src.config import load_config, get_random_seed
from src.paths import Paths


def test_load_config_returns_dict():
    cfg = load_config()
    assert isinstance(cfg, dict)

def test_config_has_required_keys():
    cfg = load_config()
    assert "project" in cfg
    assert "data" in cfg
    assert "modeling" in cfg
    assert "rfm" in cfg

def test_config_random_seed_is_int():
    cfg = load_config()
    assert isinstance(cfg["project"]["random_seed"], int)

def test_get_random_seed():
    seed = get_random_seed()
    assert seed == 42

def test_load_config_raises_on_bad_path():
    with pytest.raises(FileNotFoundError):
        load_config("nonexistent/path/config.yaml")

def test_paths_attributes_are_pathlib_paths():
    cfg = load_config()
    paths = Paths(cfg)
    for attr in ["raw", "interim", "processed", "models",
                 "reports_figs", "reports_tabs"]:
        val = getattr(paths, attr)
        assert isinstance(val, Path), f"{attr} is not a Path"

def test_paths_olist_file_returns_path():
    cfg = load_config()
    paths = Paths(cfg)
    p = paths.olist_file("orders", cfg)
    assert isinstance(p, Path)
    assert p.name == "olist_orders_dataset.csv"

def test_paths_olist_file_raises_on_unknown_key():
    cfg = load_config()
    paths = Paths(cfg)
    with pytest.raises(KeyError):
        paths.olist_file("nonexistent_key", cfg)
