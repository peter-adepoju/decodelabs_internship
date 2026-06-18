"""
tests/test_config.py
====================
Unit tests for src/config.py — the YAML configuration loader.

Why test this?
  Every notebook depends on load_config() returning the correct values.
  A broken config loader silently breaks all downstream experiments.

Run with:
  pytest tests/test_config.py -v
"""

import pytest
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config, get_project_root


class TestLoadConfig:

    def test_config_loads_successfully(self):
        """Config should load without raising any exception."""
        cfg = load_config()
        assert cfg is not None
        assert isinstance(cfg, dict)

    def test_config_has_required_top_level_keys(self):
        """Config must contain all required top-level sections."""
        cfg = load_config()
        required_keys = [
            "project", "data", "federated", "model",
            "training", "centralised", "differential_privacy",
            "splits", "augmentation", "evaluation", "paths",
        ]
        for key in required_keys:
            assert key in cfg, f"Missing top-level config key: '{key}'"

    def test_project_random_seed_is_int(self):
        """Random seed must be a non-negative integer."""
        cfg = load_config()
        seed = cfg["project"]["random_seed"]
        assert isinstance(seed, int), f"Expected int seed, got {type(seed)}"
        assert seed >= 0, f"Seed must be non-negative, got {seed}"

    def test_model_num_classes_is_two(self):
        """Binary classification: must have exactly 2 output classes."""
        cfg = load_config()
        assert cfg["model"]["num_classes"] == 2

    def test_federated_num_clients_positive(self):
        """Number of FL clients must be a positive integer."""
        cfg = load_config()
        n = cfg["federated"]["num_clients"]
        assert isinstance(n, int)
        assert n > 0, f"num_clients must be > 0, got {n}"

    def test_dp_epsilon_positive(self):
        """DP epsilon must be a positive float."""
        cfg = load_config()
        eps = cfg["differential_privacy"]["target_epsilon"]
        assert isinstance(eps, (int, float))
        assert eps > 0, f"epsilon must be > 0, got {eps}"

    def test_dp_delta_between_zero_and_one(self):
        """DP delta must be a small positive probability."""
        cfg = load_config()
        delta = cfg["differential_privacy"]["target_delta"]
        assert 0 < delta < 1, f"delta must be in (0, 1), got {delta}"

    def test_splits_sum_to_less_than_one(self):
        """Test + val split fractions must be < 1 (leaving room for training data)."""
        cfg = load_config()
        total = cfg["splits"]["test_size"] + cfg["splits"]["val_size"]
        assert total < 1.0, f"test_size + val_size = {total}, must be < 1.0"

    def test_image_size_positive(self):
        """Image size must be a positive integer."""
        cfg = load_config()
        sz = cfg["data"]["image_size"]
        assert isinstance(sz, int) and sz > 0

    def test_project_root_key_added(self):
        """load_config() must inject the '_project_root' key."""
        cfg = load_config()
        assert "_project_root" in cfg
        assert Path(cfg["_project_root"]).exists()

    def test_config_not_found_raises_file_not_found(self):
        """Requesting a non-existent config file must raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_config(config_path="/nonexistent/path/config.yaml")

    def test_get_project_root_returns_existing_path(self):
        """get_project_root() must return an existing directory."""
        root = get_project_root()
        assert root.exists(), f"Project root does not exist: {root}"
        assert root.is_dir()

    def test_site_names_length_matches_num_clients(self):
        """The site_names list must have exactly num_clients entries."""
        cfg = load_config()
        n      = cfg["federated"]["num_clients"]
        names  = cfg["federated"]["site_names"]
        assert len(names) == n, (
            f"site_names has {len(names)} entries but num_clients={n}"
        )

    def test_normalize_mean_and_std_are_length_three(self):
        """Normalisation mean and std must each have 3 channel values (RGB)."""
        cfg = load_config()
        mean = cfg["augmentation"]["normalize_mean"]
        std  = cfg["augmentation"]["normalize_std"]
        assert len(mean) == 3, f"normalize_mean must have 3 values, got {len(mean)}"
        assert len(std)  == 3, f"normalize_std must have 3 values, got {len(std)}"
