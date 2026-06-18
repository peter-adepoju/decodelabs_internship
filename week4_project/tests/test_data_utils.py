"""
tests/test_data_utils.py
========================
Unit tests for src/data_utils.py — dataset class, transforms,
Dirichlet partitioning, and mock data generation.

All tests use the synthetic mock dataset (tiny images) so they
run in seconds on CPU without any real data download.

Run with:
  pytest tests/test_data_utils.py -v
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys, tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import (
    TBDataset,
    build_transforms,
    dirichlet_split,
    uniform_split,
    get_class_weights,
    make_mock_dataset,
)


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_df(tmp_path_factory):
    """Generate a small mock dataset once for all tests in this module."""
    save_dir = tmp_path_factory.mktemp("mock")
    df = make_mock_dataset(save_dir=save_dir, num_images=30, image_size=32, seed=42)
    return df


@pytest.fixture(scope="module")
def val_transform():
    return build_transforms(image_size=32, split="val")


# ──────────────────────────────────────────────────────────────────────
# make_mock_dataset
# ──────────────────────────────────────────────────────────────────────

class TestMakeMockDataset:

    def test_returns_dataframe(self, mock_df):
        assert isinstance(mock_df, pd.DataFrame)

    def test_correct_number_of_images(self, mock_df):
        assert len(mock_df) == 30

    def test_required_columns_present(self, mock_df):
        for col in ["image_path", "label", "site"]:
            assert col in mock_df.columns, f"Missing column: {col}"

    def test_all_labels_binary(self, mock_df):
        assert set(mock_df["label"].unique()).issubset({0, 1})

    def test_all_files_exist(self, mock_df):
        for path in mock_df["image_path"]:
            assert Path(path).exists(), f"Image file not found: {path}"


# ──────────────────────────────────────────────────────────────────────
# TBDataset
# ──────────────────────────────────────────────────────────────────────

class TestTBDataset:

    def test_length_matches_dataframe(self, mock_df, val_transform):
        ds = TBDataset(mock_df, transform=val_transform)
        assert len(ds) == len(mock_df)

    def test_item_shapes(self, mock_df, val_transform):
        ds    = TBDataset(mock_df, transform=val_transform)
        img, label, path = ds[0]
        assert isinstance(img, torch.Tensor)
        assert img.shape == (3, 32, 32), f"Expected (3,32,32), got {img.shape}"
        assert label in {0, 1}
        assert isinstance(path, str)

    def test_label_matches_dataframe(self, mock_df, val_transform):
        ds = TBDataset(mock_df, transform=val_transform)
        for i in range(len(ds)):
            _, label, _ = ds[i]
            assert label == int(mock_df.iloc[i]["label"])

    def test_without_transform_raises_on_raw_pil(self, mock_df):
        # Without transform, __getitem__ returns a PIL Image (not Tensor)
        # TBDataset should still return without error
        ds = TBDataset(mock_df, transform=None)
        img, label, path = ds[0]
        # PIL Image or raw array — should not raise
        assert label in {0, 1}

    def test_missing_file_raises_file_not_found(self, tmp_path, val_transform):
        bad_df = pd.DataFrame([{"image_path": "/nonexistent/img.png", "label": 1}])
        ds = TBDataset(bad_df, transform=val_transform)
        with pytest.raises(FileNotFoundError):
            _ = ds[0]


# ──────────────────────────────────────────────────────────────────────
# build_transforms
# ──────────────────────────────────────────────────────────────────────

class TestBuildTransforms:

    def test_val_transform_is_deterministic(self, mock_df):
        """Applying val transform twice to same image gives identical result."""
        from PIL import Image
        t = build_transforms(image_size=32, split="val")
        img_pil = Image.open(mock_df["image_path"].iloc[0]).convert("RGB")
        t1 = t(img_pil)
        t2 = t(img_pil)
        assert torch.allclose(t1, t2), "Val transform must be deterministic"

    def test_output_size(self, mock_df):
        from PIL import Image
        t = build_transforms(image_size=64, split="val")
        img_pil = Image.open(mock_df["image_path"].iloc[0]).convert("RGB")
        out = t(img_pil)
        assert out.shape == (3, 64, 64), f"Expected (3,64,64), got {out.shape}"

    def test_train_transform_output_size(self, mock_df):
        from PIL import Image
        t = build_transforms(image_size=32, split="train")
        img_pil = Image.open(mock_df["image_path"].iloc[0]).convert("RGB")
        out = t(img_pil)
        assert out.shape == (3, 32, 32)

    def test_invalid_split_falls_back_to_val(self, mock_df):
        """An unrecognised split name should still return a valid transform."""
        from PIL import Image
        t = build_transforms(image_size=32, split="unknown_split")
        img_pil = Image.open(mock_df["image_path"].iloc[0]).convert("RGB")
        out = t(img_pil)
        assert out.shape == (3, 32, 32)


# ──────────────────────────────────────────────────────────────────────
# dirichlet_split
# ──────────────────────────────────────────────────────────────────────

class TestDirichletSplit:

    def setup_method(self):
        rng = np.random.default_rng(0)
        # 100 samples, roughly balanced
        self.labels = rng.choice([0, 1], size=100).tolist()

    def test_correct_number_of_clients(self):
        splits = dirichlet_split(self.labels, num_clients=4, alpha=0.5,
                                  min_samples=2, seed=42)
        assert len(splits) == 4

    def test_all_indices_covered(self):
        splits = dirichlet_split(self.labels, num_clients=3, alpha=1.0, seed=42)
        all_idx = sorted([i for s in splits for i in s])
        assert all_idx == list(range(len(self.labels)))

    def test_no_overlap_between_clients(self):
        splits = dirichlet_split(self.labels, num_clients=3, alpha=0.5, seed=42)
        seen = set()
        for s in splits:
            for idx in s:
                assert idx not in seen, f"Index {idx} appears in multiple clients"
                seen.add(idx)

    def test_min_samples_respected(self):
        splits = dirichlet_split(self.labels, num_clients=3, alpha=0.5,
                                  min_samples=5, seed=42)
        for i, s in enumerate(splits):
            assert len(s) >= 5, f"Client {i} has only {len(s)} samples (min=5)"

    def test_raises_if_min_samples_impossible(self):
        """If min_samples is too large, should raise ValueError."""
        with pytest.raises(ValueError):
            dirichlet_split([0, 1] * 5, num_clients=3, alpha=0.01,
                             min_samples=100, seed=42)


# ──────────────────────────────────────────────────────────────────────
# get_class_weights
# ──────────────────────────────────────────────────────────────────────

class TestGetClassWeights:

    def test_returns_tensor(self):
        w = get_class_weights([0, 0, 0, 1])
        assert isinstance(w, torch.Tensor)

    def test_minority_class_has_higher_weight(self):
        """Class 1 (1 sample) should have higher weight than class 0 (3 samples)."""
        w = get_class_weights([0, 0, 0, 1])
        assert w[1] > w[0], f"Expected w[1] > w[0], got {w}"

    def test_balanced_labels_give_equal_weights(self):
        w = get_class_weights([0, 1, 0, 1])
        assert abs(float(w[0]) - float(w[1])) < 1e-6

    def test_length_matches_num_classes(self):
        w = get_class_weights([0, 1])
        assert len(w) == 2
