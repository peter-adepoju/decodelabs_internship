"""
tests/test_model.py
===================
Unit tests for src/model.py — ResNet-18 builder, freeze/unfreeze,
parameter counting.

Run with:
  pytest tests/test_model.py -v
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model, freeze_backbone, unfreeze_backbone, count_parameters


class TestBuildModel:

    def test_model_builds_without_error(self):
        model = build_model(pretrained=False, num_classes=2)
        assert model is not None

    def test_output_shape_matches_num_classes(self):
        model = build_model(pretrained=False, num_classes=2)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 2), f"Expected (4, 2), got {out.shape}"

    def test_batch_size_one_works(self):
        model = build_model(pretrained=False, num_classes=2)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 2)

    def test_small_input_size_works(self):
        """Model should handle smaller input sizes too (not just 224)."""
        model = build_model(pretrained=False, num_classes=2)
        model.eval()
        x = torch.randn(2, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 2)

    def test_num_classes_two_is_default(self):
        model = build_model(pretrained=False)
        x = torch.randn(1, 3, 64, 64)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape[-1] == 2

    def test_pretrained_false_does_not_crash(self):
        """Building with pretrained=False must not raise (no network needed)."""
        model = build_model(pretrained=False, num_classes=2)
        assert model is not None


class TestFreezeUnfreeze:

    def test_freeze_reduces_trainable_params(self):
        model = build_model(pretrained=False)
        n_before = sum(1 for p in model.parameters() if p.requires_grad)
        freeze_backbone(model)
        n_after = sum(1 for p in model.parameters() if p.requires_grad)
        assert n_after < n_before, "freeze_backbone should reduce trainable params"

    def test_unfreeze_restores_all_params(self):
        model = build_model(pretrained=False)
        total = sum(1 for p in model.parameters())
        freeze_backbone(model)
        unfreeze_backbone(model)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        assert trainable == total, "unfreeze_backbone should restore all params"

    def test_frozen_head_still_trainable(self):
        """After freeze_backbone, the fc head must still have grad enabled."""
        model = build_model(pretrained=False)
        freeze_backbone(model)
        head_trainable = any(
            p.requires_grad
            for name, p in model.named_parameters()
            if "fc" in name
        )
        assert head_trainable, "FC head must remain trainable after backbone freeze"


class TestCountParameters:

    def test_returns_dict_with_required_keys(self):
        model = build_model(pretrained=False)
        counts = count_parameters(model)
        for key in ["trainable", "frozen", "total"]:
            assert key in counts

    def test_trainable_plus_frozen_equals_total(self):
        model = build_model(pretrained=False)
        counts = count_parameters(model)
        assert counts["trainable"] + counts["frozen"] == counts["total"]

    def test_total_is_positive(self):
        model = build_model(pretrained=False)
        counts = count_parameters(model)
        assert counts["total"] > 0

    def test_all_trainable_before_freeze(self):
        model = build_model(pretrained=False)
        counts = count_parameters(model)
        assert counts["frozen"] == 0

    def test_some_frozen_after_freeze_backbone(self):
        model = build_model(pretrained=False)
        freeze_backbone(model)
        counts = count_parameters(model)
        assert counts["frozen"] > 0
        assert counts["trainable"] > 0   # head still trainable
