"""
tests/test_dp_utils.py
======================
Unit tests for src/dp_utils.py — Opacus model validation,
DP privacy accounting, and make_private_model.

Note: These tests require Opacus to be installed.
If Opacus is not installed, all DP tests are skipped gracefully.

Run with:
  pytest tests/test_dp_utils.py -v
"""

import pytest
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import build_model
from src.data_utils import make_mock_dataset, TBDataset, build_transforms

# Check if Opacus is available
try:
    import opacus
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False

skip_if_no_opacus = pytest.mark.skipif(
    not OPACUS_AVAILABLE,
    reason="Opacus not installed — skipping DP tests. Run: pip install opacus"
)

from src.dp_utils import validate_model_for_dp


@pytest.fixture(scope="module")
def tiny_loader(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("dp_mock")
    df = make_mock_dataset(save_dir=save_dir, num_images=16, image_size=32, seed=7)
    t  = build_transforms(image_size=32, split="train")
    ds = TBDataset(df, transform=t)
    return DataLoader(ds, batch_size=4, shuffle=True)


class TestValidateModelForDP:

    @skip_if_no_opacus
    def test_validate_resnet_fixes_batchnorm(self):
        """
        ResNet-18 has BatchNorm layers which Opacus does not support.
        validate_model_for_dp should replace them with GroupNorm.
        """
        model = build_model(pretrained=False, num_classes=2)
        # Check original model has BatchNorm
        has_bn_before = any(
            isinstance(m, torch.nn.BatchNorm2d)
            for m in model.modules()
        )
        assert has_bn_before, "ResNet-18 should have BatchNorm before fixing"

        fixed_model = validate_model_for_dp(model)

        # After fixing, there should be no BatchNorm
        has_bn_after = any(
            isinstance(m, torch.nn.BatchNorm2d)
            for m in fixed_model.modules()
        )
        assert not has_bn_after, "validate_model_for_dp should remove BatchNorm"

    @skip_if_no_opacus
    def test_fixed_model_still_forward_passes(self):
        """Fixed model must still produce correct output shape."""
        model = build_model(pretrained=False, num_classes=2)
        fixed = validate_model_for_dp(model)
        fixed.eval()
        x = torch.randn(2, 3, 32, 32)
        with torch.no_grad():
            out = fixed(x)
        assert out.shape == (2, 2)


class TestMakePrivateModel:

    @skip_if_no_opacus
    def test_make_private_returns_three_objects(self, tiny_loader):
        from src.dp_utils import make_private_model
        model = build_model(pretrained=False, num_classes=2)
        model = validate_model_for_dp(model)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
        result = make_private_model(
            model=model, optimizer=opt, train_loader=tiny_loader,
            target_epsilon=8.0, target_delta=1e-5, max_grad_norm=1.0, epochs=1,
        )
        assert len(result) == 3

    @skip_if_no_opacus
    def test_private_model_has_privacy_engine_attribute(self, tiny_loader):
        from src.dp_utils import make_private_model
        model = build_model(pretrained=False, num_classes=2)
        model = validate_model_for_dp(model)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
        private_model, _, _ = make_private_model(
            model=model, optimizer=opt, train_loader=tiny_loader,
            target_epsilon=8.0, target_delta=1e-5, max_grad_norm=1.0, epochs=1,
        )
        assert hasattr(private_model, "_privacy_engine"), \
            "Private model must have _privacy_engine attribute"


class TestGetPrivacySpent:

    @skip_if_no_opacus
    def test_returns_epsilon_and_delta(self, tiny_loader):
        from src.dp_utils import make_private_model, get_privacy_spent
        model = build_model(pretrained=False, num_classes=2)
        model = validate_model_for_dp(model)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
        private_model, _, _ = make_private_model(
            model=model, optimizer=opt, train_loader=tiny_loader,
            target_epsilon=8.0, target_delta=1e-5, max_grad_norm=1.0, epochs=1,
        )
        spent = get_privacy_spent(private_model)
        assert "epsilon" in spent
        assert "delta"   in spent

    @skip_if_no_opacus
    def test_epsilon_positive(self, tiny_loader):
        from src.dp_utils import make_private_model, get_privacy_spent
        model = build_model(pretrained=False, num_classes=2)
        model = validate_model_for_dp(model)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-4)
        private_model, private_opt, private_loader = make_private_model(
            model=model, optimizer=opt, train_loader=tiny_loader,
            target_epsilon=8.0, target_delta=1e-5, max_grad_norm=1.0, epochs=1,
        )
        # Run one training step so accountant has something to query
        private_model.train()
        crit = torch.nn.CrossEntropyLoss()
        for images, labels, _ in private_loader:
            private_opt.zero_grad()
            logits = private_model(images)
            crit(logits, labels).backward()
            private_opt.step()
            break

        spent = get_privacy_spent(private_model)
        assert spent["epsilon"] > 0.0

    def test_get_privacy_spent_raises_without_engine(self):
        """Calling get_privacy_spent on a non-DP model should raise AttributeError."""
        from src.dp_utils import get_privacy_spent
        plain_model = build_model(pretrained=False, num_classes=2)
        with pytest.raises(AttributeError):
            get_privacy_spent(plain_model)
