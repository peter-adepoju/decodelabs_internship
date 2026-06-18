"""
tests/test_fl_client.py
=======================
Unit tests for src/fl_client.py — Flower FL client methods
using mock data and random model weights.

Run with:
  pytest tests/test_fl_client.py -v
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_utils import make_mock_dataset, TBDataset, build_transforms
from src.model import build_model
from src.fl_client import TBClient, train_one_epoch, evaluate_model
from src.config import load_config


@pytest.fixture(scope="module")
def mock_data(tmp_path_factory):
    save_dir = tmp_path_factory.mktemp("fl_mock")
    df = make_mock_dataset(save_dir=save_dir, num_images=20, image_size=32, seed=0)
    return df


@pytest.fixture(scope="module")
def loaders(mock_data):
    t = build_transforms(image_size=32, split="train")
    v = build_transforms(image_size=32, split="val")
    train_ds = TBDataset(mock_data, transform=t)
    val_ds   = TBDataset(mock_data, transform=v)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=4, shuffle=False)
    return train_loader, val_loader


@pytest.fixture(scope="module")
def model_and_loaders(loaders):
    cfg   = load_config()
    model = build_model(pretrained=False, num_classes=2, dropout=0.0)
    train_loader, val_loader = loaders
    return model, train_loader, val_loader, cfg


class TestTrainOneEpoch:

    def test_returns_dict_with_loss_and_accuracy(self, model_and_loaders):
        model, train_loader, _, _ = model_and_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        result = train_one_epoch(model, train_loader, optimizer, criterion,
                                  torch.device("cpu"))
        assert "loss"     in result
        assert "accuracy" in result

    def test_loss_is_non_negative(self, model_and_loaders):
        model, train_loader, _, _ = model_and_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        result = train_one_epoch(model, train_loader, optimizer, criterion,
                                  torch.device("cpu"))
        assert result["loss"] >= 0.0

    def test_accuracy_in_unit_interval(self, model_and_loaders):
        model, train_loader, _, _ = model_and_loaders
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        result = train_one_epoch(model, train_loader, optimizer, criterion,
                                  torch.device("cpu"))
        assert 0.0 <= result["accuracy"] <= 1.0


class TestEvaluateModel:

    def test_returns_dict_with_required_keys(self, model_and_loaders):
        model, _, val_loader, _ = model_and_loaders
        criterion = nn.CrossEntropyLoss()
        result = evaluate_model(model, val_loader, criterion, torch.device("cpu"))
        for key in ["loss", "accuracy", "y_true", "y_prob"]:
            assert key in result

    def test_y_true_and_y_prob_same_length(self, model_and_loaders):
        model, _, val_loader, _ = model_and_loaders
        criterion = nn.CrossEntropyLoss()
        result = evaluate_model(model, val_loader, criterion, torch.device("cpu"))
        assert len(result["y_true"]) == len(result["y_prob"])

    def test_y_prob_in_unit_interval(self, model_and_loaders):
        model, _, val_loader, _ = model_and_loaders
        criterion = nn.CrossEntropyLoss()
        result = evaluate_model(model, val_loader, criterion, torch.device("cpu"))
        assert np.all(result["y_prob"] >= 0.0)
        assert np.all(result["y_prob"] <= 1.0)


class TestTBClient:

    def test_get_parameters_returns_list(self, model_and_loaders):
        model, train_loader, val_loader, cfg = model_and_loaders
        client = TBClient(
            model=build_model(pretrained=False, num_classes=2),
            train_loader=train_loader, val_loader=val_loader,
            device=torch.device("cpu"), config=cfg, client_id="test_site",
        )
        params = client.get_parameters(config={})
        assert isinstance(params, list)
        assert len(params) > 0
        assert all(isinstance(p, np.ndarray) for p in params)

    def test_fit_returns_tuple_of_three(self, model_and_loaders):
        model, train_loader, val_loader, cfg = model_and_loaders
        client = TBClient(
            model=build_model(pretrained=False, num_classes=2),
            train_loader=train_loader, val_loader=val_loader,
            device=torch.device("cpu"), config=cfg, client_id="test_site",
        )
        init_params = client.get_parameters(config={})
        result = client.fit(init_params, config={})
        assert len(result) == 3   # (params, n_samples, metrics)

    def test_fit_returns_same_number_of_param_arrays(self, model_and_loaders):
        model, train_loader, val_loader, cfg = model_and_loaders
        client = TBClient(
            model=build_model(pretrained=False, num_classes=2),
            train_loader=train_loader, val_loader=val_loader,
            device=torch.device("cpu"), config=cfg, client_id="test_site",
        )
        init_params = client.get_parameters(config={})
        updated_params, n_samples, metrics = client.fit(init_params, config={})
        assert len(updated_params) == len(init_params)

    def test_evaluate_returns_tuple_of_three(self, model_and_loaders):
        model, train_loader, val_loader, cfg = model_and_loaders
        client = TBClient(
            model=build_model(pretrained=False, num_classes=2),
            train_loader=train_loader, val_loader=val_loader,
            device=torch.device("cpu"), config=cfg, client_id="test_site",
        )
        init_params = client.get_parameters(config={})
        result = client.evaluate(init_params, config={})
        assert len(result) == 3   # (loss, n_samples, metrics)

    def test_set_parameters_updates_model_weights(self, model_and_loaders):
        """After set_parameters, model weights should equal the given arrays."""
        model, train_loader, val_loader, cfg = model_and_loaders
        client = TBClient(
            model=build_model(pretrained=False, num_classes=2),
            train_loader=train_loader, val_loader=val_loader,
            device=torch.device("cpu"), config=cfg, client_id="test_site",
        )
        # Get initial params, modify the first array, set, then verify
        init_params = client.get_parameters(config={})
        modified = [p.copy() for p in init_params]
        modified[0][:] = 999.0   # Set all values in first param array to 999

        client.set_parameters(modified)

        # Read back
        read_back = client.get_parameters(config={})
        assert np.allclose(read_back[0], 999.0), \
            "set_parameters did not update the model weights correctly"
