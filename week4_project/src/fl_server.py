"""
src/fl_server.py
================
Flower server strategy configuration for FedTB-Nigeria.

Why this is a separate module:
  - The server strategy is used in notebook 08 and scripts/run_federated.py.
  - Flower's FedAvg strategy is highly configurable; keeping it here
    makes it easy to swap strategies (e.g., FedProx, FedNova).

FedAvg algorithm (McMahan et al., 2017)
-----------------------------------------
  Given N clients each with local dataset D_i:

  For each round r:
    1. Server selects a fraction of clients (all by default)
    2. Server sends global model w_r to selected clients
    3. Each client i computes: w_i = local_train(w_r, D_i)
    4. Server aggregates: w_{r+1} = Σ_i (|D_i| / |D|) * w_i
       (weighted average by number of samples)

  The key insight: gradients never leave the client.
  Only the weight update (delta) is shared with the server.

Usage
-----
  from src.fl_server import build_fedavg_strategy, get_eval_fn
  strategy = build_fedavg_strategy(cfg, test_loader, model, device)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics, NDArrays
from flwr.server.strategy import FedAvg


# ──────────────────────────────────────────────
# Aggregation metric callback
# ──────────────────────────────────────────────

def weighted_average_metrics(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate per-client metrics into a single weighted average.

    Flower calls this after each round to summarise client-reported metrics.
    We weight each client's metric by its number of samples.

    Parameters
    ----------
    metrics : list of (num_samples, metrics_dict)

    Returns
    -------
    dict of aggregated metric values
    """
    # Extract per-client (n_samples, metric_value) pairs
    total_samples = sum(n for n, _ in metrics)

    aggregated = {}
    # Take the keys from the first client's metrics
    if metrics:
        sample_keys = [k for k in metrics[0][1].keys() if k != "client_id"]
        for key in sample_keys:
            try:
                weighted_sum = sum(
                    n * float(m.get(key, 0.0)) for n, m in metrics
                )
                aggregated[key] = weighted_sum / total_samples if total_samples > 0 else 0.0
            except (TypeError, ValueError):
                pass  # Skip non-numeric metrics

    return aggregated


# ──────────────────────────────────────────────
# Server-side evaluation function
# ──────────────────────────────────────────────

def get_eval_fn(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> Callable:
    """
    Return a Flower-compatible evaluation function for server-side testing.

    Flower calls this function after aggregation each round to evaluate
    the global model on a centralised test set (when available).
    This gives us a single, consistent view of global model performance
    across all rounds — useful for plotting learning curves.

    Parameters
    ----------
    model : nn.Module
        A fresh model instance for server-side evaluation.
        IMPORTANT: This should be a separate model from the client models.
    test_loader : DataLoader
        Held-out test set (not used by any client for training).
    device : torch.device

    Returns
    -------
    callable
        Function with signature (server_round, parameters, config) → (loss, metrics)
    """
    def evaluate(
        server_round: int,
        parameters: NDArrays,
        config: dict,
    ) -> tuple[float, dict]:
        """Load global weights and evaluate on the test set."""
        # Load the aggregated weights into the evaluation model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)

        criterion = nn.CrossEntropyLoss()
        total_loss = 0.0
        all_probs  = []
        all_labels = []

        with torch.no_grad():
            for images, labels, _ in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss   = criterion(logits, labels)
                probs  = torch.softmax(logits, dim=1)[:, 1]

                total_loss += loss.item() * images.size(0)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        n         = len(all_labels)
        test_loss = total_loss / n if n > 0 else 0.0

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)

        auc = 0.0
        if len(np.unique(y_true)) == 2:
            from sklearn.metrics import roc_auc_score
            auc = float(roc_auc_score(y_true, y_prob))

        print(
            f"  [Server Round {server_round}] "
            f"Test Loss: {test_loss:.4f} | Test AUC: {auc:.4f}"
        )

        return test_loss, {"test_auc": auc, "test_loss": test_loss}

    return evaluate


# ──────────────────────────────────────────────
# Strategy builder
# ──────────────────────────────────────────────

def build_fedavg_strategy(
    cfg: dict,
    eval_model: nn.Module | None = None,
    test_loader: DataLoader | None = None,
    device: torch.device | None = None,
) -> FedAvg:
    """
    Build a configured FedAvg strategy for Flower.

    Parameters
    ----------
    cfg : dict
        Project config (from load_config()).
    eval_model : nn.Module or None
        Model for server-side evaluation. If None, no server eval is done.
    test_loader : DataLoader or None
        Test set for server-side evaluation.
    device : torch.device or None

    Returns
    -------
    flwr.server.strategy.FedAvg
    """
    fl_cfg = cfg["federated"]

    evaluate_fn = None
    if eval_model is not None and test_loader is not None and device is not None:
        evaluate_fn = get_eval_fn(eval_model, test_loader, device)

    strategy = FedAvg(
        fraction_fit         = fl_cfg["fraction_fit"],
        fraction_evaluate    = fl_cfg["fraction_fit"],  # Same fraction for eval
        min_fit_clients      = fl_cfg["min_fit_clients"],
        min_evaluate_clients = fl_cfg["min_evaluate_clients"],
        min_available_clients= fl_cfg["min_available_clients"],
        evaluate_fn          = evaluate_fn,             # Server-side eval
        fit_metrics_aggregation_fn     = weighted_average_metrics,
        evaluate_metrics_aggregation_fn= weighted_average_metrics,
    )

    return strategy
