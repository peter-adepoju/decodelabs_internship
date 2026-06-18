"""
src/fl_client.py
================
Flower federated learning client definition for FedTB-Nigeria.

Why this is a separate module:
  - The Flower client class is ~150 lines. Embedding it in the notebook
    would obscure the FL logic in notebook 08.
  - It is reused in notebooks 08, 09, 10, and scripts/run_federated.py.
  - The class is fully explained step-by-step in notebook 08 before it
    is imported here.

How Flower FL works (brief summary)
-------------------------------------
  1. The Flower SERVER sends the current global model weights to clients.
  2. Each CLIENT loads those weights into its local model.
  3. Each CLIENT runs local training for N epochs on its own data.
  4. Each CLIENT sends updated weights back to the server.
  5. The SERVER aggregates (averages) the weights → new global model.
  6. Repeat for R rounds.

This client class handles steps 2–4 for one simulated hospital site.

Usage (in notebook 08)
-----------------------
  from src.fl_client import TBClient, train_one_epoch, evaluate
  client = TBClient(model, train_loader, val_loader, device, cfg)
  fl.client.start_numpy_client(server_address="localhost:8080", client=client)
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Flower imports
import flwr as fl
from flwr.common import NDArrays, Scalar


# ──────────────────────────────────────────────
# Per-epoch training and evaluation
# ──────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Run one epoch of local training on a client's data.

    Parameters
    ----------
    model : nn.Module
        The local model (ResNet-18).
    loader : DataLoader
        Training data for this client/site.
    optimizer : torch.optim.Optimizer
    criterion : nn.Module
        Loss function (CrossEntropyLoss with class weights).
    device : torch.device

    Returns
    -------
    dict with 'loss' and 'accuracy' for this epoch.
    """
    model.train()
    model.to(device)

    total_loss   = 0.0
    correct      = 0
    total        = 0

    for images, labels, _ in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images)
        loss   = criterion(logits, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item() * images.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += images.size(0)

    return {
        "loss":     total_loss / total if total > 0 else 0.0,
        "accuracy": correct    / total if total > 0 else 0.0,
    }


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate the model on a given DataLoader (val or test).

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    criterion : nn.Module
    device : torch.device

    Returns
    -------
    dict with 'loss', 'accuracy', 'y_true', 'y_prob'
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    correct    = 0
    total      = 0
    all_probs  = []
    all_labels = []

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            # Probabilities for class 1 (TB positive)
            probs  = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * images.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return {
        "loss":     total_loss / total if total > 0 else 0.0,
        "accuracy": correct    / total if total > 0 else 0.0,
        "y_true":   np.array(all_labels),
        "y_prob":   np.array(all_probs),
    }


# ──────────────────────────────────────────────
# Flower client
# ──────────────────────────────────────────────

class TBClient(fl.client.NumPyClient):
    """
    Flower client for federated TB classification at one hospital site.

    Inherits from fl.client.NumPyClient, which communicates with the
    Flower server using NumPy arrays (simple and framework-agnostic).

    The three required methods are:
      - get_parameters : return current model weights as numpy arrays
      - fit            : receive global weights, train locally, return updated weights
      - evaluate       : receive global weights, evaluate locally, return metrics

    Parameters
    ----------
    model : nn.Module
        ResNet-18 model (freshly built for this client).
    train_loader : DataLoader
        Training data for this client's hospital site.
    val_loader : DataLoader
        Validation data for this client.
    device : torch.device
    config : dict
        Project config (from load_config()).
    client_id : str
        Identifier for this client (e.g., "Lagos_University_Teaching_Hospital").
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict,
        client_id: str = "client_0",
    ):
        self.model        = model
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.config       = config
        self.client_id    = client_id

        # Build optimizer and loss function
        train_cfg    = config["training"]
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=train_cfg["learning_rate"],
            weight_decay=train_cfg["weight_decay"],
        )
        # CrossEntropy loss — class weights set from config
        self.criterion = nn.CrossEntropyLoss()

    # ── Required by Flower ──────────────────────────────────────────

    def get_parameters(self, config: dict) -> NDArrays:
        """
        Return the current local model weights as a list of NumPy arrays.

        The Flower server calls this at the start of each round to get
        the model state before training (for initialisation or aggregation).
        """
        return [
            val.cpu().numpy()
            for _, val in self.model.state_dict().items()
        ]

    def set_parameters(self, parameters: NDArrays) -> None:
        """
        Load a list of NumPy arrays into the local model as its new weights.

        The server calls this (via fit/evaluate) to distribute the global
        aggregated model to each client before local training.
        """
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict(
            {k: torch.tensor(v) for k, v in params_dict}
        )
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> Tuple[NDArrays, int, dict]:
        """
        Load global weights, train locally for N epochs, return updated weights.

        Parameters
        ----------
        parameters : list of np.ndarray
            Global model weights from the server.
        config : dict
            Server-provided round config (e.g., {'current_round': 3}).

        Returns
        -------
        (updated_parameters, num_samples, metrics)
        """
        # Step 1: Load the global model into our local model
        self.set_parameters(parameters)

        # Step 2: Run local training for N epochs
        n_epochs = self.config["training"]["epochs_per_round"]
        last_metrics = {}

        for epoch in range(n_epochs):
            last_metrics = train_one_epoch(
                self.model, self.train_loader,
                self.optimizer, self.criterion, self.device,
            )

        # Step 3: Return updated weights + number of training samples
        n_samples = len(self.train_loader.dataset)

        return (
            self.get_parameters(config={}),
            n_samples,
            {
                "train_loss":     last_metrics["loss"],
                "train_accuracy": last_metrics["accuracy"],
                "client_id":      self.client_id,
            },
        )

    def evaluate(
        self,
        parameters: NDArrays,
        config: dict,
    ) -> Tuple[float, int, dict]:
        """
        Load global weights and evaluate locally on validation data.

        Parameters
        ----------
        parameters : list of np.ndarray
            Global model weights from the server.
        config : dict

        Returns
        -------
        (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)

        results = evaluate_model(
            self.model, self.val_loader,
            self.criterion, self.device,
        )

        n_samples = len(self.val_loader.dataset)

        # Compute AUC if both classes are present
        from sklearn.metrics import roc_auc_score
        auc = 0.0
        if len(np.unique(results["y_true"])) == 2:
            auc = float(roc_auc_score(results["y_true"], results["y_prob"]))

        return (
            results["loss"],
            n_samples,
            {
                "val_accuracy": results["accuracy"],
                "val_auc":      auc,
                "client_id":    self.client_id,
            },
        )
