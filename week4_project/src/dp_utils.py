"""
src/dp_utils.py
===============
Differential privacy utilities using Opacus for FedTB-Nigeria.

Why this is a separate module:
  - DP setup involves several non-trivial steps (privacy accountant,
    gradient clipping, noise injection). Keeping it here avoids
    duplicating these in notebooks 09 and 10.
  - The concepts are fully explained step-by-step in notebook 09
    before this module is imported.

What is Differential Privacy (DP)?
-------------------------------------
  DP provides a mathematical guarantee: even if an adversary sees the
  model updates you send to the FL server, they cannot reliably infer
  whether any specific patient's data was in your training set.

  Formally, (ε, δ)-DP means:
    For any two datasets D and D' differing by one record, and any
    output O:
      P(M(D) ∈ O) ≤ e^ε × P(M(D') ∈ O) + δ

  Where:
    ε (epsilon) = privacy budget. Lower = stronger privacy guarantee.
                  ε = 1 is very private; ε = 8 is reasonable for medical imaging.
    δ (delta)   = failure probability. Typically set to 1/n where n = dataset size.

  In DP-SGD (Opacus):
    1. Clip each gradient to a maximum norm (reduces sensitivity)
    2. Add Gaussian noise to the clipped gradients
    3. The privacy accountant tracks the cumulative ε spent each step.

Usage
-----
  from src.dp_utils import make_private_model, get_privacy_spent, print_dp_summary
  private_model, private_optimizer, private_loader = make_private_model(
      model, optimizer, train_loader, target_epsilon=8.0, target_delta=1e-5,
      max_grad_norm=1.0, epochs=50,
  )
"""

from __future__ import annotations

import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from opacus import PrivacyEngine
    from opacus.validators import ModuleValidator
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    warnings.warn(
        "Opacus is not installed. DP training is not available.\n"
        "Install with: pip install opacus"
    )


# ──────────────────────────────────────────────
# Model compatibility check
# ──────────────────────────────────────────────

def validate_model_for_dp(model: nn.Module) -> nn.Module:
    """
    Check and fix model compatibility with Opacus DP-SGD.

    Opacus requires that all modules support per-sample gradients.
    Batch normalisation (BatchNorm) does NOT support this — we replace it
    with Group Normalisation, which does.

    Parameters
    ----------
    model : nn.Module
        The model to validate (ResNet-18).

    Returns
    -------
    nn.Module
        A DP-compatible version of the model.
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is not installed. Run: pip install opacus")

    errors = ModuleValidator.validate(model, strict=False)

    if errors:
        print(f"Found {len(errors)} compatibility issues. Auto-fixing...")
        model = ModuleValidator.fix(model)
        errors_after = ModuleValidator.validate(model, strict=False)
        if errors_after:
            raise ValueError(
                f"Could not auto-fix model for Opacus DP.\n"
                f"Remaining errors: {errors_after}"
            )
        print("  BatchNorm layers replaced with GroupNorm. Model is DP-compatible.")
    else:
        print("Model is already DP-compatible (no fixes needed).")

    return model


# ──────────────────────────────────────────────
# DP model setup
# ──────────────────────────────────────────────

def make_private_model(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    target_epsilon: float = 8.0,
    target_delta: float = 1e-5,
    max_grad_norm: float = 1.0,
    epochs: int = 50,
) -> tuple:
    """
    Wrap a model, optimizer, and DataLoader with Opacus DP-SGD.

    After calling this function:
    - The optimizer automatically clips and noises gradients each step.
    - The privacy accountant tracks cumulative ε.
    - You must use the RETURNED loader (not the original) for training.

    Parameters
    ----------
    model : nn.Module
        DP-validated model (run validate_model_for_dp first).
    optimizer : torch.optim.Optimizer
        Original optimizer (e.g., Adam).
    train_loader : DataLoader
        Training DataLoader for this client.
    target_epsilon : float
        Desired privacy budget ε at the end of training.
    target_delta : float
        Failure probability δ. Typically 1/n_training_samples.
    max_grad_norm : float
        Per-sample gradient clipping norm.
        Lower = stronger privacy but more noise = worse utility.
    epochs : int
        Total number of training epochs (needed to schedule noise).

    Returns
    -------
    (private_model, private_optimizer, private_loader)
        Use these in the training loop instead of the originals.
    """
    if not OPACUS_AVAILABLE:
        raise ImportError("Opacus is not installed. Run: pip install opacus")

    privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        epochs=epochs,
    )

    # Store the engine on the model for later epsilon queries
    private_model._privacy_engine = privacy_engine

    noise_multiplier = private_optimizer.noise_multiplier
    print(f"DP-SGD configured:")
    print(f"  Target ε = {target_epsilon} | Target δ = {target_delta}")
    print(f"  Max gradient norm (clipping) = {max_grad_norm}")
    print(f"  Noise multiplier (σ) = {noise_multiplier:.4f}")
    print(f"  Training epochs = {epochs}")
    print(f"  (Higher σ = more noise = stronger privacy, lower utility)")

    return private_model, private_optimizer, private_loader


# ──────────────────────────────────────────────
# Privacy accounting
# ──────────────────────────────────────────────

def get_privacy_spent(private_model: nn.Module) -> dict:
    """
    Query the current privacy budget spent (ε, δ) from a private model.

    Call this during or after training to see how much privacy budget
    has been consumed so far.

    Parameters
    ----------
    private_model : nn.Module
        A model returned by make_private_model().

    Returns
    -------
    dict with 'epsilon' and 'delta'
    """
    if not hasattr(private_model, "_privacy_engine"):
        raise AttributeError(
            "This model was not wrapped by make_private_model().\n"
            "The _privacy_engine attribute is missing."
        )

    engine  = private_model._privacy_engine
    epsilon = engine.get_epsilon(delta=engine.target_delta)

    return {
        "epsilon": float(epsilon),
        "delta":   float(engine.target_delta),
    }


def print_dp_summary(private_model: nn.Module) -> None:
    """
    Print a human-readable DP summary for the model.

    Parameters
    ----------
    private_model : nn.Module
    """
    privacy = get_privacy_spent(private_model)
    print("=" * 50)
    print("Differential Privacy Summary")
    print("=" * 50)
    print(f"  ε (epsilon) spent so far : {privacy['epsilon']:.4f}")
    print(f"  δ (delta)                : {privacy['delta']:.2e}")
    print(f"  Interpretation           : This model is ({privacy['epsilon']:.2f}, {privacy['delta']:.0e})-DP")
    print()
    print("  Recall: lower ε = stronger privacy guarantee.")
    print("  ε < 1  → very strong privacy (often too noisy for utility)")
    print("  ε ~ 8  → reasonable privacy for medical imaging benchmarks")
    print("=" * 50)
