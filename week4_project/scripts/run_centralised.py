#!/usr/bin/env python3
"""
scripts/run_centralised.py
==========================
Train the centralised ResNet-18 baseline model end-to-end.

This script replicates the full training workflow from Notebook 07
as a runnable command-line script. Use it for reproducible re-runs
without re-opening the notebook.

Usage:
  python scripts/run_centralised.py
  python scripts/run_centralised.py --mock     # Fast run on mock data
  python scripts/run_centralised.py --epochs 30

Outputs:
  models/centralised/best_model.pth
  models/centralised/training_history.csv
  models/centralised/test_y_true.npy
  models/centralised/test_y_prob.npy
  reports/figures/centralised_training_curves.png
"""

import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from src.config import load_config
from src.paths import get_paths
from src.data_utils import TBDataset, build_transforms, get_class_weights, make_mock_dataset
from src.model import build_model, freeze_backbone, unfreeze_backbone, count_parameters
from src.metrics import compute_metrics
from src.visualization import set_publication_style, save_figure


def parse_args():
    parser = argparse.ArgumentParser(description="Train centralised ResNet-18 baseline")
    parser.add_argument("--mock",   action="store_true", help="Use mock data (fast, for testing)")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--seed",   type=int, default=None, help="Override random seed")
    return parser.parse_args()


def main():
    args  = parse_args()
    cfg   = load_config()
    paths = get_paths()

    SEED         = args.seed or cfg["project"]["random_seed"]
    MAX_EPOCHS   = args.epochs or (5 if args.mock else cfg["centralised"]["epochs"])
    BATCH_SIZE   = cfg["centralised"]["batch_size"]
    LR           = cfg["centralised"]["learning_rate"]
    WEIGHT_DECAY = cfg["centralised"]["weight_decay"]
    PATIENCE     = 3 if args.mock else cfg["centralised"]["early_stopping_patience"]
    FREEZE_EPOCHS = cfg["model"]["freeze_backbone_epochs"]
    IMAGE_SIZE   = cfg["data"]["image_size"]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("FedTB-Nigeria — Centralised Baseline Training")
    print("=" * 55)
    print(f"Device      : {DEVICE}")
    print(f"Mock mode   : {args.mock}")
    print(f"Max epochs  : {MAX_EPOCHS}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Seed        : {SEED}")
    print()

    # ── Generate mock data if needed ────────────────────────────────
    if args.mock:
        print("Generating mock data...")
        from sklearn.model_selection import train_test_split
        from src.data_utils import dirichlet_split

        mock_df = make_mock_dataset(paths["mock"], num_images=60, image_size=64, seed=SEED)
        mock_df.to_csv(paths["interim"] / "manifest_clean.csv", index=False)

        train_df, temp_df = train_test_split(mock_df, test_size=0.30,
                                              stratify=mock_df["label"], random_state=SEED)
        val_df, test_df = train_test_split(temp_df, test_size=0.50,
                                            stratify=temp_df["label"], random_state=SEED)
        train_df.to_csv(paths["processed"] / "train.csv", index=False)
        val_df.to_csv(  paths["processed"] / "val.csv",   index=False)
        test_df.to_csv( paths["processed"] / "test.csv",  index=False)
        print(f"  Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # ── Load data ────────────────────────────────────────────────────
    train_df = pd.read_csv(paths["processed"] / "train.csv")
    val_df   = pd.read_csv(paths["processed"] / "val.csv")
    test_df  = pd.read_csv(paths["processed"] / "test.csv")

    train_t = build_transforms(IMAGE_SIZE, "train",
                                cfg["augmentation"]["normalize_mean"],
                                cfg["augmentation"]["normalize_std"])
    val_t   = build_transforms(IMAGE_SIZE, "val",
                                cfg["augmentation"]["normalize_mean"],
                                cfg["augmentation"]["normalize_std"])

    train_loader = DataLoader(TBDataset(train_df, train_t), batch_size=BATCH_SIZE,
                               shuffle=True, num_workers=0)
    val_loader   = DataLoader(TBDataset(val_df,   val_t),   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)
    test_loader  = DataLoader(TBDataset(test_df,  val_t),   batch_size=BATCH_SIZE,
                               shuffle=False, num_workers=0)

    # ── Build model ──────────────────────────────────────────────────
    model = build_model(pretrained=True,
                         num_classes=cfg["model"]["num_classes"],
                         dropout=cfg["model"]["dropout"]).to(DEVICE)
    freeze_backbone(model)

    param_info = count_parameters(model)
    print(f"Model: ResNet-18 | Params: {param_info['total']:,} total, "
          f"{param_info['trainable']:,} trainable (backbone frozen)")
    print()

    # ── Loss + optimiser ─────────────────────────────────────────────
    class_weights = get_class_weights(train_df["label"].values).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)

    # ── Training loop ────────────────────────────────────────────────
    model_dir = paths["centralised_model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / "best_model.pth"

    history          = {"epoch": [], "train_loss": [], "val_loss": [], "val_auc": []}
    best_val_loss    = float("inf")
    patience_counter = 0

    print("Training...")
    for epoch in range(1, MAX_EPOCHS + 1):

        if epoch == FREEZE_EPOCHS + 1:
            unfreeze_backbone(model)
            optimizer = torch.optim.Adam(model.parameters(), lr=LR / 10,
                                          weight_decay=WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=MAX_EPOCHS - epoch)

        # Train
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0
        for imgs, lbls, _ in train_loader:
            imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            tr_loss    += loss.item() * imgs.size(0)
            tr_correct += (model(imgs).argmax(1) == lbls).sum().item()
            tr_total   += imgs.size(0)

        # Validate
        model.eval()
        vl_loss, vl_probs, vl_lbls = 0.0, [], []
        with torch.no_grad():
            for imgs, lbls, _ in val_loader:
                imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                logits = model(imgs)
                vl_loss += criterion(logits, lbls).item() * imgs.size(0)
                vl_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                vl_lbls.extend(lbls.cpu().numpy())

        vl_loss /= len(vl_lbls)
        val_auc = (float(roc_auc_score(vl_lbls, vl_probs))
                   if len(set(vl_lbls)) == 2 else 0.0)
        scheduler.step()

        history["epoch"].append(epoch)
        history["train_loss"].append(round(tr_loss / tr_total, 5))
        history["val_loss"].append(round(vl_loss, 5))
        history["val_auc"].append(round(val_auc, 4))

        print(f"  Epoch {epoch:3d}/{MAX_EPOCHS} | "
              f"train_loss={tr_loss/tr_total:.4f} | "
              f"val_loss={vl_loss:.4f} | val_auc={val_auc:.4f}")

        if vl_loss < best_val_loss:
            best_val_loss    = vl_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    # ── Save history ─────────────────────────────────────────────────
    history_df = pd.DataFrame(history)
    history_df.to_csv(model_dir / "training_history.csv", index=False)

    # ── Test evaluation ──────────────────────────────────────────────
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    all_probs, all_lbls = [], []
    with torch.no_grad():
        for imgs, lbls, _ in test_loader:
            probs = torch.softmax(model(imgs.to(DEVICE)), 1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_lbls.extend(lbls.numpy())

    y_true, y_prob = np.array(all_lbls), np.array(all_probs)
    np.save(model_dir / "test_y_true.npy", y_true)
    np.save(model_dir / "test_y_prob.npy", y_prob)

    metrics = compute_metrics(y_true, y_prob)
    print()
    print("Test Results:")
    print(f"  AUC-ROC     : {metrics['auc_roc']:.4f}")
    print(f"  Sensitivity : {metrics['sensitivity']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print()
    print(f"Best model: {best_path}")
    print("Done.")


if __name__ == "__main__":
    main()
