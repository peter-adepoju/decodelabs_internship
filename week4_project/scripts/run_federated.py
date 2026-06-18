#!/usr/bin/env python3
"""
scripts/run_federated.py
========================
Run the federated learning simulation (with and without DP) end-to-end.

Replicates the workflow from Notebooks 08 and 10 as a runnable script.

Usage:
  python scripts/run_federated.py
  python scripts/run_federated.py --mock          # Fast mock run
  python scripts/run_federated.py --no-dp         # FL only, no differential privacy
  python scripts/run_federated.py --rounds 10     # Override round count
  python scripts/run_federated.py --epsilon 4.0   # Override DP epsilon

Outputs:
  models/federated/global_model_no_dp.pth   (if --no-dp or both)
  models/federated/global_model_dp.pth      (if DP enabled)
  models/federated/fl_history_no_dp.csv
  models/federated/fl_dp_history.csv
  models/federated/test_y_prob_no_dp.npy
  models/federated/test_y_prob_dp.npy
"""

import sys
import argparse
import warnings
from pathlib import Path
from collections import OrderedDict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
import flwr as fl
from flwr.server import ServerConfig
from flwr.server.strategy import FedAvg

from src.config import load_config
from src.paths import get_paths
from src.data_utils import TBDataset, build_transforms
from src.model import build_model
from src.dp_utils import validate_model_for_dp, make_private_model
from src.fl_client import TBClient, train_one_epoch
from src.fl_server import weighted_average_metrics
from src.metrics import compute_metrics


def parse_args():
    p = argparse.ArgumentParser(description="Run FedTB-Nigeria FL simulation")
    p.add_argument("--mock",    action="store_true", help="Use mock data")
    p.add_argument("--no-dp",   action="store_true", dest="no_dp",
                   help="Run FL without DP (Notebook 08 equivalent)")
    p.add_argument("--rounds",  type=int,   default=None)
    p.add_argument("--epsilon", type=float, default=None)
    return p.parse_args()


def main():
    args    = parse_args()
    cfg     = load_config()
    paths   = get_paths()

    SEED       = cfg["project"]["random_seed"]
    N_SITES    = cfg["federated"]["num_clients"]
    N_ROUNDS   = args.rounds or (3 if args.mock else cfg["federated"]["num_rounds"])
    EPSILON    = args.epsilon or cfg["differential_privacy"]["target_epsilon"]
    DELTA      = cfg["differential_privacy"]["target_delta"]
    MAX_NORM   = cfg["differential_privacy"]["max_grad_norm"]
    LOCAL_E    = cfg["training"]["epochs_per_round"]
    IMAGE_SIZE = cfg["data"]["image_size"]
    BATCH_SIZE = cfg["training"]["batch_size"]
    SITE_NAMES = cfg["federated"]["site_names"]
    RUN_DP     = not args.no_dp

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("FedTB-Nigeria — Federated Learning Simulation")
    print("=" * 55)
    print(f"Device   : {DEVICE}")
    print(f"Mock     : {args.mock}")
    print(f"Rounds   : {N_ROUNDS}")
    print(f"Sites    : {N_SITES}")
    print(f"DP       : {RUN_DP}  (epsilon={EPSILON} if DP)")
    print()

    val_t = build_transforms(IMAGE_SIZE, "val",
                              cfg["augmentation"]["normalize_mean"],
                              cfg["augmentation"]["normalize_std"])
    train_t = build_transforms(IMAGE_SIZE, "train",
                                cfg["augmentation"]["normalize_mean"],
                                cfg["augmentation"]["normalize_std"])

    val_df  = pd.read_csv(paths["processed"] / "val.csv")
    test_df = pd.read_csv(paths["processed"] / "test.csv")
    val_loader  = DataLoader(TBDataset(val_df,  val_t), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(TBDataset(test_df, val_t), batch_size=BATCH_SIZE, shuffle=False)

    site_data = {}
    for i in range(N_SITES):
        site_data[i] = pd.read_csv(paths["processed"] / f"site_{i}_train.csv")
        print(f"  Site {i}: {len(site_data[i])} training images")
    print()

    fed_dir = paths["federated_model_dir"]
    fed_dir.mkdir(parents=True, exist_ok=True)

    def run_fl(use_dp: bool, tag: str):
        """Run one FL experiment (with or without DP). tag = 'no_dp' or 'dp'."""
        print(f"--- Running FL ({'with DP' if use_dp else 'no DP'}) ---")
        round_results = []
        eval_model = build_model(pretrained=False, num_classes=2, dropout=0.5).to(DEVICE)
        if use_dp:
            eval_model = validate_model_for_dp(eval_model).to(DEVICE)

        eps_per_round = EPSILON / N_ROUNDS if use_dp else None

        def evaluate_fn(server_round, parameters, config):
            params_dict = zip(eval_model.state_dict().keys(), parameters)
            eval_model.load_state_dict(
                OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=True)
            eval_model.eval()
            criterion = nn.CrossEntropyLoss()
            probs_all, lbls_all, tot_loss = [], [], 0.0
            with torch.no_grad():
                for imgs, lbls, _ in test_loader:
                    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                    logits = eval_model(imgs)
                    tot_loss += criterion(logits, lbls).item() * imgs.size(0)
                    probs_all.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                    lbls_all.extend(lbls.cpu().numpy())
            n   = len(lbls_all)
            auc = (float(roc_auc_score(lbls_all, probs_all))
                   if len(set(lbls_all)) == 2 else 0.0)
            round_results.append({"round": server_round, "test_auc": auc,
                                   "test_loss": tot_loss / n if n > 0 else 0.0})
            print(f"  [Round {server_round:3d}] AUC={auc:.4f}")
            return tot_loss / n if n > 0 else 0.0, {"test_auc": auc}

        class SiteClient(fl.client.NumPyClient):
            def __init__(self, cid):
                self.cid = int(cid)
                self.site_df = site_data[self.cid]

            def _fresh_model(self):
                m = build_model(pretrained=True, num_classes=2, dropout=0.5).to(DEVICE)
                if use_dp:
                    m = validate_model_for_dp(m).to(DEVICE)
                return m

            def get_parameters(self, config):
                m = self._fresh_model()
                return [v.cpu().numpy() for _, v in m.state_dict().items()]

            def fit(self, parameters, config):
                m = self._fresh_model()
                params_dict = zip(m.state_dict().keys(), parameters)
                m.load_state_dict(
                    OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=True)
                opt  = torch.optim.Adam(m.parameters(), lr=1e-4, weight_decay=1e-4)
                crit = nn.CrossEntropyLoss()
                bs   = max(2, min(BATCH_SIZE, len(self.site_df) // 2))
                loader = DataLoader(TBDataset(self.site_df, train_t),
                                    batch_size=bs, shuffle=True, num_workers=0)

                if use_dp:
                    try:
                        m, opt, loader = make_private_model(
                            m, opt, loader,
                            target_epsilon=eps_per_round, target_delta=DELTA,
                            max_grad_norm=MAX_NORM, epochs=LOCAL_E,
                        )
                    except Exception as e:
                        warnings.warn(f"DP failed for site {self.cid}: {e}")

                for _ in range(LOCAL_E):
                    m.train()
                    for imgs, lbls, _ in loader:
                        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                        opt.zero_grad()
                        crit(m(imgs), lbls).backward()
                        opt.step()

                base = m._module if hasattr(m, "_module") else m
                return ([v.cpu().numpy() for _, v in base.state_dict().items()],
                        len(self.site_df), {"cid": str(self.cid)})

            def evaluate(self, parameters, config):
                m = self._fresh_model()
                params_dict = zip(m.state_dict().keys(), parameters)
                m.load_state_dict(
                    OrderedDict({k: torch.tensor(v) for k, v in params_dict}), strict=True)
                m.eval()
                crit = nn.CrossEntropyLoss()
                probs_v, lbls_v, tot = [], [], 0.0
                with torch.no_grad():
                    for imgs, lbls, _ in val_loader:
                        imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
                        logits = m(imgs)
                        tot += crit(logits, lbls).item() * imgs.size(0)
                        probs_v.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
                        lbls_v.extend(lbls.cpu().numpy())
                n   = len(lbls_v)
                auc = float(roc_auc_score(lbls_v, probs_v)) if len(set(lbls_v)) == 2 else 0.0
                return tot / n if n > 0 else 0.0, n, {"val_auc": auc}

        strategy = FedAvg(
            fraction_fit=1.0, fraction_evaluate=1.0,
            min_fit_clients=min(2, N_SITES),
            min_evaluate_clients=min(2, N_SITES),
            min_available_clients=min(2, N_SITES),
            evaluate_fn=evaluate_fn,
            fit_metrics_aggregation_fn=weighted_average_metrics,
            evaluate_metrics_aggregation_fn=weighted_average_metrics,
        )

        fl.simulation.start_simulation(
            client_fn=SiteClient, num_clients=N_SITES,
            config=ServerConfig(num_rounds=N_ROUNDS), strategy=strategy,
        )

        # Save history
        hist_df = pd.DataFrame(round_results)
        hist_df.to_csv(fed_dir / f"fl_history_{tag}.csv", index=False)

        # Save final predictions + model
        eval_model.eval()
        probs_t, lbls_t = [], []
        with torch.no_grad():
            for imgs, lbls, _ in test_loader:
                probs = torch.softmax(eval_model(imgs.to(DEVICE)), 1)[:, 1]
                probs_t.extend(probs.cpu().numpy()); lbls_t.extend(lbls.numpy())
        y_true = np.array(lbls_t); y_prob = np.array(probs_t)
        np.save(fed_dir / f"test_y_prob_{tag}.npy", y_prob)
        torch.save(eval_model.state_dict(), fed_dir / f"global_model_{tag}.pth")

        metrics = compute_metrics(y_true, y_prob)
        print(f"  Final AUC: {metrics['auc_roc']:.4f} | "
              f"Sensitivity: {metrics['sensitivity']:.4f}")
        return metrics

    # ── Run experiments ───────────────────────────────────────────────
    metrics_no_dp = run_fl(use_dp=False, tag="no_dp")
    if RUN_DP:
        metrics_dp = run_fl(use_dp=True, tag="dp")

    print()
    print("=" * 55)
    print("Summary:")
    print(f"  FL (No DP) — AUC: {metrics_no_dp['auc_roc']:.4f}")
    if RUN_DP:
        print(f"  FL + DP    — AUC: {metrics_dp['auc_roc']:.4f}")
    print(f"  Results in: {fed_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
