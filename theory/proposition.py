"""
theory/proposition.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from tqdm import tqdm
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from alignment.metrics import mknn, procrustes
from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")



def _make_data(N: int, d_s: int, d_n: int, d_in: int,
               n_classes: int, alpha: float, noise: float,
               rng: np.random.Generator):
    """Generate N samples from the synthetic model."""
    s     = rng.standard_normal((N, d_s)).astype(np.float32)
    n_vec = rng.standard_normal((N, d_n)).astype(np.float32)
    A_sig = rng.standard_normal((d_s, d_in)).astype(np.float32)
    A_noi = rng.standard_normal((d_n, d_in)).astype(np.float32)
    # Normalize columns
    A_sig /= np.linalg.norm(A_sig, axis=0, keepdims=True).clip(min=1e-9)
    A_noi /= np.linalg.norm(A_noi, axis=0, keepdims=True).clip(min=1e-9)

    x = s @ A_sig + n_vec @ A_noi
    eps = rng.standard_normal((N, d_in)).astype(np.float32) * noise
    x   = x + eps

    logits_sig  = s[:, :n_classes] if d_s >= n_classes \
                  else np.pad(s, [(0,0),(0,n_classes-d_s)])[:, :n_classes]
    logits_noi  = rng.standard_normal((N, n_classes)).astype(np.float32)
    logits      = alpha * logits_sig + (1 - alpha) * logits_noi
    y           = np.argmax(logits, axis=1).astype(np.int64)
    return x, y, A_sig, A_noi



class _SupEncoder(nn.Module):
    """Supervised: predict labels from x via a hidden representation."""
    def __init__(self, d_in: int, d_hidden: int, n_classes: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                  nn.Linear(d_hidden, d_hidden))
        self.head = nn.Linear(d_hidden, n_classes)

    def forward(self, x):
        return self.head(self.enc(x))

    def encode(self, x):
        return F.normalize(self.enc(x), p=2, dim=-1)


class _SSLEncoder(nn.Module):
    """SSL: reconstruct x (autoencoder; learns to preserve all variance)."""
    def __init__(self, d_in: int, d_hidden: int):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                  nn.Linear(d_hidden, d_hidden))
        self.dec = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU(),
                                  nn.Linear(d_hidden, d_in))

    def forward(self, x):
        h = self.enc(x)
        return self.dec(h)

    def encode(self, x):
        return F.normalize(self.enc(x), p=2, dim=-1)


def _train_sup(x: np.ndarray, y: np.ndarray,
               d_hidden: int, n_classes: int, epochs: int = 200,
               lr: float = 3e-3) -> _SupEncoder:
    model = _SupEncoder(x.shape[1], d_hidden, n_classes)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    tx, ty = torch.tensor(x), torch.tensor(y)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(model(tx), ty)
        loss.backward()
        opt.step()
    return model.eval()


def _train_ssl(x: np.ndarray, d_hidden: int, epochs: int = 200,
               lr: float = 3e-3) -> _SSLEncoder:
    model = _SSLEncoder(x.shape[1], d_hidden)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    tx    = torch.tensor(x)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.mse_loss(model(tx), tx)
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def _encode(model, x: np.ndarray) -> np.ndarray:
    return model.encode(torch.tensor(x)).numpy()



def main_theory_proposition(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    th_cfg  = cfg["theory"]
    out_dir = th_cfg["results_e8_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "synthetic_alignment.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[E8] Output exists; skipping (force=True to recompute). ({out_csv})")
        return out_dir

    N          = int(th_cfg.get("n_synthetic", 2000))
    d_s        = int(th_cfg.get("n_signal_dims", 32))
    d_n        = int(th_cfg.get("n_nuisance_dims", 128))
    d_in       = d_s + d_n
    n_classes  = int(th_cfg.get("n_labels", 10))
    label_noise = float(th_cfg.get("label_noise", 0.1))
    n_seeds    = int(th_cfg.get("n_seeds", 5))
    alpha_grid = list(th_cfg.get("informativeness_grid", [0.1, 0.25, 0.5, 0.75, 1.0]))
    d_hidden   = 64

    rows = []
    for alpha in tqdm(alpha_grid, desc="[E8] alpha grid", unit="alpha"):
        for seed in range(n_seeds):
            rng1 = np.random.default_rng(seed * 100 + 1)
            rng2 = np.random.default_rng(seed * 100 + 2)

            x1, y1, _, _ = _make_data(N, d_s, d_n, d_in, n_classes, alpha,
                                       label_noise, rng1)
            x2, y2, _, _ = _make_data(N, d_s, d_n, d_in, n_classes, alpha,
                                       label_noise, rng2)

            sup1 = _train_sup(x1, y1, d_hidden, n_classes)
            sup2 = _train_sup(x2, y2, d_hidden, n_classes)
            ssl1 = _train_ssl(x1, d_hidden)
            ssl2 = _train_ssl(x2, d_hidden)

            # Shared held-out set
            rng_test = np.random.default_rng(99999)
            x_test, _, _, _ = _make_data(500, d_s, d_n, d_in, n_classes, alpha,
                                          label_noise, rng_test)

            e_sup1 = _encode(sup1, x_test)
            e_sup2 = _encode(sup2, x_test)
            e_ssl1 = _encode(ssl1, x_test)
            e_ssl2 = _encode(ssl2, x_test)

            pairs = [
                ("sup_vs_sup",  e_sup1, e_sup2),
                ("ssl_vs_ssl",  e_ssl1, e_ssl2),
                ("sup_vs_ssl",  e_sup1, e_ssl1),
            ]
            for pair_type, ea, eb in pairs:
                rows.append({
                    "alpha":      alpha,
                    "seed":       seed,
                    "pair_type":  pair_type,
                    "mknn":       round(mknn(ea, eb, k=10), 6),
                    "procrustes": round(procrustes(ea, eb), 6),
                })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Per (alpha, pair_type): mean/std/95%CI over seeds (bootstrap over seed reps).
    from Inference.report_utils import report_metric
    summ_rows = []
    for (alpha, ptype), grp in df.groupby(["alpha", "pair_type"]):
        rep = report_metric(grp["mknn"].values, prefix="mknn", is_percent=True)
        rep.update({"alpha": alpha, "pair_type": ptype})
        summ_rows.append(rep)
    pd.DataFrame(summ_rows).to_csv(
        os.path.join(out_dir, "informativeness_summary.csv"), index=False)

    # Informativeness curve: alignment gap (sup_sup - ssl_ssl) per alpha, with a
    # paired bootstrap CI over seeds (same seeds resampled for both pair types).
    from Inference.report_utils import report_paired_diff
    gap_rows = []
    for alpha, grp in df.groupby("alpha"):
        sup = grp[grp["pair_type"] == "sup_vs_sup"].sort_values("seed")["mknn"].values
        ssl = grp[grp["pair_type"] == "ssl_vs_ssl"].sort_values("seed")["mknn"].values
        if len(sup) >= 2 and len(sup) == len(ssl):
            rep = report_paired_diff(sup, ssl, prefix="alignment_gap", is_percent=True)
            rep["alpha"] = alpha
            gap_rows.append(rep)
    if gap_rows:
        from Inference.report_utils import add_fdr
        gap_df = add_fdr(pd.DataFrame(gap_rows), family_cols=None)
        gap_df.to_csv(os.path.join(out_dir, "informativeness_curve.csv"), index=False)
    return out_dir

