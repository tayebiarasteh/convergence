"""
theory/proposition.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from Inference.resume_utils import (MissingInput, append_status, run_units_resumable,
                                    status_path, write_csv_atomic)
from Inference.stats_utils import BOOT_SEED, N_BOOT, subsample_statistic
from alignment.metrics import mknn, procrustes
from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")


def _generative_map(d_s: int, d_n: int, d_in: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(10_000 + seed)
    a_sig = rng.standard_normal((d_s, d_in)).astype(np.float32, copy=False)
    a_noi = rng.standard_normal((d_n, d_in)).astype(np.float32, copy=False)
    a_sig /= np.linalg.norm(a_sig, axis=0, keepdims=True).clip(min=1e-9)
    a_noi /= np.linalg.norm(a_noi, axis=0, keepdims=True).clip(min=1e-9)
    return {"a_sig": a_sig, "a_noi": a_noi}


def _target_map(d_s: int, n_classes: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(20_000 + seed)
    w_cls = rng.standard_normal((d_s, n_classes)).astype(np.float32, copy=False)
    w_cls /= np.linalg.norm(w_cls, axis=0, keepdims=True).clip(min=1e-9)
    w_reg = rng.standard_normal((d_s, 1)).astype(np.float32, copy=False)
    w_reg /= np.linalg.norm(w_reg, axis=0, keepdims=True).clip(min=1e-9)
    return {"w_cls": w_cls, "w_reg": w_reg}


def _draw(n: int, gmap: Dict[str, np.ndarray], tmap: Dict[str, np.ndarray], d_s: int, d_n: int,
          n_classes: int, alpha: float, snr: float, seed: int):
    rng = np.random.default_rng(seed)
    s = rng.standard_normal((n, d_s)).astype(np.float32, copy=False)
    z = rng.standard_normal((n, d_n)).astype(np.float32, copy=False)
    x = snr * (s @ gmap["a_sig"]) + (z @ gmap["a_noi"])
    x = x + rng.standard_normal(x.shape).astype(np.float32, copy=False) * 0.1
    proj = s @ tmap["w_cls"]
    noise = rng.standard_normal((n, n_classes)).astype(np.float32, copy=False)
    y = np.argmax(alpha * proj + (1.0 - alpha) * noise, axis=1).astype(np.int64)
    t = (alpha * (s @ tmap["w_reg"])[:, 0]
         + (1.0 - alpha) * rng.standard_normal(n).astype(np.float32, copy=False)).astype(np.float32)
    return x, y, t


class _Enc(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int = 0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d_hidden), nn.GELU(),
                                 nn.Linear(d_hidden, d_hidden))
        self.head = nn.Linear(d_hidden, d_out) if d_out else None

    def forward(self, x):
        h = self.enc(x)
        return self.head(h) if self.head is not None else h

    def encode(self, x):
        return F.normalize(self.enc(x), p=2, dim=-1)


def _train(x, y, t, objective: str, d_hidden: int, n_classes: int,
           seed: int, epochs: int = 200, lr: float = 3e-3) -> _Enc:
    torch.manual_seed(seed)
    d_in = x.shape[1]
    d_out = {"classify": n_classes, "regress": 1, "reconstruct": d_in, "denoise": d_in}[objective]
    model = _Enc(d_in, d_hidden, d_out)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tx = torch.tensor(x)
    ty = torch.tensor(y)
    tt = torch.tensor(t)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        opt.zero_grad()
        if objective == "classify":
            loss = F.cross_entropy(model(tx), ty)
        elif objective == "regress":
            loss = F.mse_loss(model(tx).squeeze(-1), tt)
        elif objective == "reconstruct":
            loss = F.mse_loss(model(tx), tx)
        else:
            noisy = tx + torch.randn(tx.shape, generator=gen) * 0.5
            loss = F.mse_loss(model(noisy), tx)
        loss.backward()
        opt.step()
    return model.eval()


@torch.no_grad()
def _encode(model: _Enc, x: np.ndarray) -> np.ndarray:
    return model.encode(torch.tensor(x)).numpy()


OBJECTIVE_FAMILY = {"reconstruct": "retain", "denoise": "retain",
                    "classify": "predict", "regress": "predict"}


def pair_kind(obj_a: str, obj_b: str) -> str:
    if obj_a == obj_b:
        return "same_objective"
    if OBJECTIVE_FAMILY.get(obj_a) == OBJECTIVE_FAMILY.get(obj_b):
        return "within_family"
    return "across_family"


def _cell_id(alpha: float, snr: float, seed: int) -> str:
    return f"alpha{alpha:g}__snr{snr:g}__seed{seed}"


def main_theory_proposition(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    th = cfg["theory"]
    out_dir = th["results_e8_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "synthetic_alignment.csv")
    status = status_path(cfg, "e7_theory")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(os.path.join(out_dir, "partials"))

    n = int(th["n_synthetic"])
    d_s = int(th["n_signal_dims"])
    d_n = int(th["n_distractor_dims"])
    d_in = d_s + d_n
    n_classes = int(th["n_labels"])
    d_hidden = int(th.get("hidden_dim", 64))
    n_test = int(th.get("n_test", 500))
    objectives = list(th["objectives"])
    alphas = [float(a) for a in th["informativeness_grid"]]
    snrs = [float(v) for v in th["snr_grid"]]
    seeds = list(range(int(th["n_seeds"])))

    units = [_cell_id(a, sn, sd) for a in alphas for sn in snrs for sd in seeds]

    def compute_unit(unit: str) -> List[Dict]:
        alpha = float(unit.split("__")[0][len("alpha"):])
        snr = float(unit.split("__")[1][len("snr"):])
        seed = int(unit.split("seed")[1])
        gmap = _generative_map(d_s, d_n, d_in, seed)
        tmap = _target_map(d_s, n_classes, seed)
        x1, y1, t1 = _draw(n, gmap, tmap, d_s, d_n, n_classes, alpha, snr, seed * 100 + 1)
        x2, y2, t2 = _draw(n, gmap, tmap, d_s, d_n, n_classes, alpha, snr, seed * 100 + 2)
        xt, _, _ = _draw(n_test, gmap, tmap, d_s, d_n, n_classes, alpha, snr, 99_999)
        enc = {}
        for obj in objectives:
            enc[(obj, 1)] = _encode(_train(x1, y1, t1, obj, d_hidden, n_classes, seed * 10 + 1), xt)
            enc[(obj, 2)] = _encode(_train(x2, y2, t2, obj, d_hidden, n_classes, seed * 10 + 2), xt)
        rows = []
        chance = float(cfg["alignment"]["primary_k"]) / n_test
        for obj in objectives:
            a, b = enc[(obj, 1)], enc[(obj, 2)]
            rows.append({"alpha": alpha, "snr": snr, "seed": seed, "pair_type": f"{obj}_vs_{obj}",
                         "objective_a": obj, "objective_b": obj,
                         "pair_kind": pair_kind(obj, obj),
                         "is_null_case": pair_kind(obj, obj) == "within_family",
                         "mknn_raw": mknn(a, b, k=int(cfg["alignment"]["primary_k"])),
                         "procrustes_raw": procrustes(a, b),
                         "chance_mknn_raw": chance, "n_test": n_test})
        for i, oa in enumerate(objectives):
            for ob in objectives[i + 1:]:
                rows.append({"alpha": alpha, "snr": snr, "seed": seed,
                             "pair_type": f"{oa}_vs_{ob}", "objective_a": oa, "objective_b": ob,
                             "pair_kind": pair_kind(oa, ob),
                             "is_null_case": pair_kind(oa, ob) == "within_family",
                             "mknn_raw": mknn(enc[(oa, 1)], enc[(ob, 2)],
                                              k=int(cfg["alignment"]["primary_k"])),
                             "procrustes_raw": procrustes(enc[(oa, 1)], enc[(ob, 2)]),
                             "chance_mknn_raw": chance, "n_test": n_test})
        return rows

    df = run_units_resumable(
        partial_dir=os.path.join(out_dir, "partials"), group="e7",
        units=units, compute_unit=compute_unit,
        build_params={"alphas": alphas, "snrs": snrs, "objectives": objectives,
                      "n": n, "n_test": n_test, "d_s": d_s, "d_n": d_n,
                      "shared_generative_map": True, "pairing": "sample1_vs_sample2",
                      "label_rule": "per_class_prototype", "regression_target": "latent_projection"},
        progress_desc="[E7] synthetic cells", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("E7 produced no rows.")
    write_csv_atomic(df, out_csv)
    append_status(status, f"E7 wrote {len(df)} rows to {os.path.basename(out_csv)}")
    return out_dir


def main_theory_positive_control(global_config_path: str) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    th = cfg["theory"]
    out_dir = th["results_e8_dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "positive_control.csv")
    status = status_path(cfg, "e7_positive_control")

    n = int(th["n_synthetic"])
    n_test = int(th.get("n_test", 500))
    d_s, d_n = int(th["n_signal_dims"]), int(th["n_distractor_dims"])
    d_in = d_s + d_n
    d_hidden = int(th.get("hidden_dim", 64))
    k = int(cfg["alignment"]["primary_k"])
    doses = [float(v) for v in th["positive_control_doses"]]
    n_seeds = int(th["n_seeds"])

    rows = []
    for dose in tqdm(doses, desc="[E7] planted-effect doses", unit="dose"):
        for seed in range(n_seeds):
            gmap = _generative_map(d_s, d_n, d_in, seed)
            rng = np.random.default_rng(seed * 7 + 3)
            base = rng.standard_normal((n_test, d_in)).astype(np.float32, copy=False)
            shared = (rng.standard_normal((n_test, d_s)).astype(np.float32, copy=False) @ gmap["a_sig"])
            a = dose * shared + (1 - dose) * base
            b = dose * shared + (1 - dose) * rng.standard_normal((n_test, d_in)).astype(np.float32, copy=False)
            a = a / np.linalg.norm(a, axis=1, keepdims=True).clip(min=1e-9)
            b = b / np.linalg.norm(b, axis=1, keepdims=True).clip(min=1e-9)
            rows.append({"level": "metric", "dose": dose, "seed": seed, "mknn_raw": mknn(a, b, k=k),
                         "chance_mknn_raw": k / n_test, "n_test": n_test})
            r_q = np.random.default_rng(10_000 + seed)
            d_p = (d_in - d_s) // 2
            Q, _ = np.linalg.qr(r_q.standard_normal((d_in, d_in)))
            A_shared = Q[:, :d_s].T.astype(np.float32, copy=False)
            P_a = Q[:, d_s:d_s + d_p].T.astype(np.float32, copy=False)
            P_b = Q[:, d_s + d_p:d_s + 2 * d_p].T.astype(np.float32, copy=False)

            def _sample(blocks, sseed, m):
                r = np.random.default_rng(sseed)
                x = dose * (r.standard_normal((m, d_s)).astype(np.float32, copy=False) @ A_shared)
                for P in blocks:
                    x = x + (1.0 - dose) * (r.standard_normal((m, d_p)).astype(np.float32, copy=False) @ P) / np.sqrt(len(blocks))
                return (x + 0.1 * r.standard_normal((m, d_in)).astype(np.float32, copy=False)).astype(np.float32, copy=False)
            xa = _sample([P_a], seed * 100 + 11, n)
            xb = _sample([P_b], seed * 100 + 12, n)
            xt = _sample([P_a, P_b], 99_998, n_test)
            dummy_y = np.zeros(n, dtype=np.int64)
            dummy_t = np.zeros(n, dtype=np.float32)
            ea = _encode(_train(xa, dummy_y, dummy_t, "reconstruct", d_hidden, 2, seed * 10 + 1), xt)
            eb = _encode(_train(xb, dummy_y, dummy_t, "reconstruct", d_hidden, 2, seed * 10 + 2), xt)
            torch.manual_seed(seed * 10 + 1)
            ra = _Enc(d_in, d_hidden, d_in).eval()
            torch.manual_seed(seed * 10 + 2)
            rb = _Enc(d_in, d_hidden, d_in).eval()
            floor = mknn(_encode(ra, xt), _encode(rb, xt), k=k)
            rows.append({"level": "trained_encoders", "dose": dose, "seed": seed,
                         "mknn_raw": mknn(ea, eb, k=k), "untrained_floor_raw": floor,
                         "chance_mknn_raw": k / n_test, "n_test": n_test})
    df = pd.DataFrame(rows)
    verdicts = {}
    for level, g in df.groupby("level"):
        means = g.groupby("dose")["mknn_raw"].mean()
        sds = g.groupby("dose")["mknn_raw"].std(ddof=1).fillna(0.0)
        if level == "trained_encoders":
            tol = 3.0 * float(sds.mean())
            monotone = bool(np.all(np.diff(means.values) >= -tol))
            floor0 = g[g["dose"] == means.index[0]]
            floor1 = g[g["dose"] == means.index[-1]]
            at_zero_is_chance = bool(means.iloc[0] <= float(floor0["untrained_floor_raw"].mean())
                                     + 3.0 * float(floor0["untrained_floor_raw"].std(ddof=1) or 0.0)
                                     + k / n_test)
            noise_last = max(float(sds.iloc[-1] or 0.0), k / n_test)
            floor_last = float(floor1["untrained_floor_raw"].mean())
            floor_last_sd = max(float(floor1["untrained_floor_raw"].std(ddof=1) or 0.0), k / n_test)
            rises = bool(means.iloc[-1] > means.iloc[0] + 3.0 * noise_last
                         and means.iloc[-1] > floor_last + 3.0 * floor_last_sd)
        else:
            monotone = bool(np.all(np.diff(means.values) >= -1e-6))
            at_zero_is_chance = bool(abs(means.iloc[0] - k / n_test) < 3 * k / n_test)
            rises = bool(means.iloc[-1] > means.iloc[0] + 3 * k / n_test)
        verdicts[level] = (monotone, at_zero_is_chance, rises)
        df.loc[df["level"] == level, "control_monotone"] = monotone
        df.loc[df["level"] == level, "control_zero_at_chance"] = at_zero_is_chance
        df.loc[df["level"] == level, "control_rises"] = rises
    write_csv_atomic(df, out_csv)
    append_status(status, "positive control " + "; ".join(
        f"{lv}: monotone={m} zero_at_chance={z} rises={r}" for lv, (m, z, r) in verdicts.items()))
    bad = [lv for lv, (m, z, r) in verdicts.items() if not (m and z and r)]
    if bad:
        raise ValueError(f"[E7] the positive control did not recover the planted effect at level(s) "
                         f"{bad}, so no negative result in this project may be reported until it does.")
    return out_csv
