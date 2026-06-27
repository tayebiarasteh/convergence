"""
alignment/convergence_map.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import combinations
from typing import List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from alignment.metrics import (
    load_embeddings, align_by_case_ids, compute_all_metrics, mknn,
)
from config.serde import read_config
from Inference.stats_utils import bootstrap_proportion, bh_fdr, N_BOOT

import warnings
warnings.filterwarnings("ignore")


def _n_subsample(emb_a, emb_b, ids_a, ids_b, n: int, seed: int):
    a, b, shared = align_by_case_ids(emb_a, ids_a, emb_b, ids_b)
    if n < len(shared):
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(shared), size=n, replace=False)
        a, b = a[idx], b[idx]
    return a, b, min(n, len(shared))


def _compute_pair(
    enc_a: str, enc_b: str,
    pool: str,
    output_dir: str,
    cfg: dict,
    k_values: list,
    headline_k: int,
    n_max: int,
    seed: int,
) -> List[dict]:
    """Compute all alignment metrics for one encoder pair on one pool."""
    path_a = os.path.join(output_dir, enc_a, f"{pool}.npy")
    path_b = os.path.join(output_dir, enc_b, f"{pool}.npy")
    if not os.path.exists(path_a):
        path_a = path_a.replace(".npy", ".npz")
    if not os.path.exists(path_b):
        path_b = path_b.replace(".npy", ".npz")
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        return []

    npz = lambda p: np.load(p, allow_pickle=True)
    def _load(p):
        d = npz(p)
        if hasattr(d, "files"):
            return d["embeddings"].astype(np.float32), d["case_ids"].astype(str)
        return d.astype(np.float32), np.arange(d.shape[0]).astype(str)

    emb_a, ids_a = _load(path_a)
    emb_b, ids_b = _load(path_b)

    a_full, b_full, shared = align_by_case_ids(emb_a, ids_a, emb_b, ids_b)

    finite = np.isfinite(a_full).all(axis=1) & np.isfinite(b_full).all(axis=1)
    if not finite.all():
        a_full, b_full = a_full[finite], b_full[finite]
    n_available = a_full.shape[0]

    rows = []
    n = min(n_max, n_available)
    if n < 100:
        return []

    rng = np.random.RandomState(seed)
    idx = rng.choice(n_available, size=n, replace=False)
    a, b = a_full[idx], b_full[idx]

    m = compute_all_metrics(a, b, k_values=k_values, headline_k=headline_k)

    headline_metric = f"mknn_k{headline_k}"
    boot_n   = min(n, 3000)
    n_boot_pair = 200
    boot_vals = []
    if boot_n >= 100:
        for _ in range(n_boot_pair):
            bi = rng.choice(boot_n, size=boot_n, replace=True)
            boot_vals.append(mknn(a[:boot_n][bi], b[:boot_n][bi], k=headline_k))
        boot_vals = np.array(boot_vals, dtype=float)
        h_std = float(np.std(boot_vals, ddof=1))
        h_lo  = float(np.percentile(boot_vals, 2.5))
        h_hi  = float(np.percentile(boot_vals, 97.5))
    else:
        h_std = h_lo = h_hi = float("nan")

    for metric_name, value in m.items():
        row = {
            "encoder_a": enc_a, "encoder_b": enc_b,
            "pool": pool, "n": n, "metric": metric_name,
            "value": float(value),
        }
        if metric_name == headline_metric:
            row["value_std"]     = h_std
            row["value_ci_low"]  = h_lo
            row["value_ci_high"] = h_hi
        rows.append(row)
    return rows


def main_convergence_map(global_config_path: str) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    emb_dir = cfg["embeddings"]["output_dir"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e1_alignment")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "pairwise_alignment.csv")

    from alignment.resume_utils import load_done_units, append_unit, finalize_partial
    key_cols = ["comparison_type", "enc_a", "enc_b"]
    done_pairs, partial_csv = load_done_units(out_dir, "pairwise_alignment", key_cols)
    if done_pairs:
        print(f"[E1] {len(done_pairs)} pairs already present; computing only missing.")

    def _run_pair(enc_a, enc_b, pool, comp_type, k_values, headline_k, n_max, seed):
        if (comp_type, enc_a, enc_b) in done_pairs:
            return
        rows = _compute_pair(enc_a, enc_b, pool, emb_dir, cfg,
                             k_values, headline_k, n_max, seed)
        for r in rows:
            r["comparison_type"] = comp_type
            r["enc_a"] = enc_a
            r["enc_b"] = enc_b
        append_unit(partial_csv, rows)
        done_pairs.add((comp_type, enc_a, enc_b))

    k_values   = aln_cfg.get("k_values", [5, 10, 20, 50])
    headline_k = int(aln_cfg.get("headline_k", 10))
    n_cap      = int(aln_cfg.get("alignment_max_n", 20000))
    n_max      = min(max(aln_cfg.get("n_samples", [20000])), n_cap)
    seed       = int(cfg.get("seed", 42))
    print(f"[E1] alignment N capped at {n_max} (CKNNA/CKA are O(N^2) memory).")

    from encoders.image_encoders import list_encoder_names
    core_encoders  = list_encoder_names(global_config_path, roles=["core"])
    floor_encoders = list_encoder_names(global_config_path, roles=["random_init"])

    # Within-modality CXR pairs
    print("[E1] Computing within-CXR alignment pairs...")
    for enc_a, enc_b in tqdm(list(combinations(core_encoders, 2)),
                             desc="[E1] within-CXR", unit="pair"):
        _run_pair(enc_a, enc_b, "cxr_pool", "within_cxr",
                  k_values, headline_k, n_max, seed)

    if floor_encoders:
        print(f"[E1] Computing random-init floor ({len(floor_encoders)} x "
              f"{len(core_encoders)} pairs)...")
        for fenc in floor_encoders:
            for enc_b in tqdm(core_encoders, desc=f"[E1] floor:{fenc}", unit="pair"):
                if fenc == enc_b:
                    continue
                _run_pair(fenc, enc_b, "cxr_pool", "floor_cxr",
                          k_values, headline_k, n_max, seed)
    else:
        print("[E1] WARNING: no random_init encoder found; floor baseline skipped. "
              "Add an encoder with role 'random_init' to anchor the convergence claim.")

    # Discriminant control: within-modality alignment on UNRELATED modalities.
    print("[E1] Computing within-modality alignment on discriminant-control pools...")
    for pool_b, pool_name in [("derm_pool", "within_derm"),
                               ("mammo_pool", "within_mammo"),
                               ("fundus_pool", "within_fundus")]:
        for enc_a, enc_b in tqdm(list(combinations(core_encoders, 2)),
                                 desc=f"[E1] {pool_name}", unit="pair"):
            _run_pair(enc_a, enc_b, pool_b, pool_name,
                      k_values, headline_k, n_max, seed)

    # Pathology within-modality alignment (histo_pool).
    print("[E1] Computing within-histopathology alignment pairs...")
    histo_encoders = list_encoder_names(global_config_path, roles=["xmod"])
    all_histo = sorted(set(histo_encoders) | set(core_encoders))
    for enc_a, enc_b in tqdm(list(combinations(all_histo, 2)),
                             desc="[E1] within-histo", unit="pair"):
        _run_pair(enc_a, enc_b, "histo_pool", "within_histo",
                  k_values, headline_k, n_max, seed)

    df = finalize_partial(partial_csv, out_csv,
                          dedup_cols=key_cols + ["k", "n"])

    # Summary: within vs cross
    _within_vs_cross_summary(df, out_dir, headline_k)

    print(f"[E1] {len(df)} alignment scores -> {out_csv}")
    return out_dir


def _within_vs_cross_summary(df: pd.DataFrame, out_dir: str, headline_k: int):
    """Aggregate alignment per comparison_type for EVERY metric, each with
    bootstrap mean/std/95%CI (over encoder pairs) via the unified report layer."""
    from Inference.report_utils import report_metric, report_permutation_2, add_fdr
    rows = []
    for metric_name, msub in df.groupby("metric"):
        for comp_type, grp in msub.groupby("comparison_type"):
            vals = grp["value"].dropna().values
            if len(vals) == 0:
                continue
            rep = report_metric(vals, prefix="value", is_percent=True)
            rep.update({"metric": metric_name, "comparison_type": comp_type})
            rows.append(rep)
    summ = pd.DataFrame(rows)
    summ.to_csv(os.path.join(out_dir, "within_vs_cross.csv"), index=False)

    # Permutation test: within-modality vs cross/floor, per metric (FDR-corrected).
    cmp_rows = []
    for metric_name, msub in df.groupby("metric"):
        within = msub[msub["comparison_type"].str.startswith("within_")]["value"].dropna().values
        other  = msub[~msub["comparison_type"].str.startswith("within_")]["value"].dropna().values
        if len(within) >= 2 and len(other) >= 2:
            rep = report_permutation_2(within, other, prefix="within_minus_other",
                                       is_percent=True)
            rep["metric"] = metric_name
            cmp_rows.append(rep)
    if cmp_rows:
        cmp_df = add_fdr(pd.DataFrame(cmp_rows), family_cols=None)
        cmp_df.to_csv(os.path.join(out_dir, "within_vs_cross_tests.csv"), index=False)