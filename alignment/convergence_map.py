"""
alignment/convergence_map.py
Created on June 23, 2026

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
    align_by_case_ids, compute_all_metrics, mknn,
)
from Inference.resume_utils import (MissingInput, append_status, clear_partial_dir,
                                    run_units_resumable, status_path, write_csv_atomic)
from config.serde import read_config
from Inference.stats_utils import (N_BOOT, assert_interval_brackets, bh_fdr,
                                   bootstrap_proportion, subsample_statistic)

import warnings
warnings.filterwarnings("ignore")


def _compute_pair(
    enc_a: str, enc_b: str,
    pool: str,
    output_dir: str,
    cfg: dict,
    k_values: list,
    primary_k: int,
    n_max: int,
    seed: int,
) -> List[dict]:
    path_a = os.path.join(output_dir, enc_a, f"{pool}.npy")
    path_b = os.path.join(output_dir, enc_b, f"{pool}.npy")
    if not os.path.exists(path_a):
        path_a = path_a.replace(".npy", ".npz")
    if not os.path.exists(path_b):
        path_b = path_b.replace(".npy", ".npz")
    if not (os.path.exists(path_a) and os.path.exists(path_b)):
        raise MissingInput(f"no embedding cache for ({enc_a} or {enc_b}, {pool}); "
                           f"run main_extract_image_embeddings for that encoder before E1.")

    npz = lambda p: np.load(p, allow_pickle=True)
    def _load(p):
        d = npz(p)
        if hasattr(d, "files"):
            return d["embeddings"].astype(np.float32, copy=False), d["case_ids"].astype(str)
        return d.astype(np.float32, copy=False), np.arange(d.shape[0]).astype(str)

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

    m = compute_all_metrics(a, b, k_values=k_values, primary_k=primary_k)

    pool_cap = int(cfg["alignment"].get("pool_n_cap", 0))
    if pool_cap and min(pool_cap, n_available) > n:
        from alignment.metrics import cknna_original
        n2 = min(pool_cap, n_available)
        idx2 = rng.choice(n_available, size=n2, replace=False)
        a2, b2 = a_full[idx2], b_full[idx2]
        k2 = max(2, int(round(float(cfg["alignment"]["k_fraction"]) * n2)))
        for name, val in (("mknn_at_pool_cap", mknn(a2, b2, k=k2)),
                          ("cknna_at_pool_cap",
                           cknna_original(a2, b2, k=k2, max_n=n2))):
            rows.append({"encoder_a": enc_a, "encoder_b": enc_b, "pool": pool,
                         "n": n2, "metric": name, "value": float(val),
                         "k_at_size": k2, "size_regime": "pool_own_cap"})

    headline_metric = f"mknn_k{primary_k}"
    pool_rows = min(n_available, 4 * n)
    if pool_rows >= 2 * n:
        idx_pool = rng.choice(n_available, size=pool_rows, replace=False)
        res = subsample_statistic(
            [a_full[idx_pool], b_full[idx_pool]],
            lambda xa, xb: mknn(xa, xb, k=primary_k),
            n_sub=n,
            n_boot=int(cfg["stats"]["n_boot_neighbor"]), seed=seed)
        h_point, h_std = res["point"], res["std"]
        h_lo, h_hi, h_nsub = res["ci_lower"], res["ci_upper"], res["n_sub"]
        assert_interval_brackets(h_point, h_lo, h_hi,
                                 label=f"{enc_a} vs {enc_b} on {pool}")
    else:
        h_point = h_std = h_lo = h_hi = float("nan")
        h_nsub = 0

    for metric_name, value in m.items():
        row = {
            "encoder_a": enc_a, "encoder_b": enc_b,
            "pool": pool, "n": n, "metric": metric_name,
            "value": float(value),
        }
        if metric_name == headline_metric:
            row["value_std"]      = h_std
            row["value_ci_low"]   = h_lo
            row["value_ci_high"]  = h_hi
            row["value_at_n_sub"] = h_point
            row["n_sub"]          = h_nsub
            row["resample"]       = "subsample_without_replacement"
        rows.append(row)
    return rows


def main_convergence_map(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    emb_dir = cfg["embeddings"]["output_dir"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e1_alignment")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "pairwise_alignment.csv")
    partial_dir = os.path.join(out_dir, "partials")
    status = status_path(cfg, "e1_convergence_map")
    if force:
        clear_partial_dir(partial_dir)

    matched_n = int(aln_cfg["matched_n"])
    k_frac = float(aln_cfg["k_fraction"])
    primary_k = max(2, int(round(k_frac * matched_n)))
    k_values = sorted({max(2, int(round(f * matched_n))) for f in (k_frac / 2, k_frac, k_frac * 2, k_frac * 5)})
    seed = int(cfg["seed"])

    from encoders.panel import list_encoder_names
    core = list_encoder_names(global_config_path, roles=list(aln_cfg["e1_roles"]))
    floor = list_encoder_names(global_config_path, roles=["random_init"])

    units = []
    for pool in aln_cfg["e1_pools"]:
        comp = "within_" + pool.replace("_pool", "")
        for a, b in combinations(core, 2):
            units.append(f"{comp}|{pool}|{a}|{b}")
        for f in floor:
            for a in core:
                units.append(f"floor_{pool.replace('_pool','')}|{pool}|{f}|{a}")

    def compute_unit(unit: str) -> List[dict]:
        comp, pool, enc_a, enc_b = unit.split("|")
        rows = _compute_pair(enc_a, enc_b, pool, emb_dir, cfg,
                             k_values, primary_k, matched_n, seed)
        for r in rows:
            r.update({"comparison_type": comp, "enc_a": enc_a, "enc_b": enc_b,
                      "pool": pool, "matched_n": matched_n, "k_primary": primary_k})
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="e1", units=units, compute_unit=compute_unit,
        build_params={"matched_n": matched_n, "primary_k": primary_k,
                      "k_values": list(k_values), "seed": seed,
                      "metric_suite": "cknna_original_primary"},
        progress_desc="[E1] encoder pairs", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("E1 produced no rows; extract embeddings (main_extract_image_embeddings) before running it.")
    write_csv_atomic(df, out_csv)
    append_status(status, f"E1 wrote {len(df)} rows over {df.enc_a.nunique()} encoders "
                          f"and {df['pool'].nunique()} pools at matched N = {matched_n}")
    _within_vs_cross_summary(df, out_dir, primary_k)
    return out_csv


def _within_vs_cross_summary(df: pd.DataFrame, out_dir: str, primary_k: int):
    from Inference.report_utils import report_metric, report_permutation_2, add_fdr
    from Inference.stats_utils import jackknife_over_units
    rows = []
    for metric_name, msub in df.groupby("metric"):
        as_percent = not str(metric_name).startswith(("procrustes", "rsa_"))
        for comp_type, grp in msub.groupby("comparison_type"):
            g = grp.dropna(subset=["value"])
            if g.empty:
                continue
            rep = report_metric(g["value"].values, prefix="value", is_percent=as_percent)
            jk = jackknife_over_units(g["value"].values, g["enc_a"].values, g["enc_b"].values)
            rep.update({"metric": metric_name, "comparison_type": comp_type,
                        "kind": "metric" if as_percent else "statistic",
                        "crossed_std_raw": jk["std"], "crossed_ci_low_raw": jk["ci_lower"],
                        "crossed_ci_high_raw": jk["ci_upper"],
                        "crossed_n_encoders": jk.get("n_units"),
                        "crossed_resample": jk["resample"]})
            rows.append(rep)
    summ = pd.DataFrame(rows)
    write_csv_atomic(summ, os.path.join(out_dir, "within_vs_cross.csv"))

    cmp_rows = []
    for metric_name, msub in df.groupby("metric"):
        within = msub[msub["comparison_type"].str.startswith("within_")]["value"].dropna().values
        other  = msub[~msub["comparison_type"].str.startswith("within_")]["value"].dropna().values
        if len(within) >= 2 and len(other) >= 2:
            rep = report_permutation_2(
                within, other, prefix="within_minus_other",
                is_percent=not str(metric_name).startswith(("procrustes", "rsa_")),
                tested="within_modality")
            rep["metric"] = metric_name
            cmp_rows.append(rep)
    if cmp_rows:
        cmp_df = add_fdr(pd.DataFrame(cmp_rows), family_cols=["tested"])
        write_csv_atomic(cmp_df, os.path.join(out_dir, "within_vs_cross_tests.csv"))
