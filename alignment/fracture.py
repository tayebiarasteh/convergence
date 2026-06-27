"""
alignment/fracture.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from tqdm import tqdm
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from alignment.linear_probe import per_finding_auroc, fit_probe, eval_probe
from alignment.metrics import load_embeddings, align_by_case_ids, mknn
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
from Inference.stats_utils import bootstrap_spearman, bootstrap_slope, N_BOOT

import warnings
warnings.filterwarnings("ignore")


def _safe_mknn(A: np.ndarray, B: np.ndarray, k: int = 10,
              max_n: int = 2000, seed: int = 42) -> float:
    """mKNN that drops non-finite rows (medgemma_* have some NaN cases) and caps N
    (mKNN builds a kNN graph; keep it bounded). Returns NaN if too few rows remain."""
    finite = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    A, B = A[finite], B[finite]
    if A.shape[0] < 20:
        return float("nan")
    if A.shape[0] > max_n:
        rng = np.random.RandomState(seed)
        sel = rng.choice(A.shape[0], size=max_n, replace=False)
        A, B = A[sel], B[sel]
    return mknn(A, B, k=k)


def _per_finding_mknn(
    emb_a: np.ndarray, ids_a: np.ndarray,
    emb_b: np.ndarray, ids_b: np.ndarray,
    manifest: pd.DataFrame,
    finding: str,
    k: int = 10,
    max_n: int = 2000,
    seed: int = 42,
) -> float:
    """mKNN restricted to cases where finding is labeled (positive or negative)."""
    if finding not in manifest.columns:
        return float("nan")
    col    = pd.to_numeric(manifest[finding], errors="coerce")
    labeled_ids = manifest["case_id"].astype(str).values[col.notna().values]
    if len(labeled_ids) < 20:
        return float("nan")
    # Subset both embeddings to labeled cases
    set_a   = {str(c): i for i, c in enumerate(ids_a)}
    set_b   = {str(c): i for i, c in enumerate(ids_b)}
    shared  = [c for c in labeled_ids if c in set_a and c in set_b]
    if len(shared) < 20:
        return float("nan")
    if len(shared) > max_n:
        rng    = np.random.RandomState(seed)
        shared = list(rng.choice(shared, size=max_n, replace=False))
    idx_a   = [set_a[c] for c in shared]
    idx_b   = [set_b[c] for c in shared]
    return _safe_mknn(emb_a[idx_a], emb_b[idx_b], k=k, max_n=max_n, seed=seed)


def _partial_corr(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Partial Spearman correlation of x and y controlling for z."""
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if mask.sum() < 4:
        return float("nan")
    x, y, z = x[mask], y[mask], z[mask]
    # Residualize x on z
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(z.reshape(-1, 1), x)
    rx = x - lr.predict(z.reshape(-1, 1))
    lr.fit(z.reshape(-1, 1), y)
    ry = y - lr.predict(z.reshape(-1, 1))
    rho, _ = spearmanr(rx, ry)
    return float(rho)


def main_fracture_law(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    emb_cfg = cfg["embeddings"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e6_fracture")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "per_finding_alignment.csv")
    if os.path.exists(out_csv) and not force:
        print(f"[E6] Output exists; skipping (force=True to recompute). ({out_csv})")
        return out_dir

    manifest    = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    prev_csv    = cfg["prevalence"]["out_csv"]
    emb_dir     = cfg["embeddings"]["output_dir"]
    k           = int(aln_cfg.get("headline_k", 10))
    seed        = int(cfg.get("seed", 42))

    if not os.path.exists(prev_csv):
        print(f"[E6] Prevalence CSV not found: {prev_csv}. "
              f"Run compute_cxr_prevalence first.")
        return out_dir
    prevalence = read_csv_defensively(prev_csv)
    prev_map   = dict(zip(prevalence["finding"], prevalence["prevalence"]))

    # Load DINOv3-L embeddings as learnability covariate
    dinov3_path = os.path.join(emb_dir, "dinov3_l", "cxr_pool.npz")
    if not os.path.exists(dinov3_path):
        print("[E6] DINOv3-L embedding not found; learnability covariate will be NaN.")
        dinov3_emb, dinov3_ids = None, None
    else:
        d = np.load(dinov3_path, allow_pickle=True)
        dinov3_emb = d["embeddings"].astype(np.float32)
        dinov3_ids = d["case_ids"].astype(str)

    # Compute per-finding DINOv3 AUROC (learnability)
    learn_map: Dict[str, float] = {}
    if dinov3_emb is not None:
        man_ids = manifest["case_id"].astype(str).values
        enc_id_map = {c: i for i, c in enumerate(dinov3_ids)}
        shared = [c for c in man_ids if c in enc_id_map]
        man_idx = [np.where(man_ids == c)[0][0] for c in shared]
        enc_idx = [enc_id_map[c] for c in shared]
        emb_sub = dinov3_emb[enc_idx]
        labels  = {f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float)[man_idx]
                   for f in CANONICAL_CXR_FINDINGS if f in manifest.columns}
        n = len(shared)
        rng2 = np.random.RandomState(seed)
        train_m = np.zeros(n, dtype=bool)
        test_m  = np.zeros(n, dtype=bool)
        idx_shuf = rng2.permutation(n)
        cut = int(0.7 * n)
        train_m[idx_shuf[:cut]] = True
        test_m[idx_shuf[cut:]]  = True
        from alignment.linear_probe import per_finding_auroc
        learn_map = per_finding_auroc(emb_sub, labels, train_m, test_m)

    # Get all core encoder pairs
    from encoders.image_encoders import list_encoder_names
    from itertools import combinations
    core_encs = list_encoder_names(global_config_path, roles=["core"])
    pairs = list(combinations(core_encs, 2))

    # All findings (canonical + extended that are in manifest)
    all_findings = [f for f in list(manifest.columns)
                    if f not in {"case_id","dataset","modality","split",
                                 "image_key","image_subdir","subject_id",
                                 "study_id","age","sex","race","ethnicity",
                                 "insurance","view","report_rel_path"}
                    and pd.to_numeric(manifest[f], errors="coerce").notna().sum() > 20]

    from alignment.resume_utils import case_subsample_idx
    max_cases = cfg.get("embeddings", {}).get("consensus_max_cases", 50000)
    man_ids_all = manifest["case_id"].astype(str).values
    sub_idx = case_subsample_idx(len(man_ids_all), max_cases, seed)
    keep_ids = set(man_ids_all[sub_idx])
    if len(sub_idx) != len(man_ids_all):
        print(f"[E6] Subsampling {len(sub_idx)}/{len(man_ids_all)} cases for the "
              f"fracture alignment (memory bound; set embeddings.consensus_max_cases:0 "
              f"to use all).")

    emb_cache: Dict[str, tuple] = {}
    for enc in core_encs:
        p = os.path.join(emb_dir, enc, "cxr_pool.npz")
        if os.path.exists(p):
            d = np.load(p, allow_pickle=True)
            ids = d["case_ids"].astype(str)
            emb = d["embeddings"].astype(np.float32)
            # Keep only subsampled cases for this encoder.
            mask = np.array([c in keep_ids for c in ids])
            emb_cache[enc] = (emb[mask], ids[mask])
            del emb, ids

    # Per-finding alignment averaged over pairs
    rows = []
    for finding in tqdm(all_findings, desc="[E6] findings", unit="finding"):
        pair_vals = []
        for enc_a, enc_b in pairs:
            if enc_a not in emb_cache or enc_b not in emb_cache:
                continue
            ea, ia = emb_cache[enc_a]
            eb, ib = emb_cache[enc_b]
            val = _per_finding_mknn(ea, ia, eb, ib, manifest, finding, k=k, seed=seed)
            if np.isfinite(val):
                pair_vals.append(val)
        if not pair_vals:
            continue
        from Inference.report_utils import report_metric
        rep = report_metric(np.array(pair_vals, dtype=float),
                            prefix="alignment", is_percent=True)
        row = {
            "finding":     finding,
            "prevalence":  prev_map.get(finding, float("nan")),
            "log_prev":    float(np.log10(prev_map[finding]))
                           if prev_map.get(finding, 0) > 0 else float("nan"),
            "learnability": round(learn_map.get(finding, float("nan")), 4),
            "n_pairs":     len(pair_vals),
        }
        row.update(rep)
        # keep a plain 'alignment' alias (raw mean) for the regression step
        row["alignment"] = rep["alignment_mean_raw"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    # Fracture regression
    _fracture_regression(df, out_dir)

    # Per-demographic subgroup analysis
    _demographic_analysis(manifest, emb_cache, pairs, k, seed, out_dir, learn_map)

    # Gold-label re-test if available
    gold_csv = cfg["reader_study"]["gold_csv"]
    if os.path.exists(gold_csv):
        _gold_label_fracture(gold_csv, emb_cache, pairs, k, seed, out_dir)

    print(f"[E6] Fracture law -> {out_dir}")
    return out_dir


def _fracture_regression(df: pd.DataFrame, out_dir: str):
    if df.empty or "log_prev" not in df.columns:
        return
    sub = df.dropna(subset=["log_prev", "alignment"])
    if len(sub) < 4:
        return
    from Inference.report_utils import (report_spearman, report_partial_spearman,
                                        report_slope, add_fdr)
    x = sub["log_prev"].values
    y = sub["alignment"].values

    rho_rep   = report_spearman(x, y, prefix="spearman")      # estimate+CI+p
    slope_rep = report_slope(x, y, prefix="slope")            # estimate+CI+p+R^2

    # Partial Spearman controlling learnability (estimate+CI+p)
    partial_rep = {}
    if "learnability" in sub.columns:
        sub2 = sub.dropna(subset=["learnability"])
        if len(sub2) >= 5:
            partial_rep = report_partial_spearman(
                sub2["log_prev"].values, sub2["alignment"].values,
                sub2["learnability"].values, prefix="partial_spearman")

    row = {"n_findings": int(len(sub))}
    row.update({k: rho_rep.get(k) for k in (
        "spearman_estimate", "spearman_ci_low", "spearman_ci_high",
        "spearman_estimate_raw")})
    row["spearman_p_raw"] = rho_rep.get("p_raw")
    row.update({k: slope_rep.get(k) for k in (
        "slope_estimate", "slope_ci_low", "slope_ci_high",
        "slope_estimate_raw", "r_squared")})
    row["slope_p_raw"] = slope_rep.get("p_raw")
    if partial_rep:
        row.update({k: partial_rep.get(k) for k in (
            "partial_spearman_estimate", "partial_spearman_ci_low",
            "partial_spearman_ci_high", "partial_spearman_estimate_raw")})
        row["partial_p_raw"] = partial_rep.get("p_raw")

    reg_df = pd.DataFrame([row])
    reg_df.to_csv(os.path.join(out_dir, "fracture_regression.csv"), index=False)
    print(f"[E6] Fracture Spearman rho={rho_rep.get('spearman_estimate')} "
          f"CI[{rho_rep.get('spearman_ci_low')}, "
          f"{rho_rep.get('spearman_ci_high')}] p={rho_rep.get('p_raw')}")


def _demographic_analysis(
    manifest: pd.DataFrame,
    emb_cache: Dict[str, tuple],
    pairs: list,
    k: int,
    seed: int,
    out_dir: str,
    learn_map: Dict[str, float],
):
    from alignment.linear_probe import fit_probe, eval_probe
    demo_cols = ["sex", "age", "race", "ethnicity", "insurance"]
    rows = []
    for demo in demo_cols:
        if demo not in manifest.columns:
            continue
        col = manifest[demo]
        groups = col.dropna().unique()
        if len(groups) < 2 or len(groups) > 20:
            continue
        for grp in groups:
            mask = (col == grp).values
            n_grp = mask.sum()
            if n_grp < 30:
                continue
            grp_ids = manifest["case_id"].astype(str).values[mask]
            pair_vals = []
            for enc_a, enc_b in pairs[:20]:   # limit pairs for speed
                if enc_a not in emb_cache or enc_b not in emb_cache:
                    continue
                ea, ia = emb_cache[enc_a]
                eb, ib = emb_cache[enc_b]
                a_dict = {c: i for i, c in enumerate(ia)}
                b_dict = {c: i for i, c in enumerate(ib)}
                shared = [c for c in grp_ids if c in a_dict and c in b_dict]
                if len(shared) < 20:
                    continue
                idx_a = [a_dict[c] for c in shared]
                idx_b = [b_dict[c] for c in shared]
                pair_vals.append(_safe_mknn(ea[idx_a], eb[idx_b], k=k))
            if not pair_vals:
                continue
            from Inference.report_utils import report_metric
            rep = report_metric(np.array(pair_vals, dtype=float),
                               prefix="alignment", is_percent=True)
            rep.update({"demographic": demo, "group": str(grp),
                        "n_cases": n_grp, "n_pairs": len(pair_vals)})
            rep["_pair_vals"] = pair_vals   # kept transiently for the group test
            rows.append(rep)
    if rows:
        from Inference.report_utils import report_permutation_k, add_fdr
        # k-group permutation test across groups within each demographic.
        test_rows = []
        rdf = pd.DataFrame(rows)
        for demo, grp_rows in rdf.groupby("demographic"):
            groups = [np.array(pv, dtype=float) for pv in grp_rows["_pair_vals"]
                      if isinstance(pv, list) and len(pv) >= 2]
            if len(groups) >= 2:
                rep = report_permutation_k(groups)
                rep["demographic"] = demo
                test_rows.append(rep)
        out_df = rdf.drop(columns=["_pair_vals"])
        out_df.to_csv(os.path.join(out_dir, "per_demographic_alignment.csv"),
                      index=False)
        if test_rows:
            tdf = add_fdr(pd.DataFrame(test_rows), family_cols=None)
            tdf.to_csv(os.path.join(out_dir, "per_demographic_tests.csv"),
                       index=False)


def _gold_label_fracture(
    gold_csv: str,
    emb_cache: Dict[str, tuple],
    pairs: list,
    k: int,
    seed: int,
    out_dir: str,
):
    gold = read_csv_defensively(gold_csv)
    if "finding" not in gold.columns or "label" not in gold.columns:
        return
    rows = []
    for finding, grp in gold.groupby("finding"):
        grp_ids = grp["case_id"].astype(str).values
        pair_vals = []
        for enc_a, enc_b in pairs[:20]:
            if enc_a not in emb_cache or enc_b not in emb_cache:
                continue
            ea, ia = emb_cache[enc_a]
            eb, ib = emb_cache[enc_b]
            a_dict = {c: i for i, c in enumerate(ia)}
            b_dict = {c: i for i, c in enumerate(ib)}
            shared = [c for c in grp_ids if c in a_dict and c in b_dict]
            if len(shared) < 10:
                continue
            idx_a = [a_dict[c] for c in shared]
            idx_b = [b_dict[c] for c in shared]
            pair_vals.append(_safe_mknn(ea[idx_a], eb[idx_b], k=k))
        if pair_vals:
            rows.append({"finding": finding,
                         "gold_alignment": round(float(np.mean(pair_vals)), 6),
                         "n_pairs": len(pair_vals)})
    if rows:
        pd.DataFrame(rows).to_csv(
            os.path.join(out_dir, "gold_label_fracture.csv"), index=False
        )