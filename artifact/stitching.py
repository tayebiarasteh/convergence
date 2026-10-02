"""
artifact/stitching.py
Created on June 22, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from collections import OrderedDict
from itertools import permutations
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from alignment.linear_probe import eval_probe_scores, fit_probe
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

import warnings
from Inference.resume_utils import (MissingInput, append_status, heartbeat_claim, run_units_resumable,
                                    status_path, write_csv_atomic)
warnings.filterwarnings("ignore")


def _affine_map(A: np.ndarray, B: np.ndarray) -> tuple:
    N = A.shape[0]
    A_aug = np.hstack([A, np.ones((N, 1), dtype=A.dtype)])
    W_aug, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
    return W_aug[:-1], W_aug[-1]


def _aligned_to_manifest(emb_dir: str, enc: str, man_ids: np.ndarray, rows: np.ndarray = None):
    p = os.path.join(emb_dir, enc, "cxr_pool.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32, copy=False)
    pos = pd.Series(np.arange(len(d["case_ids"])), index=d["case_ids"].astype(str))
    want = man_ids if rows is None else man_ids[rows]
    idx = pos.reindex(want).values
    aligned = np.full((len(want), emb.shape[1]), np.nan, dtype=np.float32)
    have = np.where(np.isfinite(idx))[0]
    aligned[have] = emb[idx[have].astype(int)]
    return aligned


def main_stitching(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    status  = status_path(cfg, "stitching")
    aln_cfg = cfg["alignment"]
    emb_dir = cfg["embeddings"]["output_dir"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e7_artifact")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "stitching.csv")
    partial_dir = os.path.join(out_dir, "stitching_partials")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(partial_dir)

    manifest = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    findings = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]
    labels   = {f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float, copy=False)
                for f in findings}
    man_ids  = manifest["case_id"].astype(str).values
    seed     = int(cfg.get("seed", 42))

    split_col  = manifest["split"].astype(str).str.lower()
    train_mask = (split_col == "train").values
    test_mask  = (split_col == "test").values
    if not train_mask.any() or not test_mask.any():
        raise MissingInput("the pool manifest carries no train/test split; run main_build_splits first.")
    from artifact.universal_probe import _assert_patient_disjoint_split
    _assert_patient_disjoint_split(manifest)

    from encoders.panel import list_encoder_names
    core_encs = list_encoder_names(global_config_path, roles=["core"])
    pairs = list(permutations(core_encs, 2))
    max_pairs = int(aln_cfg.get("stitching_max_pairs", 0))
    if max_pairs and len(pairs) > max_pairs:
        rng = np.random.RandomState(seed)
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), max_pairs, replace=False))]
    units = [f"{a}|{b}" for a, b in pairs]

    _aligned_cache: OrderedDict = OrderedDict()
    _ALIGN_CACHE_MAX = 4
    def _load_aligned(enc: str):
        if enc in _aligned_cache:
            _aligned_cache.move_to_end(enc)
            return _aligned_cache[enc]
        aligned = _aligned_to_manifest(emb_dir, enc, man_ids)
        _aligned_cache[enc] = aligned
        while len(_aligned_cache) > _ALIGN_CACHE_MAX:
            _aligned_cache.popitem(last=False)
        return aligned

    probe_max_train = int(cfg["embeddings"].get("probe_max_train", 50000))
    _cap_rng = np.random.RandomState(seed)
    def _cap(idx):
        if len(idx) > probe_max_train:
            return np.sort(_cap_rng.choice(idx, probe_max_train, replace=False))
        return idx

    _oracle: Dict[tuple, tuple] = {}
    def _oracle_for(enc_tgt: str, fname: str):
        key = (enc_tgt, fname)
        if key in _oracle:
            return _oracle[key]
        emb_tgt = _load_aligned(enc_tgt)
        col = labels[fname]
        out = None
        if emb_tgt is not None:
            fin = np.isfinite(emb_tgt[:, 0])
            tr = np.where(train_mask & np.isfinite(col) & fin)[0]
            te = np.where(test_mask & np.isfinite(col) & fin)[0]
            if (len(tr) >= 20 and len(te) >= 10 and len(np.unique(col[tr])) >= 2
                    and len(np.unique(col[te])) >= 2):
                trc = _cap(tr)
                probe, scaler = fit_probe(emb_tgt[trc], col[trc])
                from sklearn.metrics import roc_auc_score
                yt, ys = eval_probe_scores(probe, scaler, emb_tgt[te], col[te])
                try:
                    out = (probe, scaler, float(roc_auc_score(yt, ys)))
                except ValueError:
                    out = None
        _oracle[key] = out
        return out

    def compute_unit(unit: str) -> List[Dict]:
        from sklearn.metrics import roc_auc_score
        enc_src, enc_tgt = unit.split("|")
        emb_src = _load_aligned(enc_src)
        emb_tgt = _load_aligned(enc_tgt)
        if emb_src is None or emb_tgt is None:
            raise MissingInput(f"no embedding cache for {enc_src} or {enc_tgt}; run main_extract_image_embeddings first.")

        tr_both = np.where(train_mask & np.isfinite(emb_src[:, 0])
                           & np.isfinite(emb_tgt[:, 0]))[0]
        if len(tr_both) < 50:
            raise MissingInput(f"({enc_src}, {enc_tgt}) share {len(tr_both)} finite training "
                               f"rows, too few to fit the affine map.")
        tr_both_c = _cap(tr_both)
        W, b = _affine_map(emb_src[tr_both_c], emb_tgt[tr_both_c])

        rows: List[Dict] = []
        for fname in findings:
            col = labels[fname]
            orc = _oracle_for(enc_tgt, fname)
            if orc is None:
                continue
            probe_tgt, scaler_tgt, oracle_auroc = orc
            te_both = np.where(test_mask & np.isfinite(col)
                               & np.isfinite(emb_src[:, 0]))[0]
            if len(te_both) < 10 or len(np.unique(col[te_both])) < 2:
                continue
            method_scores = {}
            mapped_test = emb_src[te_both] @ W + b
            method_scores["affine"] = eval_probe_scores(probe_tgt, scaler_tgt,
                                                        mapped_test, col[te_both])
            if emb_src.shape[1] == emb_tgt.shape[1]:
                method_scores["identity"] = eval_probe_scores(probe_tgt, scaler_tgt,
                                                              emb_src[te_both], col[te_both])
            for method, (yt, ys) in method_scores.items():
                try:
                    auroc = float(roc_auc_score(yt, ys))
                except ValueError:
                    continue
                rows.append({
                    "enc_src": enc_src, "enc_tgt": enc_tgt, "finding": fname,
                    "method": method, "auroc": round(auroc, 4),
                    "oracle_auroc": round(oracle_auroc, 4),
                    "retention": (round(auroc / oracle_auroc, 4)
                                  if oracle_auroc > 0 else float("nan")),
                    "n_test": int(len(te_both)),
                    "unit_of_analysis": "encoder_pair_by_finding"})
            rows.append({
                "enc_src": enc_src, "enc_tgt": enc_tgt, "finding": fname,
                "method": "oracle", "auroc": round(oracle_auroc, 4),
                "oracle_auroc": round(oracle_auroc, 4), "retention": 1.0,
                "n_test": int(len(te_both)),
                "unit_of_analysis": "encoder_pair_by_finding"})
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="stitch", units=units, compute_unit=compute_unit,
        build_params={"probe_max_train": probe_max_train, "seed": seed,
                      "split": "manifest_patient_disjoint", "n_pairs": len(units)},
        progress_desc="[E7/stitch] pairs", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("stitching produced no rows; run main_extract_image_embeddings before this stage.")
    n_skipped = int(df.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(f"{n_skipped} of {len(units)} stitching pairs could not run, so the "
                           f"table would be short of pairs; the finished ones are cached and "
                           f"skip, and the stage completes once their input exists.")
    write_csv_atomic(df, out_csv)
    append_status(status, f"stitching: {len(df)} rows over {df.enc_src.nunique()} sources "
                          f"and {len(units)} directed pairs")
    return out_dir


def main_stitching_budget(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "stitching_budget")
    aln_cfg = cfg["alignment"]
    emb_dir = cfg["embeddings"]["output_dir"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e7_artifact")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "stitching_budget.csv")
    summary_csv = os.path.join(out_dir, "stitching_budget_summary.csv")
    partial_dir = os.path.join(out_dir, "stitching_budget_partials")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(partial_dir)
    budgets = sorted(int(b) for b in aln_cfg["stitching_map_budgets"])
    seed = int(cfg.get("seed", 42))
    manifest_csv = cfg["cxr"]["pool_manifest_csv"]
    from Inference.regimes import regime_extra
    from encoders.panel import list_encoder_names
    from Inference.resume_utils import output_is_current, record_sources
    core_encs = list_encoder_names(global_config_path, roles=["core"])
    _sources = [p for p in [manifest_csv] + [os.path.join(emb_dir, e, "cxr_pool.npz") for e in core_encs]
                if os.path.exists(p)]
    extra = {**regime_extra("e12_map_budget"), "budgets": budgets}
    if not force and output_is_current(summary_csv, _sources, owner="E12/budget", extra=extra):
        return out_dir

    manifest = read_csv_defensively(manifest_csv)
    from artifact.universal_probe import _assert_patient_disjoint_split
    _assert_patient_disjoint_split(manifest)
    findings = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]
    man_ids = manifest["case_id"].astype(str).values
    split_col = manifest["split"].astype(str).str.lower()
    train_all = np.where((split_col == "train").values)[0]
    test_rows = np.where((split_col == "test").values)[0]
    train_rows = np.random.RandomState(seed).permutation(train_all)[:budgets[-1] * 2]
    rows = np.concatenate([train_rows, test_rows])
    X = {}
    for enc in tqdm(core_encs, desc="[E12/budget] caches", unit="enc"):
        a = _aligned_to_manifest(emb_dir, enc, man_ids, rows)
        if a is None:
            raise MissingInput(f"no chest cache for {enc}; run main_extract_image_embeddings first.")
        X[enc] = a
    fin_all = np.all([np.isfinite(X[e]).all(axis=1) for e in core_encs], axis=0)
    ntr = len(train_rows)
    fit_pos = np.where(fin_all[:ntr])[0][:budgets[-1]]
    if len(fit_pos) < budgets[-1]:
        raise MissingInput(f"only {len(fit_pos)} training images are finite in every encoder, fewer "
                           f"than the largest map budget of {budgets[-1]}.")
    too_small = [b for b in budgets if b < max(X[e].shape[1] for e in core_encs) + 2]
    if too_small:
        raise ValueError(f"map budgets {too_small} are below the widest embedding plus 2, so the "
                         f"least-squares map would be underdetermined; raise stitching_map_budgets.")
    te_pos = ntr + np.arange(len(test_rows))
    labels = {f: pd.to_numeric(manifest[f], errors="coerce").values[rows] for f in findings}

    def compute_unit(enc_tgt: str) -> List[Dict]:
        from sklearn.metrics import roc_auc_score
        Xt = X[enc_tgt]
        probes = {}
        for f in tqdm(findings, desc=f"[E12/budget] {enc_tgt[:20]} classifiers", unit="finding", leave=False):
            heartbeat_claim(partial_dir, f"stitch_budget__{enc_tgt}")
            col = labels[f]
            tr = fit_pos[np.isfinite(col[fit_pos])]
            te = te_pos[np.isfinite(col[te_pos]) & fin_all[te_pos]]
            if len(tr) < 20 or len(te) < 10 or len(np.unique(col[tr])) < 2 or len(np.unique(col[te])) < 2:
                continue
            probe, scaler = fit_probe(Xt[tr], col[tr])
            yt, ys = eval_probe_scores(probe, scaler, Xt[te], col[te])
            probes[f] = (probe, scaler, te, float(roc_auc_score(yt, ys)))
        out: List[Dict] = []
        for enc_src in tqdm([e for e in core_encs if e != enc_tgt],
                            desc=f"[E12/budget] {enc_tgt[:20]} mapped", unit="enc", leave=False):
            heartbeat_claim(partial_dir, f"stitch_budget__{enc_tgt}")
            Xs = X[enc_src]
            for b in budgets:
                W, bias = _affine_map(Xs[fit_pos[:b]], Xt[fit_pos[:b]])
                for f, (probe, scaler, te, oracle) in probes.items():
                    yt, ys = eval_probe_scores(probe, scaler, Xs[te] @ W + bias, labels[f][te])
                    auroc = float(roc_auc_score(yt, ys))
                    out.append({"enc_src": enc_src, "enc_tgt": enc_tgt, "finding": f,
                                "n_map_images": int(b), "auroc": auroc, "oracle_auroc": oracle,
                                "retention": auroc / oracle if oracle > 0 else float("nan"),
                                "retention_chance_corrected": ((auroc - 0.5) / (oracle - 0.5)
                                                               if oracle > 0.5 else float("nan")),
                                "n_test": int(len(te)),
                                "unit_of_analysis": "encoder_pair_by_finding"})
        return out

    df = run_units_resumable(
        partial_dir=partial_dir, group="stitch_budget", units=list(core_encs),
        compute_unit=compute_unit,
        build_params={"seed": seed, "budgets": budgets, "n_fit": int(len(fit_pos)),
                      "n_test_rows": int(len(test_rows)), "probe": "fit_probe_c0.1",
                      "split": "manifest_patient_disjoint"},
        progress_desc="[E12/budget] classifier encoders", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("the map budget stage produced no rows.")
    n_skipped = int(df.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(f"{n_skipped} of {len(core_encs)} classifier encoders could not run, so the "
                           f"table would be short of pairs; the finished ones are cached and skip.")
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources, extra=extra)
    summary = df.groupby("n_map_images").agg(
        n_rows=("auroc", "size"),
        median_auroc=("auroc", "median"),
        median_oracle_auroc=("oracle_auroc", "median"),
        median_retention=("retention", "median"),
        q25_retention=("retention", lambda v: float(np.nanpercentile(v, 25))),
        q75_retention=("retention", lambda v: float(np.nanpercentile(v, 75))),
        median_retention_chance_corrected=("retention_chance_corrected", "median"),
    ).reset_index()
    write_csv_atomic(summary, summary_csv)
    record_sources(summary_csv, _sources, extra=extra)
    append_status(status, f"map budget: {len(df)} rows over {df.enc_tgt.nunique()} classifier encoders")
    return out_dir
