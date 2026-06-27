"""
artifact/stitching.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import permutations
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes

from alignment.linear_probe import fit_probe, eval_probe
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

import warnings
warnings.filterwarnings("ignore")


def _affine_map(A: np.ndarray, B: np.ndarray) -> tuple:
    """Fit affine map B ≈ A @ W + b via OLS. Returns (W, b)."""
    N = A.shape[0]
    A_aug = np.hstack([A, np.ones((N, 1), dtype=A.dtype)])
    W_aug, _, _, _ = np.linalg.lstsq(A_aug, B, rcond=None)
    W = W_aug[:-1]
    b = W_aug[-1]
    return W, b


def main_stitching(global_config_path: str) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    emb_dir = cfg["embeddings"]["output_dir"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e7_artifact")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "stitching.csv")

    manifest = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    findings = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]
    labels   = {f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float)
                for f in findings}
    man_ids  = manifest["case_id"].astype(str).values
    seed     = int(cfg.get("seed", 42))

    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(manifest))
    cut = int(0.7 * len(idx))
    train_mask = np.zeros(len(manifest), dtype=bool)
    test_mask  = np.zeros(len(manifest), dtype=bool)
    train_mask[idx[:cut]] = True
    test_mask[idx[cut:]]  = True

    from encoders.image_encoders import list_encoder_names

    _aligned_cache: "OrderedDict" = __import__("collections").OrderedDict()
    _ALIGN_CACHE_MAX = 4
    def _load_aligned(enc: str) -> np.ndarray:
        if enc in _aligned_cache:
            _aligned_cache.move_to_end(enc)
            return _aligned_cache[enc]
        p = os.path.join(emb_dir, enc, "cxr_pool.npz")
        if not os.path.exists(p):
            return None
        d   = np.load(p, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32)
        enc_ids = d["case_ids"].astype(str)
        # Vectorized alignment: map encoder rows to manifest positions via a pandas
        # index join instead of a 650k-iteration Python loop (called for every pair).
        pos = pd.Series(np.arange(len(enc_ids)), index=enc_ids)
        idx = pos.reindex(man_ids).values          # NaN where the case is missing
        aligned = np.full((len(man_ids), emb.shape[1]), np.nan, dtype=np.float32)
        have = np.where(np.isfinite(idx))[0]
        aligned[have] = emb[idx[have].astype(int)]
        _aligned_cache[enc] = aligned
        # Bound cache memory: keep only the most recently used few encoders.
        while len(_aligned_cache) > _ALIGN_CACHE_MAX:
            _aligned_cache.popitem(last=False)
        return aligned

    core_encs = list_encoder_names(global_config_path, roles=["core"])
    from tqdm import tqdm
    pairs = list(permutations(core_encs, 2))[:30]  # cap pairs
    partial_csv = os.path.join(out_dir, "stitching.partial.csv")
    done_pairs = set()
    prior_frames = []
    for src in (out_csv, partial_csv):
        if os.path.exists(src):
            try:
                pf = pd.read_csv(src)
                if {"enc_src", "enc_tgt"}.issubset(pf.columns):
                    done_pairs |= set(zip(pf["enc_src"], pf["enc_tgt"]))
                    prior_frames.append(pf)
            except Exception as e:
                print(f"could not read {os.path.basename(src)} ({e}).")
    if done_pairs:
        print(f"{len(done_pairs)} pairs already done; computing only missing.")
    if prior_frames and not os.path.exists(partial_csv):
        pd.concat(prior_frames, ignore_index=True).to_csv(partial_csv, index=False)

    probe_max_train = int(cfg["embeddings"].get("probe_max_train", 50000))
    _cap_rng = np.random.RandomState(int(cfg.get("seed", 42)))
    def _cap(idx):
        if len(idx) > probe_max_train:
            return np.sort(_cap_rng.choice(idx, probe_max_train, replace=False))
        return idx

    for enc_src, enc_tgt in tqdm(pairs, desc="[E7/stitch] pairs", unit="pair"):
        if (enc_src, enc_tgt) in done_pairs:
            continue
        emb_src = _load_aligned(enc_src)
        emb_tgt = _load_aligned(enc_tgt)
        if emb_src is None or emb_tgt is None:
            continue

        pair_rows = []
        for fname in findings:
            col    = labels[fname]
            # Fit oracle probe on encoder B
            te_idx = np.where(test_mask & np.isfinite(col)
                              & np.isfinite(emb_tgt[:, 0]))[0]
            tr_idx = np.where(train_mask & np.isfinite(col)
                              & np.isfinite(emb_tgt[:, 0]))[0]
            if len(tr_idx) < 20 or len(te_idx) < 10:
                continue
            if len(np.unique(col[tr_idx])) < 2:
                continue
            tr_idx_c = _cap(tr_idx)
            probe_tgt, scaler_tgt = fit_probe(emb_tgt[tr_idx_c], col[tr_idx_c])

            # Affine map: learn A -> B on paired training rows
            tr_both = np.where(train_mask & np.isfinite(emb_src[:, 0])
                                & np.isfinite(emb_tgt[:, 0]))[0]
            te_both = np.where(test_mask & np.isfinite(col)
                                & np.isfinite(emb_src[:, 0]))[0]
            if len(tr_both) < 50 or len(te_both) < 10:
                continue
            if len(np.unique(col[te_both])) < 2:
                continue
            tr_both_c = _cap(tr_both)
            W, b = _affine_map(emb_src[tr_both_c], emb_tgt[tr_both_c])
            mapped_test = emb_src[te_both] @ W + b

            from alignment.linear_probe import eval_probe_scores
            from Inference.report_utils import report_auroc

            # Collect (y_true, y_score) per method for bootstrap CIs.
            method_scores = {}
            yt_aff, ys_aff = eval_probe_scores(probe_tgt, scaler_tgt, mapped_test, col[te_both])
            method_scores["affine"] = (yt_aff, ys_aff)
            yt_or, ys_or = eval_probe_scores(probe_tgt, scaler_tgt, emb_tgt[te_idx], col[te_idx])
            method_scores["oracle"] = (yt_or, ys_or)

            if emb_src.shape[1] == emb_tgt.shape[1]:
                te_src = np.where(test_mask & np.isfinite(col)
                                  & np.isfinite(emb_src[:, 0]))[0]
                if len(te_src) >= 10 and len(np.unique(col[te_src])) >= 2:
                    yt_id, ys_id = eval_probe_scores(probe_tgt, scaler_tgt,
                                                     emb_src[te_src], col[te_src])
                    method_scores["identity"] = (yt_id, ys_id)

            from sklearn.metrics import roc_auc_score
            def _auroc(yt, ys):
                try:
                    return float(roc_auc_score(yt, ys))
                except ValueError:
                    return float("nan")
            # Plain point AUROC (no 10k-resample bootstrap per pair x finding x method:
            # there are thousands of these and the CI fields were never used).
            oracle_auroc = _auroc(yt_or, ys_or)
            for method, (yt, ys) in method_scores.items():
                auroc = _auroc(yt, ys)
                row = {
                    "enc_src":     enc_src,
                    "enc_tgt":     enc_tgt,
                    "finding":     fname,
                    "method":      method,
                    "retention":   round(auroc / oracle_auroc, 4)
                                   if np.isfinite(oracle_auroc) and oracle_auroc > 0
                                      and np.isfinite(auroc) else float("nan"),
                    "auroc":       round(auroc, 4),
                    "oracle_auroc": round(oracle_auroc, 4),
                }
                pair_rows.append(row)
        if pair_rows:
            hdr = not os.path.exists(partial_csv)
            pd.DataFrame(pair_rows).to_csv(partial_csv, mode="a", header=hdr, index=False)
        done_pairs.add((enc_src, enc_tgt))

    df_final = pd.read_csv(partial_csv) if os.path.exists(partial_csv) else pd.DataFrame()
    df_final.to_csv(out_csv, index=False)
    try:
        os.remove(partial_csv)
    except OSError:
        pass
    return out_dir