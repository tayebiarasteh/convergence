"""
alignment/consensus.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")



def sample_anchors(
    manifest: pd.DataFrame,
    n_anchors: int,
    finding_cols: List[str],
    seed: int = 42,
) -> np.ndarray:
    rng = np.random.RandomState(seed)
    N   = len(manifest)
    if n_anchors >= N:
        return np.arange(N)

    # Stratify by which findings are positive
    candidates = list(range(N))
    selected   = set()
    per_finding = max(1, n_anchors // max(len(finding_cols), 1))

    for f in finding_cols:
        if f not in manifest.columns:
            continue
        col = pd.to_numeric(manifest[f], errors="coerce")
        pos_idx = col[col == 1.0].index.tolist()
        if not pos_idx:
            continue
        chosen = rng.choice(pos_idx, size=min(per_finding, len(pos_idx)), replace=False)
        selected.update(chosen.tolist())

    # Fill remainder randomly
    remaining = [i for i in candidates if i not in selected]
    if len(selected) < n_anchors and remaining:
        extra = rng.choice(remaining,
                           size=min(n_anchors - len(selected), len(remaining)),
                           replace=False)
        selected.update(extra.tolist())

    result = np.array(sorted(selected))[:n_anchors]
    return result


def compute_relative_reps(
    embeddings: np.ndarray,    # (N, D) L2-normalized
    anchor_idx: np.ndarray,    # (N_a,) indices
) -> np.ndarray:
    """Compute (N, N_a) cosine-similarity matrix to anchor embeddings.

    Because embeddings are L2-normalized, cosine sim = dot product.
    """
    anchors = embeddings[anchor_idx]    # (N_a, D)
    return (embeddings @ anchors.T).astype(np.float32)



def _procrustes_align(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    """Align source to target via orthogonal Procrustes. Returns aligned source."""
    R, _ = orthogonal_procrustes(source, target)
    return source @ R


def generalized_procrustes(
    matrices: List[np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> Tuple[np.ndarray, List[float]]:
    """Iterative GPA. All matrices must be (N, N_a).

    Returns (consensus, [per-iteration mean distance]).
    """
    K = len(matrices)
    N, D = matrices[0].shape

    # Initialize consensus as the first matrix
    consensus = matrices[0].copy()
    history   = []

    for it in range(max_iter):
        acc = np.zeros_like(consensus, dtype=np.float64)
        aligned_cache = []
        for m in matrices:
            a = _procrustes_align(consensus, m)
            acc += a
            aligned_cache.append(a)
        new_consensus = (acc / K).astype(consensus.dtype)
        del acc
        scale = np.linalg.norm(new_consensus, "fro")
        if scale > 0:
            new_consensus /= scale

        total = 0.0
        for a in aligned_cache:
            total += float(np.linalg.norm(a - new_consensus, "fro")) / np.sqrt(N)
        del aligned_cache
        history.append(total / K)

        change = float(np.linalg.norm(new_consensus - consensus, "fro"))
        consensus = new_consensus
        print(f"  [GPA] iter {it+1}: mean_dist={history[-1]:.5f}, "
              f"change={change:.2e}", flush=True)
        if change < tol:
            break

    return consensus, history


def per_encoder_residual(
    matrices: List[np.ndarray],
    consensus: np.ndarray,
) -> List[float]:
    """Procrustes distance of each encoder's relative representation to consensus."""
    residuals = []
    for m in matrices:
        aligned  = _procrustes_align(consensus, m)
        N        = consensus.shape[0]
        residuals.append(float(np.linalg.norm(aligned - consensus, "fro")) / np.sqrt(N))
    return residuals


def per_case_deviation(
    matrices: List[np.ndarray],
    k: int = 10,
) -> np.ndarray:
    """Per-case cross-encoder neighbor disagreement.

    For each case i, average fraction of its k-NNs that are NOT shared across
    all encoder pairs. Higher = more disagreement = more deviant case.

    Returns (N,) float array.
    """
    from alignment.metrics import _knn_graph
    N = matrices[0].shape[0]
    K = len(matrices)

    nn_graphs = [_knn_graph(m, k)
                 for m in tqdm(matrices, desc="  kNN graphs", unit="enc")]
    deviations = np.zeros(N, dtype=np.float32)

    for i in tqdm(range(N), desc="  per-case deviation", unit="case"):
        sets = [set(nn_graphs[e][i]) for e in range(K)]
        # Fraction of unique neighbors not in intersection across all encoders
        intersection = set.intersection(*sets)
        union        = set.union(*sets)
        # Disagreement = 1 - |intersection| / |union| (Jaccard distance)
        if union:
            deviations[i] = 1.0 - len(intersection) / len(union)
        else:
            deviations[i] = 0.0

    return deviations



def main_build_consensus(global_config_path: str) -> str:
    """Compute and cache the GPA consensus manifold over all core CXR encoders.

    Returns the consensus_dir path.
    """
    cfg        = read_config(global_config_path)["Convergence"]
    emb_cfg    = cfg["embeddings"]
    aln_cfg    = cfg["alignment"]
    cons_dir   = aln_cfg["consensus_dir"]
    output_dir = emb_cfg["output_dir"]
    n_anchors  = int(aln_cfg.get("headline_anchors", 1024))
    seed       = int(cfg.get("seed", 42))

    consensus_path   = os.path.join(cons_dir, "consensus.npy")
    residuals_path   = os.path.join(cons_dir, "encoder_residuals.csv")
    deviations_path  = os.path.join(cons_dir, "case_deviations.npy")
    anchors_path     = os.path.join(cons_dir, "anchors.npy")

    if all(os.path.exists(p) for p in
           [consensus_path, residuals_path, deviations_path, anchors_path]):
        print("[build_consensus] All outputs exist; skipping.")
        return cons_dir

    os.makedirs(cons_dir, exist_ok=True)

    # Load CXR pool manifest for anchor stratification
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    manifest = read_csv_defensively(pool_csv)
    from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

    N_full = len(manifest)
    max_cases = cfg.get("embeddings", {}).get("consensus_max_cases", 50000)
    if max_cases and N_full > int(max_cases):
        rng = np.random.RandomState(seed)
        case_subset = np.sort(rng.choice(N_full, size=int(max_cases), replace=False))
        print(f"[build_consensus] Subsampling {len(case_subset)}/{N_full} cases for "
              f"the consensus (set embeddings.consensus_max_cases:0 to use all).")
    else:
        case_subset = np.arange(N_full)
    np.save(os.path.join(cons_dir, "consensus_case_subset_idx.npy"), case_subset)
    # Anchors are sampled from the full manifest, then remapped into subset-local
    # row positions (anchors must be rows that exist in the subset).
    anchor_idx_full = sample_anchors(manifest, n_anchors, CANONICAL_CXR_FINDINGS, seed)
    subset_pos = {int(g): i for i, g in enumerate(case_subset)}
    # Ensure anchors are inside the subset: keep those present, top up from subset.
    anchor_local = [subset_pos[a] for a in anchor_idx_full if int(a) in subset_pos]
    if len(anchor_local) < n_anchors:
        extra = [i for i in range(len(case_subset)) if i not in set(anchor_local)]
        rng2 = np.random.RandomState(seed + 1)
        need = min(n_anchors - len(anchor_local), len(extra))
        if need > 0:
            anchor_local += list(rng2.choice(extra, size=need, replace=False))
    anchor_idx = np.array(sorted(anchor_local))   # subset-local anchor positions
    np.save(anchors_path, anchor_idx)
    print(f"[build_consensus] {len(anchor_idx)} anchors (subset-local positions).")

    # Load core image encoder embeddings for the CXR pool
    from encoders.image_encoders import list_encoder_names
    core_encoders = list_encoder_names(global_config_path, roles=["core"])
    pool_name     = "cxr_pool"

    matrices:     List[np.ndarray] = []
    encoder_names: List[str]       = []

    nan_audit = []   # (encoder, finite_row_fraction, status)
    for enc_name in tqdm(core_encoders, desc="[build_consensus] encoders", unit="enc"):
        npz_path = os.path.join(output_dir, enc_name, f"{pool_name}.npz")
        if not os.path.exists(npz_path):
            print(f"[build_consensus] Missing embedding for {enc_name}; skipping.")
            nan_audit.append((enc_name, 0.0, "MISSING_NPZ"))
            continue
        print(f"  loading {enc_name} ...", flush=True)
        d   = np.load(npz_path, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32)
        # Restrict to the sampled cases (rows) so memory stays bounded.
        if len(case_subset) != emb.shape[0]:
            emb = emb[case_subset]
        finite_frac = float(np.isfinite(emb).all(axis=1).mean())
        # Skip an encoder that is mostly NaN (a failed Stage-2 extraction) so it does
        # not poison the cross-encoder common-valid mask and silently drop every case.
        if finite_frac < 0.5:
            print(f"  WARNING: {enc_name} has only {finite_frac:.1%} finite rows; "
                  f"skipping it from the consensus (re-extract if unexpected).")
            nan_audit.append((enc_name, finite_frac, "SKIPPED_MOSTLY_NAN"))
            continue
        rel = compute_relative_reps(emb, anchor_idx)
        matrices.append(rel)
        encoder_names.append(enc_name)
        nan_audit.append((enc_name, finite_frac, "USED"))
        print(f"  {enc_name}: relative rep shape {rel.shape} "
              f"({finite_frac:.1%} finite rows)", flush=True)

    audit_df = pd.DataFrame(nan_audit, columns=["encoder", "finite_row_fraction", "status"])
    audit_path = os.path.join(cons_dir, "consensus_nan_audit.csv")
    audit_df.to_csv(audit_path, index=False)
    n_requested = len(core_encoders)
    n_used      = int((audit_df["status"] == "USED").sum())
    n_dropped_enc = n_requested - n_used
    print(f"[build_consensus] NaN audit -> {audit_path}")
    print(f"  encoders: {n_used}/{n_requested} used, {n_dropped_enc} dropped "
          f"(missing npz or mostly-NaN). See the audit CSV for per-encoder detail.")
    if n_dropped_enc:
        bad = audit_df[audit_df["status"] != "USED"]
        for _, r in bad.iterrows():
            print(f"    DROPPED {r['encoder']}: {r['status']} "
                  f"({r['finite_row_fraction']:.1%} finite)")

    if len(matrices) < 2:
        raise RuntimeError("[build_consensus] Need at least 2 encoders for GPA.")

    max_drop_frac = float(cfg.get("embeddings", {}).get("consensus_max_encoder_drop_frac", 0.25))
    allow_degraded = bool(cfg.get("embeddings", {}).get("consensus_allow_degraded", False))
    if (n_dropped_enc / max(n_requested, 1)) > max_drop_frac and not allow_degraded:
        raise RuntimeError(
            f"[build_consensus] {n_dropped_enc}/{n_requested} encoders were dropped "
            f"for missing or mostly-NaN embeddings (> {max_drop_frac:.0%} threshold). "
            f"This usually means several Stage-2 extractions are incomplete or failed. "
            f"Inspect {audit_path}, re-extract the bad encoders, or set "
            f"embeddings.consensus_allow_degraded: true to proceed on the reduced panel."
        )

    N_total, Na = matrices[0].shape
    finite_per_cell = np.ones((N_total, Na), dtype=bool)
    for m in matrices:
        np.logical_and(finite_per_cell, np.isfinite(m), out=finite_per_cell)

    good_cols = finite_per_cell.all(axis=0)
    if good_cols.sum() < 2:
        good_cols = finite_per_cell.any(axis=0)
    good_rows = finite_per_cell[:, good_cols].all(axis=1)
    del finite_per_cell
    n_drop_rows = int((~good_rows).sum())
    n_drop_cols = int((~good_cols).sum())
    row_drop_frac = n_drop_rows / max(N_total, 1)
    if n_drop_rows or n_drop_cols:
        print(f"[build_consensus] Dropping {n_drop_rows} non-finite cases "
              f"({row_drop_frac:.1%}) and {n_drop_cols} non-finite anchors "
              f"(kept {int(good_rows.sum())} cases, {int(good_cols.sum())} anchors).")

    max_case_drop = float(cfg.get("embeddings", {}).get("consensus_max_case_drop_frac", 0.20))
    if row_drop_frac > max_case_drop and not allow_degraded:
        raise RuntimeError(
            f"[build_consensus] {row_drop_frac:.1%} of cases are non-finite in at "
            f"least one encoder (> {max_case_drop:.0%} threshold), so the consensus "
            f"would rest on a minority of cases. Inspect {audit_path} to find which "
            f"encoders carry the NaNs, re-extract them, or set "
            f"embeddings.consensus_allow_degraded: true to proceed."
        )

    matrices = [m[np.ix_(good_rows, good_cols)] for m in matrices]
    # kept rows are subset-local; map back through case_subset to ORIGINAL manifest
    # indices so any downstream per-case join uses true case positions.
    kept_local = np.where(good_rows)[0]
    kept_idx   = case_subset[kept_local]
    np.save(os.path.join(cons_dir, "consensus_kept_case_idx.npy"), kept_idx)

    if matrices[0].shape[0] < 2 or matrices[0].shape[1] < 2:
        raise RuntimeError(
            "[build_consensus] After dropping non-finite rows/cols, too little data "
            "remains for GPA. Check which encoders have all-NaN embeddings "
            "(a failed Stage-2 extraction) and re-extract or skip them."
        )

    gpa_max_iter = int(aln_cfg.get("gpa_max_iter", 30))
    gpa_tol      = float(aln_cfg.get("gpa_tol", 1e-5))
    print(f"[build_consensus] Running GPA over {len(matrices)} encoders "
          f"({matrices[0].shape[0]} cases x {matrices[0].shape[1]} anchors, "
          f"max_iter={gpa_max_iter}, tol={gpa_tol:g})...", flush=True)
    consensus, history = generalized_procrustes(matrices, max_iter=gpa_max_iter,
                                                tol=gpa_tol)
    np.save(consensus_path, consensus)
    print(f"  Converged in {len(history)} iterations. "
          f"Final mean distance: {history[-1]:.4f}")

    residuals = per_encoder_residual(matrices, consensus)
    pd.DataFrame({
        "encoder":  encoder_names,
        "residual": residuals,
    }).to_csv(residuals_path, index=False)

    dev_cap = int(cfg["embeddings"].get("deviation_max_cases", 20000)) \
        if isinstance(cfg.get("embeddings"), dict) else 20000
    Nkept = matrices[0].shape[0]
    if dev_cap and Nkept > dev_cap:
        rng_dev = np.random.RandomState(int(cfg.get("seed", 42)))
        dev_sel = np.sort(rng_dev.choice(Nkept, size=dev_cap, replace=False))
        print(f"[build_consensus] Computing per-case deviation on {dev_cap}/{Nkept} "
              f"cases (set embeddings.deviation_max_cases:0 for all).")
        dev_small = per_case_deviation([m[dev_sel] for m in matrices], k=10)
        deviations = np.full(Nkept, np.nan, dtype=np.float32)
        deviations[dev_sel] = dev_small
    else:
        deviations = per_case_deviation(matrices, k=10)
    np.save(deviations_path, deviations)

    print(f"[build_consensus] Consensus written to {cons_dir}")
    return cons_dir