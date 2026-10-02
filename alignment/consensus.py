"""
alignment/consensus.py
Created on June 21, 2026

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
from Inference.resume_utils import (print_projected_peak, MissingInput, append_status, status_path,
                                    write_build_params, write_csv_atomic, write_npy_atomic)
warnings.filterwarnings("ignore")


def compute_relative_reps(
    embeddings: np.ndarray,
    anchor_idx: np.ndarray,
) -> np.ndarray:
    anchors = embeddings[anchor_idx]
    return (embeddings @ anchors.T).astype(np.float32, copy=False)


def _procrustes_align(target: np.ndarray, source: np.ndarray) -> np.ndarray:
    R, _ = orthogonal_procrustes(source, target)
    return source @ R


def _anchor_case_ids(cfg: dict, manifest, n_anchors: int, seed: int):
    import pandas as pd
    from data_loader.build_utils import read_csv_defensively
    dev_csv = cfg["splits"]["anchor_dev_csv"]
    if not os.path.exists(dev_csv):
        raise MissingInput(f"the anchor development set is absent at {dev_csv}; run main_build_splits first.")
    return [str(c) for c in read_csv_defensively(dev_csv)["case_id"].astype(str)][:n_anchors]


def write_anchor_case_ids(path: str, case_ids) -> None:
    import pandas as pd
    from Inference.resume_utils import write_csv_atomic
    if os.path.exists(path):
        try:
            if read_anchor_case_ids(path) == [str(c) for c in case_ids]:
                return
        except Exception:
            pass
    write_csv_atomic(pd.DataFrame({"anchor_case_id": list(case_ids)}), path)


def read_anchor_case_ids(path: str):
    from data_loader.build_utils import read_csv_defensively
    if not os.path.exists(path):
        raise MissingInput(f"anchors not found at {path}; run the consensus stage first.")
    return [str(c) for c in read_csv_defensively(path)["anchor_case_id"].astype(str)]


_CONSENSUS_LEGACY = {"gpa_centering": "none"}


def consensus_build_params(cfg: dict, global_config_path: str) -> dict:
    from encoders.panel import list_encoder_names
    from Inference.resume_utils import fingerprint_file
    aln_cfg = cfg["alignment"]
    n_anchors = int(aln_cfg.get("primary_anchors", 1024))
    core = list_encoder_names(global_config_path, roles=list(aln_cfg["xmod_roles"]))
    return {
        "n_anchors": n_anchors,
        "anchor_sweep_max": max([int(x) for x in aln_cfg.get("anchor_sweep", [n_anchors])] + [n_anchors]),
        "gpa_max_iter": int(aln_cfg.get("gpa_max_iter", 30)),
        "gpa_tol": float(aln_cfg.get("gpa_tol", 1e-5)),
        "consensus_max_cases": int(cfg.get("embeddings", {}).get("consensus_max_cases", 50000) or 0),
        "encoders": sorted(core),
        "anchor_dev": fingerprint_file(cfg["splits"]["anchor_dev_csv"]),
        "gpa_centering": "column_means",
    }


def consensus_state(cfg: dict, global_config_path: str) -> Tuple[str, str]:
    from Inference.resume_utils import _moved_keys, claim_is_live, read_build_params
    cons_dir = cfg["alignment"]["consensus_dir"]
    flag = os.path.join(cons_dir, "gpa_converged.flag")
    consensus_path = os.path.join(cons_dir, "consensus.npy")
    if claim_is_live(cons_dir, "consensus"):
        return "pending", "a live job is rebuilding the consensus (main_build_consensus)"
    if not (os.path.exists(flag) and os.path.exists(consensus_path)):
        return "pending", f"no finished consensus fit at {cons_dir}; run main_build_consensus"
    stored = dict(read_build_params(consensus_path) or {})
    for k, was in _CONSENSUS_LEGACY.items():
        stored.setdefault(k, was)
    moved = _moved_keys(stored, consensus_build_params(cfg, global_config_path))
    if moved:
        return "pending", (f"the consensus on disk was built under other parameters "
                           f"({', '.join(k for k, _, _ in moved)}); main_build_consensus rebuilds it")
    with open(flag) as fh:
        ok = fh.read().strip() == "1"
    if ok:
        return "ready", "the fit reached its tolerance"
    return "not_converged", "the generalized Procrustes fit did not reach its tolerance"


def _center_and_scale(m: np.ndarray) -> np.ndarray:
    c = (m - m.mean(axis=0, dtype=np.float64, keepdims=True)).astype(np.float32, copy=False)
    f = float(np.linalg.norm(c, "fro"))
    return (c / f) if f > 0 else c


def generalized_procrustes(
    matrices: List[np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-6,
    on_iteration=None,
) -> Tuple[np.ndarray, List[float], bool]:
    K = len(matrices)
    N, D = matrices[0].shape

    normed = [_center_and_scale(m) for m in matrices]

    consensus = normed[0].copy()
    history: List[float] = []
    converged = False

    d_resid = float("inf")
    for it in range(max_iter):
        acc = np.zeros_like(consensus, dtype=np.float64)
        aligned_cache = []
        for m in normed:
            a = _procrustes_align(consensus, m)
            acc += a
            aligned_cache.append(a)
        new_consensus = (acc / K).astype(consensus.dtype)
        del acc
        scale = float(np.linalg.norm(new_consensus, "fro"))
        if scale > 0:
            new_consensus /= scale
        new_consensus = _procrustes_align(consensus, new_consensus)

        total = 0.0
        for a in aligned_cache:
            total += float(np.linalg.norm(a - new_consensus, "fro")) / np.sqrt(N)
        del aligned_cache
        history.append(total / K)

        change = float(np.linalg.norm(new_consensus - consensus, "fro"))
        consensus = new_consensus
        if on_iteration is not None:
            on_iteration()
        d_resid = abs(history[-1] - history[-2]) if len(history) > 1 else float("inf")
        if d_resid < tol:
            converged = True
            break

    return consensus, history, converged


def procrustes_residuals(matrices: List[np.ndarray], consensus: np.ndarray) -> List[float]:
    out = []
    n = consensus.shape[0]
    for m in matrices:
        mn = _center_and_scale(m)
        out.append(float(np.linalg.norm(_procrustes_align(consensus, mn) - consensus, "fro"))
                   / np.sqrt(n))
    return out


def per_case_deviation(
    matrices: List[np.ndarray],
    k: int = 10,
    on_step=None,
) -> np.ndarray:
    from alignment.metrics import _knn_graph
    N = matrices[0].shape[0]
    K = len(matrices)

    nn_graphs = []
    for m in tqdm(matrices, desc="  kNN graphs", unit="enc"):
        nn_graphs.append(_knn_graph(m, k))
        if on_step is not None:
            on_step()
    deviations = np.zeros(N, dtype=np.float32)

    for i in tqdm(range(N), desc="  per-case deviation", unit="case"):
        if on_step is not None and i % 2000 == 0:
            on_step()
        sets = [set(nn_graphs[e][i]) for e in range(K)]
        intersection = set.intersection(*sets)
        union        = set.union(*sets)
        if union:
            deviations[i] = 1.0 - len(intersection) / len(union)
        else:
            deviations[i] = 0.0

    return deviations


def main_build_consensus(global_config_path: str) -> str:
    cfg        = read_config(global_config_path)["Convergence"]
    emb_cfg    = cfg["embeddings"]
    aln_cfg    = cfg["alignment"]
    cons_dir   = aln_cfg["consensus_dir"]
    status = status_path(cfg, "consensus")
    output_dir = emb_cfg["output_dir"]
    n_anchors  = int(aln_cfg.get("primary_anchors", 1024))
    seed       = int(cfg.get("seed", 42))

    consensus_path   = os.path.join(cons_dir, "consensus.npy")
    residuals_path   = os.path.join(cons_dir, "encoder_residuals.csv")
    deviations_path  = os.path.join(cons_dir, "case_deviations.npy")
    anchors_path     = os.path.join(cons_dir, "anchor_case_ids.csv")

    from encoders.panel import list_encoder_names
    from Inference.resume_utils import check_build_params, heartbeat_claim
    core_encoders = list_encoder_names(global_config_path,
                                       roles=list(aln_cfg["xmod_roles"]))
    build_params = consensus_build_params(cfg, global_config_path)
    flag_path = os.path.join(cons_dir, "gpa_converged.flag")

    def _finished() -> bool:
        return (all(os.path.exists(p) for p in
                    [consensus_path, residuals_path, deviations_path, anchors_path, flag_path])
                and check_build_params(consensus_path, build_params, owner="build_consensus",
                                       legacy=_CONSENSUS_LEGACY))

    if _finished():
        return cons_dir

    os.makedirs(cons_dir, exist_ok=True)
    from Inference.resume_utils import claim_unit, release_claim
    if not claim_unit(cons_dir, "consensus"):
        return cons_dir
    if _finished():
        release_claim(cons_dir, "consensus")
        return cons_dir
    if os.path.exists(flag_path):
        os.remove(flag_path)

    def _beat():
        heartbeat_claim(cons_dir, "consensus")

    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    manifest = read_csv_defensively(pool_csv)
    from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

    sweep_max = max([int(n) for n in aln_cfg.get("anchor_sweep", [n_anchors])] + [n_anchors])
    anchor_ids_all = _anchor_case_ids(cfg, manifest, sweep_max, seed)
    anchor_ids = anchor_ids_all[:n_anchors]
    id_to_row = {str(c): i for i, c in enumerate(manifest["case_id"].astype(str))}
    absent = [c for c in anchor_ids_all if c not in id_to_row]
    if absent:
        raise MissingInput(f"{len(absent)} anchor case ids are not in the pool manifest; "
                           f"rerun main_build_splits against the current pool.")
    anchor_rows = np.array([id_to_row[c] for c in anchor_ids_all], dtype=np.int64)

    N_full = len(manifest)
    max_cases = cfg.get("embeddings", {}).get("consensus_max_cases", 50000)
    if max_cases and N_full > int(max_cases):
        rng = np.random.RandomState(seed)
        n_random = max(0, int(max_cases) - len(anchor_rows))
        others = np.setdiff1d(np.arange(N_full), anchor_rows, assume_unique=False)
        drawn = rng.choice(others, size=min(n_random, len(others)), replace=False)
        case_subset = np.sort(np.concatenate([anchor_rows, drawn]))
    else:
        case_subset = np.arange(N_full)
    write_npy_atomic(os.path.join(cons_dir, "consensus_case_subset_idx.npy"), case_subset)
    subset_pos = {str(c): i for i, c in enumerate(manifest["case_id"].astype(str).values[case_subset])}
    anchor_idx = np.array([subset_pos[c] for c in anchor_ids], dtype=np.int64)
    write_anchor_case_ids(anchors_path, anchor_ids_all)

    _n_cases = int(aln_cfg.get("consensus_max_cases", 50000)) or 50000
    print_projected_peak("consensus", {
        "relative reps": len(core_encoders) * _n_cases * n_anchors * 4 / 1024 ** 3,
        "consensus and rotations":
            (_n_cases * n_anchors + len(core_encoders) * n_anchors ** 2) * 4 / 1024 ** 3,
    }, note=f"{len(core_encoders)} encoders, {_n_cases} cases, {n_anchors} anchors.")
    pool_name     = "cxr_pool"

    matrices:     List[np.ndarray] = []
    encoder_names: List[str]       = []

    nan_audit = []
    for enc_name in tqdm(core_encoders, desc="[build_consensus] encoders", unit="enc"):
        _beat()
        npz_path = os.path.join(output_dir, enc_name, f"{pool_name}.npz")
        if not os.path.exists(npz_path):
            nan_audit.append((enc_name, 0.0, "MISSING_NPZ"))
            continue
        d   = np.load(npz_path, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32, copy=False)
        if len(case_subset) != emb.shape[0]:
            emb = emb[case_subset]
        finite_frac = float(np.isfinite(emb).all(axis=1).mean())
        if finite_frac < 0.5:
            nan_audit.append((enc_name, finite_frac, "SKIPPED_MOSTLY_NAN"))
            continue
        rel = compute_relative_reps(emb, anchor_idx)
        matrices.append(rel)
        encoder_names.append(enc_name)
        nan_audit.append((enc_name, finite_frac, "USED"))

    audit_df = pd.DataFrame(nan_audit, columns=["encoder", "finite_row_fraction", "status"])
    audit_path = os.path.join(cons_dir, "consensus_nan_audit.csv")
    write_csv_atomic(audit_df, audit_path)
    n_requested = len(core_encoders)
    n_used      = int((audit_df["status"] == "USED").sum())
    n_dropped_enc = n_requested - n_used
    if n_dropped_enc:
        bad = audit_df[audit_df["status"] != "USED"]

    if len(matrices) < 2:
        raise RuntimeError("[build_consensus] Need at least 2 encoders for GPA.")

    max_drop_frac = float(cfg.get("embeddings", {}).get("consensus_max_encoder_drop_frac", 0.25))
    allow_degraded = bool(cfg.get("embeddings", {}).get("consensus_allow_degraded", False))
    if (n_dropped_enc / max(n_requested, 1)) > max_drop_frac and not allow_degraded:
        raise RuntimeError(
            f"[build_consensus] {n_dropped_enc}/{n_requested} encoders were dropped "
            f"for missing or mostly-NaN embeddings (> {max_drop_frac:.0%} threshold). "
            f"This usually means several image embedding extractions are incomplete or failed. "
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
    kept_local = np.where(good_rows)[0]
    kept_idx   = case_subset[kept_local]
    write_npy_atomic(os.path.join(cons_dir, "consensus_kept_case_idx.npy"), kept_idx)

    if matrices[0].shape[0] < 2 or matrices[0].shape[1] < 2:
        raise RuntimeError(
            "[build_consensus] After dropping non-finite rows/cols, too little data "
            "remains for GPA. Check which encoders have all-NaN embeddings "
            "(a failed image embedding extraction) and re-extract or skip them."
        )

    gpa_max_iter = int(aln_cfg.get("gpa_max_iter", 30))
    gpa_tol      = float(aln_cfg.get("gpa_tol", 1e-5))
    consensus, history, converged = generalized_procrustes(
        matrices, max_iter=gpa_max_iter, tol=gpa_tol, on_iteration=_beat)
    write_npy_atomic(consensus_path, consensus)
    write_build_params(consensus_path, build_params)

    residuals = procrustes_residuals(matrices, consensus)
    write_csv_atomic(pd.DataFrame({
        "encoder":  encoder_names,
        "residual": residuals,
        "gpa_converged": converged,
        "gpa_iterations": len(history),
        "gpa_final_mean_distance": history[-1],
        "gpa_last_residual_change": (abs(history[-1] - history[-2]) if len(history) > 1
                                     else float("nan")),
        "gpa_tolerance": gpa_tol,
        "gpa_criterion": "change in the mean distance to the consensus",
    }), residuals_path)
    from Inference.regimes import regime_extra
    write_build_params(residuals_path, {"gpa_converged": converged,
                                        "n_encoders": len(encoder_names),
                                        "gpa_tol": gpa_tol, **regime_extra("consensus")})

    _per_group_fits(cfg, encoder_names, matrices, cons_dir, gpa_max_iter, gpa_tol,
                    on_iteration=_beat)

    dev_cap = int(cfg["embeddings"].get("deviation_max_cases", 20000)) \
        if isinstance(cfg.get("embeddings"), dict) else 20000
    Nkept = matrices[0].shape[0]
    if dev_cap and Nkept > dev_cap:
        rng_dev = np.random.RandomState(int(cfg.get("seed", 42)))
        dev_sel = np.sort(rng_dev.choice(Nkept, size=dev_cap, replace=False))
        dev_small = per_case_deviation([m[dev_sel] for m in matrices], k=10, on_step=_beat)
        deviations = np.full(Nkept, np.nan, dtype=np.float32)
        deviations[dev_sel] = dev_small
    else:
        deviations = per_case_deviation(matrices, k=10, on_step=_beat)
    write_npy_atomic(deviations_path, deviations)

    with open(flag_path + ".tmp", "w") as f:
        f.write("1" if converged else "0")
        f.flush()
        os.fsync(f.fileno())
    os.replace(flag_path + ".tmp", flag_path)
    if not converged:
        append_status(status, f"GPA did NOT converge in {gpa_max_iter} iterations; "
                              f"consensus-derived analyses are blocked.")

    release_claim(cons_dir, "consensus")
    append_status(status, f"consensus refitted over {len(encoder_names)} encoders, "
                          f"converged={converged}")
    return cons_dir

def _per_group_fits(cfg: dict, encoder_names, matrices, cons_dir: str,
                    max_iter: int, tol: float, on_iteration=None) -> str:
    groups = (cfg["alignment"].get("consensus_groups", {}) or {})
    if not groups:
        return ""
    pos = {str(n): i for i, n in enumerate(encoder_names)}
    rows = []
    for gname, members in groups.items():
        have = [m for m in members if str(m) in pos]
        if len(have) < 3:
            rows.append({"group": gname, "n_encoders": len(have), "fitted": False,
                         "reason": "fewer than 3 members of this group are in the panel fit"})
            continue
        mats = [matrices[pos[str(m)]] for m in have]
        cons_g, hist_g, conv_g = generalized_procrustes(mats, max_iter=max_iter, tol=tol,
                                                        on_iteration=on_iteration)
        res_g = procrustes_residuals(mats, cons_g)
        rows.append({"group": gname, "n_encoders": len(have), "fitted": True,
                     "members": ";".join(have), "gpa_converged": bool(conv_g),
                     "gpa_iterations": len(hist_g),
                     "gpa_final_mean_distance": float(hist_g[-1]),
                     "residual_mean": float(np.mean(res_g)),
                     "residual_min": float(np.min(res_g)),
                     "residual_max": float(np.max(res_g))})
    out = os.path.join(cons_dir, "consensus_per_group.csv")
    write_csv_atomic(pd.DataFrame(rows), out)
    ok = [r["group"] for r in rows if r.get("gpa_converged")]
    return out
