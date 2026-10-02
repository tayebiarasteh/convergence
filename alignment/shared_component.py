"""
alignment/shared_component.py
Created on August 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from Inference.report_utils import add_fdr, report_metric, report_spearman
from Inference.resume_utils import (MissingInput, append_status, check_build_params,
                                    ensure_dir, fingerprint_file, status_path,
                                    write_build_params, write_csv_atomic, write_npz_atomic)
from Inference.stats_utils import exceedance_p
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _neighbor_graph(cfg: Dict, enc: str, pool: str, case_ids: np.ndarray, k: int,
                    cache_dir: str) -> np.ndarray:
    from alignment.metrics import _knn_graph
    from encoders.cache_utils import embedding_cache_path, load_embedding_cache
    path = os.path.join(cache_dir, f"knn__{enc}__{pool}.npz")
    expected = {"cache": fingerprint_file(embedding_cache_path(cfg, enc, pool)), "k": int(k),
                "n_cases": int(len(case_ids)), "first": str(case_ids[0]),
                "last": str(case_ids[-1])}
    if os.path.exists(path) and check_build_params(path, expected, owner="E14_knn"):
        return np.load(path)["graph"]
    emb, ids = load_embedding_cache(cfg, enc, pool)
    pos = {str(c): i for i, c in enumerate(ids)}
    sub = emb[np.array([pos[c] for c in case_ids])]
    if not np.isfinite(sub).all():
        raise MissingInput(
            f"{enc} carries a non-finite embedding among the selected cases; restrict the "
            f"selection to rows finite in every encoder before calling this.")
    g = np.asarray(_knn_graph(sub, k))
    ensure_dir(cache_dir)
    write_npz_atomic(path, graph=g)
    write_build_params(path, expected)
    return g


def _row_overlap(ga: np.ndarray, gb: np.ndarray) -> np.ndarray:
    k = ga.shape[1]
    gb_sorted = np.sort(gb, axis=1)
    pos = np.empty_like(ga)
    for j in range(k):
        pos[:, j] = (gb_sorted < ga[:, j:j + 1]).sum(axis=1)
    pos = np.clip(pos, 0, k - 1)
    return (np.take_along_axis(gb_sorted, pos, axis=1) == ga).sum(axis=1) / float(k)


def _per_case_agreement(cfg: Dict, encoders: List[str], pool: str, case_ids: np.ndarray,
                        k: int, cache_dir: str):
    from itertools import combinations
    graphs = []
    for enc in tqdm(encoders, desc="[E14] neighbor graphs", unit="enc"):
        graphs.append(np.asarray(_neighbor_graph(cfg, enc, pool, case_ids, k, cache_dir)))
    n = len(case_ids)
    pairs = list(combinations(range(len(graphs)), 2))
    mean_overlap = np.zeros(n, dtype=np.float64)
    for a, b in tqdm(pairs, desc="[E14] pair overlaps", unit="pair"):
        mean_overlap += _row_overlap(graphs[a], graphs[b])
    mean_overlap /= max(1, len(pairs))
    all_share = np.zeros(n, dtype=np.float32)
    for i in range(n):
        sets = [set(g[i]) for g in graphs]
        union = set.union(*sets)
        all_share[i] = (len(set.intersection(*sets)) / len(union)) if union else 0.0
    return mean_overlap.astype(np.float32), all_share


def main_shared_component(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln = cfg["alignment"]
    out_dir = os.path.join(aln["results_base_dir"], "results_e14_shared")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "shared_component_attribution.csv")
    status = status_path(cfg, "e14_shared")
    from Inference.regimes import regime_extra
    from encoders.cache_utils import embedding_cache_path, finite_case_ids, finite_ids_cache_dir
    from encoders.panel import list_encoder_names
    from Inference.resume_utils import output_is_current, record_sources
    _sources = ([cfg["cxr"]["pool_manifest_csv"]]
                + [embedding_cache_path(cfg, e, "cxr_pool")
                   for e in list_encoder_names(global_config_path, roles=["core"])])
    _sources = [p for p in _sources if os.path.exists(p)]
    extra = regime_extra("e14")
    if not force and output_is_current(out_csv, _sources, owner="E14", extra=extra):
        return out_csv
    from Inference.resume_utils import claim_unit, release_claim
    if not claim_unit(out_dir, "shared"):
        return out_csv

    encoders = list_encoder_names(global_config_path, roles=["core"])
    pool = "cxr_pool"
    matched_n = int(aln["matched_n"])
    k = max(2, int(round(float(aln["k_fraction"]) * matched_n)))
    seed = int(cfg["seed"])

    man = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    cache_dir = finite_ids_cache_dir(cfg)
    ids_present = None
    for enc in tqdm(encoders, desc="[E14] finite rows", unit="enc"):
        s = set(finite_case_ids(cfg, enc, pool).tolist())
        ids_present = s if ids_present is None else (ids_present & s)
    common = sorted(set(man["case_id"].astype(str)) & ids_present)
    if len(common) < matched_n:
        raise MissingInput(f"only {len(common)} cases are cached for every core encoder, fewer "
                           f"than the matched count of {matched_n}.")
    rng = np.random.default_rng(seed)
    sel = np.array(common)[rng.choice(len(common), size=matched_n, replace=False)]
    agree, all_share = _per_case_agreement(cfg, encoders, pool, sel, k, cache_dir)

    sub = man.set_index(man["case_id"].astype(str)).loc[sel].reset_index(drop=True)
    from data_loader.build_utils import age_band
    from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
    findings = [f for f in CANONICAL_CXR_FINDINGS if f in sub.columns]
    if "age" in sub.columns:
        sub["age_band"] = age_band(sub["age"].values).astype(str).values
    blocks = {
        "acquisition_view": [c for c in ("view",) if c in sub.columns],
        "site": [c for c in ("dataset",) if c in sub.columns],
        "patient": [c for c in ("sex", "age_band") if c in sub.columns],
        "clinical_finding": findings,
    }
    blocks = {b: cols for b, cols in blocks.items() if cols}
    if not blocks:
        raise MissingInput("the chest manifest carries none of the attributes E14 decomposes.")

    def _design(cols: List[str]) -> np.ndarray:
        mats = []
        for c in cols:
            v = sub[c]
            if c in findings:
                mats.append(pd.to_numeric(v, errors="coerce").fillna(0.0).values.reshape(-1, 1))
            else:
                d = pd.get_dummies(v.astype(str), drop_first=True).values.astype(float)
                if d.shape[1]:
                    mats.append(d)
        return np.hstack(mats) if mats else np.zeros((len(sub), 0))

    def _r2(X: np.ndarray, y: np.ndarray) -> float:
        if X.shape[1] == 0:
            return 0.0
        A = np.hstack([np.ones((len(y), 1)), X])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        ss_tot = float(((y - y.mean()) ** 2).sum())
        return float(1.0 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else float("nan")

    y = agree.astype(float)
    designs = {b: _design(cols) for b, cols in blocks.items()}
    full = np.hstack([designs[b] for b in blocks]) if blocks else np.zeros((len(y), 0))
    r2_full = _r2(full, y)

    n_perm = int(cfg["stats"]["n_perm"])
    rows: List[Dict] = []
    for b in blocks:
        others = np.hstack([designs[o] for o in blocks if o != b]) \
            if len(blocks) > 1 else np.zeros((len(y), 0))
        unique = r2_full - _r2(others, y)
        alone = _r2(designs[b], y)
        rp = np.random.RandomState(seed)
        draws = np.empty(n_perm, dtype=float)
        for i in range(n_perm):
            perm = designs[b][rp.permutation(len(y))]
            draws[i] = _r2(np.hstack([others, perm]), y) - _r2(others, y)
        rows.append({"block": b, "columns": ";".join(blocks[b]),
                     "n_columns": int(designs[b].shape[1]),
                     "r2_unique_raw": unique, "r2_alone_raw": alone,
                     "r2_full_raw": r2_full,
                     "reference_mean_raw": float(np.mean(draws)),
                     "reference_ci_lower_raw": float(np.percentile(draws, 2.5)),
                     "reference_ci_upper_raw": float(np.percentile(draws, 97.5)),
                     "excess_over_reference_raw": unique - float(np.mean(draws)),
                     "p_raw": exceedance_p(unique, draws), "p_fdr": float("nan"),
                     "n_cases": int(len(y)), "n_encoders": len(encoders),
                     "matched_n": matched_n, "k": k,
                     "outcome": "mean_pairwise_neighbor_overlap", "consensus_dependent": False})
    df = add_fdr(pd.DataFrame(rows), family_cols=None)

    per_case = pd.DataFrame({"case_id": sel, "cross_encoder_agreement": agree,
                             "all_encoder_share": all_share})
    for c in ("dataset", "view", "sex", "age_band"):
        if c in sub.columns:
            per_case[c] = sub[c].values
    per_case_csv = os.path.join(out_dir, "per_case_agreement.csv")
    write_csv_atomic(per_case, per_case_csv)
    record_sources(per_case_csv, _sources, extra=extra)
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources, extra=extra)
    release_claim(out_dir, "shared")
    append_status(status, f"E14 decomposed cross-encoder agreement over {len(blocks)} blocks")
    return out_csv
