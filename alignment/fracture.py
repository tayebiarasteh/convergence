"""
alignment/fracture.py
Created on June 21, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from alignment.linear_probe import per_finding_auroc
from alignment.metrics import mknn
from Inference.resume_utils import append_status, status_path, write_csv_atomic
from Inference.stats_utils import exceedance_p, jackknife_over_units
from config.serde import read_config
from data_loader.build_utils import age_band, read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

import warnings
from Inference.resume_utils import MissingInput
warnings.filterwarnings("ignore")

_NOT_FINDINGS = {"case_id", "dataset", "modality", "split", "image_key", "image_subdir",
                 "subject_id", "study_id", "age", "sex", "race", "ethnicity", "insurance",
                 "view", "report_rel_path"}
DEMOGRAPHICS = ["sex", "age_band", "race", "ethnicity", "insurance"]


def _safe_mknn(A: np.ndarray, B: np.ndarray, k: int = 10,
              max_n: int = 2000, seed: int = 42) -> float:
    finite = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1)
    A, B = A[finite], B[finite]
    if A.shape[0] < 20:
        return float("nan")
    if A.shape[0] > max_n:
        rng = np.random.RandomState(seed)
        sel = rng.choice(A.shape[0], size=max_n, replace=False)
        A, B = A[sel], B[sel]
    return mknn(A, B, k=k)


def _pair_values(emb_cache: Dict[str, tuple], encoders: List[str], case_ids: np.ndarray,
                 k: int) -> Dict[str, float]:
    from alignment.metrics import _knn_graph, _mknn_from_graphs
    graphs = {}
    for enc in encoders:
        emb, pos = emb_cache[enc]
        graphs[enc] = _knn_graph(emb[np.array([pos[c] for c in case_ids])], k)
    return {f"{a}__vs__{b}": float(_mknn_from_graphs(graphs[a], graphs[b], k))
            for a, b in combinations(encoders, 2)}


def _crossed(values: np.ndarray, pair_keys: List[str]) -> Dict:
    a = [p.split("__vs__")[0] for p in pair_keys]
    b = [p.split("__vs__")[1] for p in pair_keys]
    jk = jackknife_over_units(np.asarray(values, dtype=float), a, b)
    return {"crossed_std_raw": jk["std"], "crossed_ci_low_raw": jk["ci_lower"],
            "crossed_ci_high_raw": jk["ci_upper"], "crossed_n_encoders": jk.get("n_units")}


def subgroup_reference(values_by_group: Dict[str, np.ndarray], n_perm: int = 1000,
                       seed: int = 0) -> dict:
    names = [g for g, v in values_by_group.items()
             if np.isfinite(np.asarray(v, dtype=float)).any()]
    if len(names) < 2:
        return {"reference_mean": np.nan, "reference_ci_lower": np.nan,
                "reference_ci_upper": np.nan, "observed_range": np.nan,
                "excess_over_reference": np.nan, "p_reference": np.nan,
                "n_groups": len(names), "n_perm": 0, "exploratory": True}
    mat = np.column_stack([np.asarray(values_by_group[g], dtype=float) for g in names])
    keep = np.isfinite(mat).all(axis=1)
    mat = mat[keep]
    if mat.shape[0] < 2:
        return {"reference_mean": np.nan, "reference_ci_lower": np.nan,
                "reference_ci_upper": np.nan, "observed_range": np.nan,
                "excess_over_reference": np.nan, "p_reference": np.nan,
                "n_groups": len(names), "n_perm": 0, "exploratory": True}
    observed = float(mat.mean(axis=0).max() - mat.mean(axis=0).min())
    rng = np.random.RandomState(seed)
    draws = np.empty(int(n_perm), dtype=float)
    for b in range(int(n_perm)):
        shuffled = np.take_along_axis(
            mat, np.argsort(rng.random_sample(mat.shape), axis=1), axis=1)
        m = shuffled.mean(axis=0)
        draws[b] = float(m.max() - m.min())
    return {"reference_mean": float(np.mean(draws)),
            "reference_std": float(np.std(draws, ddof=1)),
            "reference_ci_lower": float(np.percentile(draws, 2.5)),
            "reference_ci_upper": float(np.percentile(draws, 97.5)),
            "observed_range": observed,
            "excess_over_reference": observed - float(np.mean(draws)),
            "p_reference": exceedance_p(observed, draws),
            "n_groups": int(mat.shape[1]), "n_pairs": int(mat.shape[0]),
            "n_perm": int(n_perm), "exploratory": True}


def matched_n_subgroup_indices(ids, matched_n: int, seed: int = 0):
    ids = np.asarray(ids)
    if len(ids) < matched_n:
        return None
    if len(ids) == matched_n:
        return ids
    rng = np.random.RandomState(seed)
    return ids[rng.choice(len(ids), size=matched_n, replace=False)]


def _subgroup_draws(manifest: pd.DataFrame, finite_ids: set, matched_n: int,
                    seed: int) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    ids_all = manifest["case_id"].astype(str).values
    usable = np.array([c in finite_ids for c in ids_all])
    for demo in DEMOGRAPHICS:
        if demo == "age_band":
            if "age" not in manifest.columns:
                continue
            col = age_band(manifest["age"].values)
        elif demo in manifest.columns:
            col = manifest[demo]
        else:
            continue
        col = pd.Series(col.values if hasattr(col, "values") else col).astype(object)
        groups = sorted({str(g) for g in col.dropna().unique()
                         if str(g).strip().lower() not in ("", "nan", "none")})
        if len(groups) < 2 or len(groups) > 20:
            continue
        draws, counts = {}, {}
        for grp in groups:
            mask = (col.astype(str).values == grp)
            counts[grp] = int(mask.sum())
            draws[grp] = matched_n_subgroup_indices(ids_all[mask & usable], matched_n, seed=seed)
        out[demo] = {"groups": groups, "draws": draws, "counts": counts,
                     "counts_finite": {g: int(((col.astype(str).values == g) & usable).sum())
                                       for g in groups}}
    return out


def main_fracture_law(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "fracture")
    aln_cfg = cfg["alignment"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e6_fracture")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "per_finding_alignment.csv")
    prev_csv    = cfg["prevalence"]["out_csv"]
    from Inference.regimes import regime_extra
    from encoders.cache_utils import embedding_cache_path, finite_case_ids, load_embedding_cache
    from encoders.panel import list_encoder_names
    from Inference.resume_utils import (claim_unit, heartbeat_claim, output_is_current,
                                        record_sources, release_claim)
    core_encs = list_encoder_names(global_config_path, roles=["core"])
    pool = str(aln_cfg["e6_pools"][0])
    seed = int(cfg.get("seed", 42))
    matched_n = int(aln_cfg["matched_n"])
    pos_n = int(aln_cfg["e6_positives_n"])
    k_fraction = float(aln_cfg["k_fraction"])
    k_pos = max(1, int(round(k_fraction * pos_n)))
    k_sub = max(1, int(round(k_fraction * matched_n)))
    max_cases = cfg.get("embeddings", {}).get("consensus_max_cases", 50000)
    _sources = [p for p in ([cfg["cxr"]["pool_manifest_csv"], prev_csv]
                            + [embedding_cache_path(cfg, e, pool) for e in core_encs])
                if os.path.exists(p)]
    extra = {**regime_extra("e6"), "positives_n": pos_n, "matched_n": matched_n,
             "k_fraction": k_fraction, "seed": seed, "case_cap": int(max_cases or 0)}
    if not force and output_is_current(out_csv, _sources, owner="E6", extra=extra):
        return out_dir
    if not claim_unit(out_dir, "e6_fracture"):
        return out_dir
    if not force and output_is_current(out_csv, _sources, owner="E6", extra=extra):
        release_claim(out_dir, "e6_fracture")
        return out_dir

    def _beat():
        heartbeat_claim(out_dir, "e6_fracture")

    if not os.path.exists(prev_csv):
        raise MissingInput(f"[E6] the prevalence table is absent at {prev_csv}; run main_compute_cxr_prevalence first.")
    manifest = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    prevalence = read_csv_defensively(prev_csv)
    prev_map = dict(zip(prevalence["finding"], prevalence["prevalence"]))
    man_ids_all = manifest["case_id"].astype(str).values

    finite_ids = None
    for enc in tqdm(core_encs, desc="[E6] finite rows", unit="enc"):
        _beat()
        s = set(finite_case_ids(cfg, enc, pool).tolist())
        finite_ids = s if finite_ids is None else (finite_ids & s)
    finite_ids = finite_ids or set()

    from alignment.resume_utils import case_subsample_idx
    sub_idx = case_subsample_idx(len(man_ids_all), max_cases, seed)
    sub_ids = set(man_ids_all[sub_idx]) & finite_ids
    draws = _subgroup_draws(manifest, finite_ids, matched_n, seed)
    load_ids = set(sub_ids)
    for d in draws.values():
        for sel in d["draws"].values():
            if sel is not None:
                load_ids |= set(map(str, sel))
    from encoders.cache_utils import npz_member_shape
    from Inference.resume_utils import print_projected_peak
    shapes = [npz_member_shape(embedding_cache_path(cfg, e, pool), "embeddings") or (0, 0)
              for e in core_encs]
    print_projected_peak("E6", {
        "kept rows": sum(len(load_ids) * int(s[1]) for s in shapes) * 4 / 1024 ** 3,
        "one full cache": max(int(s[0]) * int(s[1]) for s in shapes) * 4 / 1024 ** 3,
    }, note=f"{len(core_encs)} encoders.")

    emb_cache: Dict[str, tuple] = {}
    for enc in tqdm(core_encs, desc="[E6] load caches", unit="enc"):
        _beat()
        emb, ids = load_embedding_cache(cfg, enc, pool)
        mask = np.array([c in load_ids for c in ids])
        kept_ids = ids[mask]
        emb_cache[enc] = (emb[mask], {c: i for i, c in enumerate(kept_ids)})
        del emb, ids

    learn_map: Dict[str, float] = {}
    try:
        d_emb, d_ids = load_embedding_cache(cfg, "dinov3_l", pool)
    except MissingInput:
        pass
    else:
        man_pos = {c: i for i, c in enumerate(man_ids_all)}
        enc_id_map = {c: i for i, c in enumerate(d_ids)}
        shared = [c for c in man_ids_all if c in enc_id_map]
        man_idx = np.array([man_pos[c] for c in shared], dtype=np.int64)
        emb_sub = d_emb[[enc_id_map[c] for c in shared]]
        labels = {f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float, copy=False)[man_idx]
                  for f in CANONICAL_CXR_FINDINGS if f in manifest.columns}
        split_sub = manifest["split"].astype(str).str.lower().values[man_idx]
        learn_map = per_finding_auroc(emb_sub, labels, split_sub == "train",
                                      np.isin(split_sub, ["test", "valid"]))
        del d_emb, emb_sub

    all_findings = [f for f in manifest.columns if f not in _NOT_FINDINGS
                    and pd.to_numeric(manifest[f], errors="coerce").notna().sum() > 20]
    sub_mask = np.array([c in sub_ids for c in man_ids_all])
    site_of = dict(zip(man_ids_all, manifest["dataset"].astype(str)))
    from Inference.report_utils import report_metric
    rows, excluded = [], []
    for finding in tqdm(all_findings, desc="[E6] findings", unit="finding"):
        _beat()
        lab = pd.to_numeric(manifest[finding], errors="coerce").values
        pos_mask = sub_mask & (lab == 1.0)
        n_pos = int(pos_mask.sum())
        if n_pos < pos_n:
            excluded.append({"finding": finding, "n_positive_loaded": n_pos, "positives_n": pos_n,
                             "n_labeled": int(np.isfinite(lab).sum()),
                             "reason": "fewer positive cases in the loaded subsample than the "
                                       "common count"})
            continue
        sel = matched_n_subgroup_indices(man_ids_all[pos_mask], pos_n, seed=seed)
        vals = _pair_values(emb_cache, core_encs, sel, k_pos)
        keys = sorted(vals)
        v = np.array([vals[p] for p in keys], dtype=float)
        sites = pd.Series([site_of[c] for c in sel])
        rep = report_metric(v, prefix="alignment", is_percent=True)
        row = {"finding": finding,
               "prevalence": prev_map.get(finding, float("nan")),
               "log_prev": float(np.log10(prev_map[finding]))
                           if prev_map.get(finding, 0) > 0 else float("nan"),
               "learnability": round(learn_map.get(finding, float("nan")), 4),
               "n_pairs": len(v), "n_cases": pos_n, "k": k_pos,
               "cases": "positive", "n_sites": int(sites.nunique()),
               "largest_site_share": float(sites.value_counts(normalize=True).iloc[0]),
               "exploratory": True}
        row.update(rep)
        row.update(_crossed(v, keys))
        row["alignment"] = rep["alignment_mean_raw"]
        rows.append(row)

    df = pd.DataFrame(rows)
    written = [out_csv]
    write_csv_atomic(df, out_csv)
    nm_csv = os.path.join(out_dir, "findings_not_measurable.csv")
    write_csv_atomic(pd.DataFrame(excluded, columns=["finding", "n_positive_loaded", "positives_n",
                                                     "n_labeled", "reason"]), nm_csv)
    written.append(nm_csv)

    written += _fracture_regression(df, out_dir)
    written += _demographic_analysis(draws, emb_cache, core_encs, k_sub, seed, out_dir, matched_n,
                                     n_perm=int(cfg["stats"]["n_perm"]), on_step=_beat)

    gold_csv = cfg["reader_study"]["gold_csv"]
    if os.path.exists(gold_csv):
        _gold_label_fracture(gold_csv, emb_cache, list(combinations(core_encs, 2)),
                             k_sub, seed, out_dir)

    for p in written[1:] + written[:1]:
        record_sources(p, _sources, extra=extra)
    release_claim(out_dir, "e6_fracture")
    append_status(status, f"E6: {len(df)} findings measured, {len(excluded)} not measurable")
    return out_dir


def _fracture_regression(df: pd.DataFrame, out_dir: str) -> List[str]:
    out = os.path.join(out_dir, "fracture_regression.csv")
    if df.empty or "log_prev" not in df.columns:
        write_csv_atomic(pd.DataFrame([{"n_findings": 0,
                                        "reason": "no finding reached the common count"}]), out)
        return [out]
    sub = df.dropna(subset=["log_prev", "alignment"])
    if len(sub) < 4:
        write_csv_atomic(pd.DataFrame([{"n_findings": int(len(sub)),
                                        "reason": "fewer than 4 measurable findings"}]), out)
        return [out]
    from Inference.report_utils import (report_spearman, report_partial_spearman,
                                        report_slope)
    x = sub["log_prev"].values
    y = sub["alignment"].values

    rho_rep   = report_spearman(x, y, prefix="spearman")
    slope_rep = report_slope(x, y, prefix="slope")

    partial_rep = {}
    if "learnability" in sub.columns:
        sub2 = sub.dropna(subset=["learnability"])
        if len(sub2) >= 5:
            partial_rep = report_partial_spearman(
                sub2["log_prev"].values, sub2["alignment"].values,
                sub2["learnability"].values, prefix="partial_spearman")

    row = {"n_findings": int(len(sub)), "cases": "positive"}
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

    write_csv_atomic(pd.DataFrame([row]), out)
    return [out]


def _demographic_analysis(
    draws: Dict[str, Dict],
    emb_cache: Dict[str, tuple],
    encoders: List[str],
    k: int,
    seed: int,
    out_dir: str,
    matched_n: int,
    n_perm: int = 1000,
    on_step=None,
) -> List[str]:
    from Inference.report_utils import report_metric, report_permutation_k, add_fdr
    rows, ref_rows, test_rows = [], [], []
    for demo, d in draws.items():
        per_group: Dict[str, Dict[str, float]] = {}
        for grp in d["groups"]:
            sel = d["draws"][grp]
            if sel is None:
                continue
            if on_step is not None:
                on_step()
            per_group[grp] = _pair_values(emb_cache, encoders, np.asarray(sel), k)
        measurable = list(per_group)
        if len(measurable) < 2:
            continue
        common_pairs = sorted(set.intersection(*[set(per_group[g]) for g in measurable]))
        values_by_group = {g: np.array([per_group[g][pk] for pk in common_pairs], dtype=float)
                           for g in measurable}
        for g, v in values_by_group.items():
            rep = report_metric(v, prefix="alignment", is_percent=True)
            rep.update(_crossed(v, common_pairs))
            rep.update({"demographic": demo, "group": g, "n_cases": d["counts"].get(g, 0),
                        "n_cases_finite": d["counts_finite"].get(g, 0),
                        "n_measured": matched_n, "k": k, "n_pairs": int(len(v)),
                        "n_groups_measurable": len(measurable),
                        "n_groups_total": len(d["groups"]), "exploratory": True})
            rows.append(rep)

        ref = subgroup_reference(values_by_group, n_perm=n_perm, seed=seed)
        ref.update({"demographic": demo, "n_groups_total": len(d["groups"]),
                    "n_not_measurable": len(d["groups"]) - len(measurable),
                    "groups_not_measurable": ";".join(g for g in d["groups"] if g not in per_group)})
        ref_rows.append(ref)

        rep = report_permutation_k([values_by_group[g] for g in measurable])
        rep["demographic"] = demo
        test_rows.append(rep)

    ref_df, test_df = pd.DataFrame(ref_rows), pd.DataFrame(test_rows)
    if not ref_df.empty:
        ref_df["p_raw"] = ref_df["p_reference"]
        ref_df = add_fdr(ref_df, family_cols=None)
    if not test_df.empty:
        test_df = add_fdr(test_df, family_cols=None)
    written = [os.path.join(out_dir, "per_demographic_alignment.csv"),
               os.path.join(out_dir, "per_demographic_reference.csv"),
               os.path.join(out_dir, "per_demographic_tests.csv")]
    write_csv_atomic(pd.DataFrame(rows), os.path.join(out_dir, "per_demographic_alignment.csv"))
    write_csv_atomic(ref_df, os.path.join(out_dir, "per_demographic_reference.csv"))
    write_csv_atomic(test_df, os.path.join(out_dir, "per_demographic_tests.csv"))
    return written


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
        for enc_a, enc_b in pairs:
            if enc_a not in emb_cache or enc_b not in emb_cache:
                continue
            ea, pa = emb_cache[enc_a]
            eb, pb = emb_cache[enc_b]
            shared = [c for c in grp_ids if c in pa and c in pb]
            if len(shared) < 10:
                continue
            pair_vals.append(_safe_mknn(ea[[pa[c] for c in shared]], eb[[pb[c] for c in shared]],
                                        k=k))
        if pair_vals:
            rows.append({"finding": finding,
                         "gold_alignment": round(float(np.mean(pair_vals)), 6),
                         "n_pairs": len(pair_vals)})
    if rows:
        write_csv_atomic(pd.DataFrame(rows), os.path.join(out_dir, "gold_label_fracture.csv"))
