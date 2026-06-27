"""
artifact/reader_study.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

import warnings
warnings.filterwarnings("ignore")


def _stratified_sample_cases(
    manifest: pd.DataFrame,
    deviations: np.ndarray,
    n_target: int,
    findings: List[str],
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    selected_idx = set()

    # Deviation quantiles
    q33 = float(np.percentile(deviations, 33))
    q67 = float(np.percentile(deviations, 67))

    def _add_stratum(mask: np.ndarray, n: int):
        cands = np.where(mask)[0]
        if len(cands) == 0:
            return
        chosen = rng.choice(cands, size=min(n, len(cands)), replace=False)
        selected_idx.update(chosen.tolist())

    n_per_finding = max(1, n_target // (len(findings) * 3))  # 3 = 3 deviation buckets

    for f in findings:
        if f not in manifest.columns:
            continue
        col  = pd.to_numeric(manifest[f], errors="coerce")
        pos  = col == 1.0
        for lo, hi in [(None, q33), (q33, q67), (q67, None)]:
            dev_mask  = (deviations >= (lo or -np.inf)) & \
                        (deviations <  (hi or  np.inf))
            stratum   = pos.values & dev_mask
            _add_stratum(stratum, n_per_finding)

    # Fill to n_target with random cases from MIMIC only
    mimic_mask = (manifest["dataset"] == "mimic").values
    remaining  = [i for i in range(len(manifest))
                  if i not in selected_idx and mimic_mask[i]]
    if len(selected_idx) < n_target and remaining:
        extra = rng.choice(remaining,
                           size=min(n_target - len(selected_idx), len(remaining)),
                           replace=False)
        selected_idx.update(extra.tolist())

    idx = sorted(selected_idx)[:n_target]
    out = manifest.iloc[idx].copy()
    out["deviation_score"] = deviations[idx]
    # Shuffle for blinding
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out["reader_case_id"] = [f"RC_{i:04d}" for i in range(len(out))]
    return out


def _generate_triplets(
    gold_cases: pd.DataFrame,
    deviations: np.ndarray,
    manifest: pd.DataFrame,
    n_triplets: int,
    seed: int,
) -> pd.DataFrame:
    rng    = np.random.RandomState(seed)
    ids    = gold_cases["case_id"].tolist()
    n      = len(ids)
    rows   = []
    for _ in range(n_triplets * 3):   # oversample and deduplicate
        if len(rows) >= n_triplets:
            break
        ia, ib, ic = rng.choice(n, size=3, replace=False)
        rows.append({
            "triplet_id": f"T_{len(rows):04d}",
            "case_a":     ids[ia],
            "case_b":     ids[ib],
            "case_c":     ids[ic],
            "choice":     "",    # to be filled by radiologist
            "confidence": "",
        })
    return pd.DataFrame(rows[:n_triplets])


def _stratified_sample_histo(
    manifest: pd.DataFrame,
    deviations: Optional[np.ndarray],
    n_target: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    selected_idx = set()
    if "tissue_class" in manifest.columns:
        classes = sorted(manifest["tissue_class"].dropna().astype(str).unique())
    else:
        classes = []

    have_dev = deviations is not None and len(deviations) == len(manifest)
    if have_dev:
        q33 = float(np.percentile(deviations, 33))
        q67 = float(np.percentile(deviations, 67))
        buckets = [(None, q33), (q33, q67), (q67, None)]
    else:
        buckets = [(None, None)]
    n_per_class = max(1, n_target // (max(1, len(classes)) * len(buckets)))

    def _add(mask: np.ndarray, n: int):
        cands = np.where(mask)[0]
        if len(cands) == 0:
            return
        chosen = rng.choice(cands, size=min(n, len(cands)), replace=False)
        selected_idx.update(chosen.tolist())

    for cls in classes:
        cls_mask = (manifest["tissue_class"].astype(str) == cls).values
        for lo, hi in buckets:
            if have_dev:
                dev_mask = (deviations >= (lo if lo is not None else -np.inf)) & \
                           (deviations <  (hi if hi is not None else  np.inf))
                _add(cls_mask & dev_mask, n_per_class)
            else:
                _add(cls_mask, n_per_class)

    # Fill to target with any remaining patches
    if len(selected_idx) < n_target:
        remaining = [i for i in range(len(manifest)) if i not in selected_idx]
        if remaining:
            extra = rng.choice(remaining,
                               size=min(n_target - len(selected_idx), len(remaining)),
                               replace=False)
            selected_idx.update(extra.tolist())

    idx = sorted(selected_idx)[:n_target]
    out = manifest.iloc[idx].copy()
    if have_dev:
        out["deviation_score"] = deviations[idx]
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    out["reader_case_id"] = [f"PC_{i:04d}" for i in range(len(out))]
    return out


def main_reader_study_pathology(global_config_path: str) -> str:
    cfg    = read_config(global_config_path)["Convergence"]
    rs_cfg = cfg["reader_study"]
    out_dir = os.path.join(
        cfg["alignment"]["results_base_dir"], "results_e7_artifact", "reader"
    )
    os.makedirs(out_dir, exist_ok=True)

    gold_path = os.path.join(out_dir, "path_gold_sample.csv")
    if os.path.exists(gold_path):
        print("Outputs exist; skipping sampling.")
        return out_dir

    histo_manifest_csv = cfg["histo"]["pool_manifest_csv"]
    if not os.path.exists(histo_manifest_csv):
        print(f"Histo manifest not found: {histo_manifest_csv}. "
              f"Run main_build_histo_pool first.")
        return out_dir
    manifest = read_csv_defensively(histo_manifest_csv)

    # Optional per-case deviation score for histo (if a histo drift pass exists).
    dev_csv = rs_cfg.get("histo_deviation_score_csv", "")
    deviations = None
    if dev_csv and os.path.exists(dev_csv):
        dev_df   = read_csv_defensively(dev_csv)
        manifest = manifest.merge(dev_df[["case_id", "deviation_score"]],
                                  on="case_id", how="left")
        deviations = manifest["deviation_score"].fillna(0.0).values.astype(float)

    seed       = int(cfg.get("seed", 42))
    n_gold     = int(rs_cfg.get("n_gold_path", 500))
    n_triplets = int(rs_cfg.get("n_triplets_path", 300))

    path_gold = _stratified_sample_histo(manifest, deviations, n_gold, seed)
    path_gold.to_csv(gold_path, index=False)

    path_trips = _generate_triplets(path_gold, deviations, manifest,
                                    n_triplets, seed)
    path_trips.to_csv(os.path.join(out_dir, "path_triplets.csv"), index=False)
    return out_dir


def main_reader_study(global_config_path: str) -> str:
    cfg    = read_config(global_config_path)["Convergence"]
    rs_cfg = cfg["reader_study"]
    out_dir = os.path.join(
        cfg["alignment"]["results_base_dir"], "results_e7_artifact", "reader"
    )
    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(os.path.join(out_dir, "r1_gold_sample.csv")):
        print("Outputs exist; skipping sampling.")
        return out_dir

    # Load deviation scores
    dev_csv = rs_cfg["deviation_score_csv"]
    if not os.path.exists(dev_csv):
        print(f"Deviation scores not found: {dev_csv}. "
              f"Run main_drift_detector first.")
        return out_dir

    manifest   = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    dev_df     = read_csv_defensively(dev_csv)
    manifest   = manifest.merge(dev_df[["case_id","deviation_score"]],
                                on="case_id", how="left")
    deviations = manifest["deviation_score"].fillna(0.0).values.astype(float)

    seed    = int(cfg.get("seed", 42))
    rng     = np.random.RandomState(seed)
    findings = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]

    n_r1        = int(rs_cfg.get("n_gold_r1", 500))
    n_trips_r1  = int(rs_cfg.get("n_triplets_r1", 300))
    n_overlap   = int(rs_cfg.get("n_overlap_r2", 150))
    n_ext       = int(rs_cfg.get("n_extension_r2", 150))
    n_trip_ov   = int(rs_cfg.get("n_triplet_overlap_r2", 100))

    r1_gold = _stratified_sample_cases(
        manifest, deviations, n_r1, findings, seed
    )
    r1_gold.to_csv(os.path.join(out_dir, "r1_gold_sample.csv"), index=False)

    r1_trips = _generate_triplets(r1_gold, deviations, manifest, n_trips_r1, seed)
    r1_trips.to_csv(os.path.join(out_dir, "r1_triplets.csv"), index=False)

    r2_overlap = r1_gold.head(n_overlap).copy()
    r2_overlap.to_csv(os.path.join(out_dir, "r2_overlap.csv"), index=False)

    r2_trip_ov = r1_trips.head(n_trip_ov).copy()
    r2_trip_ov.to_csv(os.path.join(out_dir, "r2_triplets_overlap.csv"), index=False)

    r1_ids     = set(r1_gold["case_id"].astype(str))
    ext_mask   = ~manifest["case_id"].astype(str).isin(r1_ids)
    ext_manifest = manifest[ext_mask].copy()
    ext_devs     = deviations[ext_mask.values]
    r2_ext = _stratified_sample_cases(ext_manifest.reset_index(drop=True),
                                       ext_devs, n_ext, findings, seed + 1)
    r2_ext.to_csv(os.path.join(out_dir, "r2_extension.csv"), index=False)

    return out_dir