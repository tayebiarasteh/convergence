"""
data_loader/build_taix_pool.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from Inference.resume_utils import (MissingInput, append_status, check_build_params,
                                    fingerprint_file, status_path, write_build_params,
                                    write_csv_atomic)
from config.serde import read_config
from data_loader.build_utils import (binarize_presence, read_csv_defensively,
                                     write_manifest)

import warnings
warnings.filterwarnings("ignore")

CANONICAL_FROM_TAIX = ("cardiomegaly", "edema", "pleural_effusion", "lung_opacity", "atelectasis")


def _decode_binary_labels(df: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for native, canonical in label_map.items():
        if native not in df.columns:
            raise MissingInput(f"TAIX master has no column {native!r}; the release layout changed "
                               f"and the label map in config must be updated before this runs.")
        out[canonical] = df[native].map(
            lambda x: binarize_presence(x, positive_code=1, negative_codes=(0,)))
    return out


def _side_resolved_labels(df: pd.DataFrame, findings: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for f in findings:
        for side in ("right", "left"):
            col = f"{f}_{side}_binary"
            if col in df.columns:
                    out[f"side__{f}__{side}"] = df[col].map(
                    lambda x: binarize_presence(x, positive_code=1, negative_codes=(0,)))
    return out


def _graded_columns(df: pd.DataFrame, graded: List[str]) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for g in graded:
        if g in df.columns:
            out[f"grade__{g}"] = pd.to_numeric(df[g], errors="coerce")
    return out


def main_build_taix_pool(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    tx = cfg["taix"]
    out_csv = tx["pool_manifest_csv"]
    status = status_path(cfg, "taix_pool")
    if not tx.get("enabled", True):
        raise MissingInput("taix.enabled is false; enable it in config before running main_build_taix_pool.")

    params = {"master": fingerprint_file(tx["master_csv"]),
              "label_map": tx["binary_label_map"],
              "graded": list(tx["graded_columns"]),
              "side_resolved": list(tx["side_resolved"])}
    if os.path.exists(out_csv) and not force and check_build_params(out_csv, params,
                                                                    owner="build_taix_pool"):
        return out_csv

    if not os.path.exists(tx["master_csv"]):
        raise MissingInput(f"TAIX master list not found at {tx['master_csv']}; it is built by the "
                           f"taix_preprocessing project and is not produced by this pipeline.")
    df = read_csv_defensively(tx["master_csv"])

    keep = ["image_id", "img_rel_path", "subject_id", tx["rater_col"], "split",
            "age", "gender", "view", "study_date"]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise MissingInput(f"TAIX master is missing {missing}; the release layout changed.")

    pool = df[keep].copy()
    pool = pool.rename(columns={"img_rel_path": "image_key", "gender": "sex",
                                tx["rater_col"]: "reader_id"})
    pool.insert(0, "case_id", "taix__" + pool["image_id"].astype(str))
    pool.insert(1, "dataset", "taix")
    pool.insert(2, "modality", "cxr")
    pool["split"] = pool["split"].astype(str).str.lower().replace({"val": "valid"})

    pool = pd.concat([pool.reset_index(drop=True),
                      _decode_binary_labels(df, tx["binary_label_map"]).reset_index(drop=True),
                      _side_resolved_labels(df, tx["side_resolved"]).reset_index(drop=True),
                      _graded_columns(df, tx["graded_columns"]).reset_index(drop=True)], axis=1)

    for native, canonical in tx["binary_label_map"].items():
        n_native = int((pd.to_numeric(df[native], errors="coerce") == 1.0).sum())
        n_pool = int((pool[canonical] == 1.0).sum())
        if n_pool != n_native:
            raise ValueError(f"[build_taix_pool] {canonical} has {n_pool} positives against {n_native} in "
                             f"{native}; a graded column has leaked into the canonical slot.")

    cap = int(tx.get("max_cases", 0))
    if cap and len(pool) > cap:
        rng = np.random.default_rng(int(cfg["seed"]))
        keep = []
        for split, grp in pool.groupby("split"):
            want = int(round(cap * len(grp) / len(pool)))
            pats = grp["subject_id"].astype(str).unique()
            rng.shuffle(pats)
            taken, chosen = 0, set()
            for pt in pats:
                n = int((grp["subject_id"].astype(str) == pt).sum())
                if taken + n > want:
                    continue
                chosen.add(pt); taken += n
                if taken >= want:
                    break
            keep.append(grp[grp["subject_id"].astype(str).isin(chosen)])
        pool = pd.concat(keep, ignore_index=True)
    params["max_cases"] = cap
    write_manifest(pool, out_csv, params, owner="taix_pool")
    prev = {c: round(float((pool[c] == 1.0).mean()) * 100, 2) for c in CANONICAL_FROM_TAIX
            if c in pool.columns}
    append_status(status, f"TAIX pool {len(pool)} rows, {pool.subject_id.nunique()} patients, "
                          f"{pool.reader_id.nunique()} readers, prevalence {prev}")
    return out_csv
