"""
data_loader/build_derm_pool.py
Created on June 13, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, cap_per_group, finalize_manifest,
    read_csv_defensively,
)
from Inference.resume_utils import write_csv_atomic, MissingInput, append_status, status_path
from data_loader.build_utils import manifest_exists_and_valid, write_manifest


def _resolve_derm_path(image_root: str, source_tag: str,
                        image_id: str, resolution: int = 224) -> str:
    res = "preprocessed224" if resolution == 224 else "preprocessed"
    return os.path.join(image_root, res, source_tag, f"{image_id}.jpg")


def main_build_derm_pool(global_config_path: str) -> str:
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    dcfg   = cfg["derm"]

    if not dcfg.get("enabled", True):
        return ""

    image_root = dcfg["image_root"]
    out_csv    = dcfg["pool_manifest_csv"]
    _expected = {"source": "derm", "seed": int(cfg.get("seed", 42))}
    if manifest_exists_and_valid(out_csv, _expected, owner="derm_pool"):
        return out_csv
    cap        = int(dcfg.get("cases_per_class", 1000))
    seed       = int(cfg.get("seed", 42))

    sources    = dcfg.get("sources", {})
    isic_cfg   = sources.get("isic2019", {})

    gt_csv     = isic_cfg["gt_csv"]
    source_tag = isic_cfg.get("source_tag", "isic2019")
    id_col     = isic_cfg.get("id_col", "image")
    classes    = isic_cfg.get("classes",
                              ["MEL","NV","BCC","AK","BKL","DF","VASC","SCC"])
    class_map  = isic_cfg.get("class_map", {c: c.lower() for c in classes})

    if not os.path.exists(gt_csv):
        raise MissingInput(
            f"[build_derm_pool] ISIC-2019 ground truth CSV not found: {gt_csv}"
        )

    df = read_csv_defensively(gt_csv)
    df = df.drop(columns=["UNK"], errors="ignore")

    present_classes = [c for c in classes if c in df.columns]
    if not present_classes:
        raise KeyError(f"[build_derm_pool] No class columns found. "
                       f"Expected: {classes}. Got: {list(df.columns)}")

    df["_primary_class"] = df[present_classes].idxmax(axis=1)

    def _exists(row):
        return os.path.exists(
            _resolve_derm_path(image_root, source_tag, str(row[id_col]))
        )

    exists_mask = df.apply(_exists, axis=1)
    n_miss = (~exists_mask).sum()
    df = df[exists_mask].copy()

    df = cap_per_group(df, "_primary_class", cap=cap, seed=seed)

    canonical_names = [class_map.get(c, c.lower()) for c in present_classes]
    rows_data: dict = {
        "case_id":      [f"derm__isic2019__{str(row[id_col])}"
                         for _, row in df.iterrows()],
        "dataset":      "derm_isic2019",
        "modality":     "derm",
        "split":        "train",
        "image_key":    df[id_col].astype(str).values,
        "image_subdir": source_tag,
    }
    for col, cname in zip(present_classes, canonical_names):
        rows_data[cname] = df[col].astype(float, copy=False).values

    pool = pd.DataFrame(rows_data)
    pool = finalize_manifest(pool, label_cols=canonical_names)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv_atomic(pool, out_csv)
    for cls, grp in pool.groupby("dataset"):
        pass
    for cname in canonical_names:
        n_pos = int((pool[cname] == 1).sum())
    return out_csv
