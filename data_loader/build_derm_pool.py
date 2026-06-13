"""
data_loader/build_derm_pool.py

Builds the dermatology embedding-pool manifest from ISIC-2019 for the
discriminant control experiment.

Role in the paper: derm images fed through general encoders (DINOv3, CLIP,
SigLIP) measure the cross-modality alignment floor. The core discriminant
claim is that general encoders converge less across unrelated modality pairs
(CXR-derm, mammo-derm) than specialist encoders converge within a modality
(CXR specialists, pathology specialists). Derm and mammo together anchor this
floor without requiring a specialist encoder for either.

ISIC-2019 has 8 lesion classes (MEL, NV, BCC, AK, BKL, DF, VASC, SCC).
The one-hot ground-truth CSV is read; UNK class is excluded. Each row
carries one-hot presence labels for all 8 classes (exactly one is 1 per row).

Run:
    python -m data_loader.build_derm_pool

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


def _resolve_derm_path(image_root: str, source_tag: str,
                        image_id: str, resolution: int = 224) -> str:
    res = "preprocessed224" if resolution == 224 else "preprocessed"
    return os.path.join(image_root, res, source_tag, f"{image_id}.jpg")


def main_build_derm_pool(global_config_path: str) -> str:
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    dcfg   = cfg["derm"]

    if not dcfg.get("enabled", True):
        print("[build_derm_pool] Derm disabled in config; nothing to do.")
        return ""

    image_root = dcfg["image_root"]
    out_csv    = dcfg["pool_manifest_csv"]
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
        raise FileNotFoundError(
            f"[build_derm_pool] ISIC-2019 ground truth CSV not found: {gt_csv}"
        )

    df = read_csv_defensively(gt_csv)
    # Drop UNK column if present
    df = df.drop(columns=["UNK"], errors="ignore")
    print(f"[build_derm_pool] ISIC-2019: {len(df)} rows loaded.")

    # Each row is one-hot; derive the primary class label for capping
    present_classes = [c for c in classes if c in df.columns]
    if not present_classes:
        raise KeyError(f"[build_derm_pool] No class columns found. "
                       f"Expected: {classes}. Got: {list(df.columns)}")

    df["_primary_class"] = df[present_classes].idxmax(axis=1)

    # Verify images exist at 224px
    def _exists(row):
        return os.path.exists(
            _resolve_derm_path(image_root, source_tag, str(row[id_col]))
        )

    exists_mask = df.apply(_exists, axis=1)
    n_miss = (~exists_mask).sum()
    if n_miss:
        print(f"[build_derm_pool] {n_miss} images not found at 224px; dropping.")
    df = df[exists_mask].copy()

    # Cap per class
    df = cap_per_group(df, "_primary_class", cap=cap, seed=seed)
    print(f"[build_derm_pool] {len(df)} rows after cap ({cap}/class).")

    # Build manifest rows: one row per image, binary presence cols for all classes
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
        rows_data[cname] = df[col].astype(float).values

    pool = pd.DataFrame(rows_data)
    pool = finalize_manifest(pool, label_cols=canonical_names)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pool.to_csv(out_csv, index=False)
    print(f"[build_derm_pool] {len(pool)} rows -> {out_csv}")
    for cls, grp in pool.groupby("dataset"):
        pass   # single dataset; class breakdown via primary col
    for cname in canonical_names:
        n_pos = int((pool[cname] == 1).sum())
        print(f"  {cname}: {n_pos} positive")
    return out_csv


if __name__ == "__main__":
    main_build_derm_pool(
        "/home/homesOnMaster/sarasteh/Documents/Repositories/convergence/config/config.yaml"
    )
