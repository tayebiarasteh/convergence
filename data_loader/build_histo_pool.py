"""
data_loader/build_histo_pool.py
Created on June 15, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import glob
import os
from typing import List

import h5py
import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, cap_per_group, finalize_manifest,
    read_csv_defensively,
)
from Inference.resume_utils import write_csv_atomic, MissingInput, append_status, status_path
from data_loader.build_utils import manifest_exists_and_valid, write_manifest


_LABEL_COLS_HISTO = ["tumor_label"]
_META_COLS_HISTO  = ["tissue_class"]


def _build_pcam(hcfg: dict, pcfg: dict, cap: int, seed: int) -> pd.DataFrame:
    h5_dir    = pcfg["h5_dir"]
    split     = pcfg.get("split", "test")
    patch_dir = os.path.join(h5_dir, pcfg.get("patches_subdir", "patches"))

    y_path = os.path.join(h5_dir,
                          f"camelyonpatch_level_2_split_{split}_y.h5")
    if not os.path.exists(y_path):
        return pd.DataFrame()

    with h5py.File(y_path, "r") as fy:
        y = np.array(fy["y"]).reshape(-1).astype(int)

    rows: List[dict] = []
    missing = 0
    for idx, label in enumerate(y):
        fname    = f"pcam_{split}_{idx:05d}.png"
        abs_path = os.path.join(patch_dir, fname)
        if not os.path.exists(abs_path):
            missing += 1
            continue
        rows.append({
            "case_id":     f"pcam__{split}_{idx:05d}",
            "dataset":     "pcam",
            "modality":    "histo",
            "split":       split,
            "image_key":   os.path.join(pcfg.get("patches_subdir", "patches"), fname),
            "image_subdir": np.nan,
            "tissue_class": "tumor" if label == 1 else "non_tumor",
            "tumor_label":  float(label),
        })


    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = cap_per_group(df, "tumor_label", cap=cap, seed=seed)
    return df


def _build_nct_crc(hcfg: dict, ncfg: dict, cap: int, seed: int) -> pd.DataFrame:
    root        = ncfg["root"]
    train_sub   = ncfg.get("train_subdir", "NCT-CRC-HE-100K")
    val_sub     = ncfg.get("val_subdir",   "CRC-VAL-HE-7K")
    classes     = ncfg.get("classes",
                           ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"])

    rows: List[dict] = []
    for split_tag, subdir in [("train", train_sub), ("valid", val_sub)]:
        split_root = os.path.join(root, subdir)
        if not os.path.isdir(split_root):
            continue
        for cls in classes:
            cls_dir = os.path.join(split_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            files = glob.glob(os.path.join(cls_dir, "*.tif"))
            for fpath in files:
                rel = os.path.relpath(fpath, root)
                stem = os.path.splitext(os.path.basename(fpath))[0]
                rows.append({
                    "case_id":      f"nct_crc__{split_tag}_{cls}_{stem}",
                    "dataset":      "nct_crc",
                    "modality":     "histo",
                    "split":        split_tag,
                    "image_key":    rel,
                    "image_subdir": np.nan,
                    "tissue_class": cls,
                    "tumor_label":  1.0 if cls == "TUM" else 0.0,
                })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df = cap_per_group(df, "tissue_class", cap=cap, seed=seed)
    return df


def main_build_histo_pool(global_config_path: str) -> str:
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    hcfg   = cfg["histo"]
    seed   = int(cfg.get("seed", 42))
    cap    = int(hcfg.get("cases_per_class", 2000))
    out_csv       = hcfg["pool_manifest_csv"]
    _expected = {"source": "histo", "seed": int(cfg.get("seed", 42))}
    if manifest_exists_and_valid(out_csv, _expected, owner="histo_pool"):
        return out_csv
    labels_csv    = hcfg["nct_crc_labels_csv"]


    parts: List[pd.DataFrame] = []

    if hcfg["pcam"].get("enabled", True):
        pcam_df = _build_pcam(hcfg, hcfg["pcam"], cap, seed)
        if not pcam_df.empty:
            parts.append(pcam_df)

    if hcfg["nct_crc"].get("enabled", True):
        nct_df = _build_nct_crc(hcfg, hcfg["nct_crc"], cap, seed)
        if not nct_df.empty:
            parts.append(nct_df)
            nct_labels = nct_df[["case_id", "image_key", "tissue_class",
                                   "split"]].copy()
            os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
            write_csv_atomic(nct_labels, labels_csv)

    if not parts:
        raise MissingInput("[build_histo_pool] no histopathology source produced rows; check that "
                           "the NCT-CRC and PCam sources are present.")

    pool = pd.concat(parts, ignore_index=True)
    pool = finalize_manifest(pool,
                             label_cols=_LABEL_COLS_HISTO,
                             meta_cols=_META_COLS_HISTO)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv_atomic(pool, out_csv)
    return out_csv
