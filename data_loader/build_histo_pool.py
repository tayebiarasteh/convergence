"""
data_loader/build_histo_pool.py
Created on May 25, 2026

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
    new_rng, read_csv_defensively,
)


_LABEL_COLS_HISTO = ["tumor_label"]       # PCam: 1 = tumor, 0 = no tumor
_META_COLS_HISTO  = ["tissue_class"]      # NCT-CRC: class name



def _build_pcam(hcfg: dict, pcfg: dict, cap: int, seed: int) -> pd.DataFrame:
    h5_dir    = pcfg["h5_dir"]
    split     = pcfg.get("split", "test")
    patch_dir = os.path.join(h5_dir, pcfg.get("patches_subdir", "patches"))

    y_path = os.path.join(h5_dir,
                          f"camelyonpatch_level_2_split_{split}_y.h5")
    if not os.path.exists(y_path):
        print(f"[build_histo_pool/pcam] Label H5 not found: {y_path}; skipping PCam.")
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

    if missing:
        print(f"[build_histo_pool/pcam] {missing} PNG patches not found "
              f"(run preprocess_histo_pcam.py first); {len(rows)} rows kept.")

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Cap per class
    df = cap_per_group(df, "tumor_label", cap=cap, seed=seed)
    print(f"[build_histo_pool/pcam] {len(df)} rows "
          f"(pos={int((df['tumor_label']==1).sum())} / "
          f"neg={int((df['tumor_label']==0).sum())})")
    return df


def _build_nct_crc(hcfg: dict, ncfg: dict, cap: int, seed: int) -> pd.DataFrame:
    root        = ncfg["root"]
    train_sub   = ncfg.get("train_subdir", "NCT-CRC-HE-100K")
    val_sub     = ncfg.get("val_subdir",   "CRC-VAL-HE-7K")
    classes     = ncfg.get("classes",
                           ["ADI","BACK","DEB","LYM","MUC","MUS","NORM","STR","TUM"])

    rows: List[dict] = []
    for split_tag, subdir in [("train", train_sub), ("val", val_sub)]:
        split_root = os.path.join(root, subdir)
        if not os.path.isdir(split_root):
            print(f"[build_histo_pool/nct_crc] {split_root} not found; skipping {split_tag}.")
            continue
        for cls in classes:
            cls_dir = os.path.join(split_root, cls)
            if not os.path.isdir(cls_dir):
                continue
            # Glob for .tif files (native format, 224px, no preprocessing needed)
            files = glob.glob(os.path.join(cls_dir, "*.tif"))
            for fpath in files:
                rel = os.path.relpath(fpath, root)   # e.g. NCT-CRC-HE-100K/ADI/ADI-TCGA-XXXX.tif
                stem = os.path.splitext(os.path.basename(fpath))[0]
                rows.append({
                    "case_id":      f"nct_crc__{split_tag}_{cls}_{stem}",
                    "dataset":      "nct_crc",
                    "modality":     "histo",
                    "split":        split_tag,
                    "image_key":    rel,       # relative to root; loader prepends root
                    "image_subdir": np.nan,
                    "tissue_class": cls,
                    "tumor_label":  1.0 if cls == "TUM" else 0.0,
                })

    if not rows:
        print("[build_histo_pool/nct_crc] No TIF files found; check nct_crc.root path.")
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
            # Write standalone NCT-CRC labels file for E5 training loaders
            nct_labels = nct_df[["case_id", "image_key", "tissue_class",
                                   "split"]].copy()
            os.makedirs(os.path.dirname(labels_csv), exist_ok=True)
            nct_labels.to_csv(labels_csv, index=False)
            print(f"[build_histo_pool] NCT-CRC labels -> {labels_csv}")

    if not parts:
        raise RuntimeError("[build_histo_pool] No histo rows produced.")

    pool = pd.concat(parts, ignore_index=True)
    pool = finalize_manifest(pool,
                             label_cols=_LABEL_COLS_HISTO,
                             meta_cols=_META_COLS_HISTO)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pool.to_csv(out_csv, index=False)
    for ds, grp in pool.groupby("dataset"):
        print(f"  {ds}: {len(grp)} rows")
    return out_csv
