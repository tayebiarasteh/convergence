"""
data_loader/preprocess_histo_pcam.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Optional

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from config.serde import read_config


def _select_indices(
    y: np.ndarray,
    cap_per_class: Optional[int],
    seed: int,
) -> np.ndarray:
    if cap_per_class is None:
        return np.arange(len(y))
    rng = np.random.default_rng(int(seed))
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    n = min(int(cap_per_class), len(pos), len(neg))
    sel = np.concatenate([
        rng.choice(pos, size=n, replace=False),
        rng.choice(neg, size=n, replace=False),
    ])
    return np.sort(sel)   # sorted for sequential H5 read performance


def main_preprocess_pcam(global_config_path: str):
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    hcfg   = cfg["histo"]
    pcfg   = hcfg["pcam"]

    if not pcfg.get("enabled", True):
        print("[preprocess_pcam] PCam disabled in config; nothing to do.")
        return

    seed       = int(cfg.get("seed", 42))
    out_res    = int(hcfg.get("output_resolution", 224))
    split      = pcfg.get("split", "test")
    h5_dir     = pcfg["h5_dir"]
    patch_dir  = os.path.join(h5_dir, pcfg.get("patches_subdir", "patches"))
    cap        = pcfg.get("extract_cap_per_class", None)   # null in config -> all

    x_path = os.path.join(h5_dir, f"camelyonpatch_level_2_split_{split}_x.h5")
    y_path = os.path.join(h5_dir, f"camelyonpatch_level_2_split_{split}_y.h5")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        print(f"[preprocess_pcam] missing H5 files in {h5_dir}; expected "
              f"camelyonpatch_level_2_split_{split}_x.h5 and _y.h5.")
        return

    os.makedirs(patch_dir, exist_ok=True)

    with h5py.File(y_path, "r") as fy:
        y = np.array(fy["y"]).reshape(-1).astype(int)
    indices = _select_indices(y, cap, seed)
    print(f"[preprocess_pcam] {len(y):,} patches in split '{split}'; "
          f"extracting {len(indices):,} to {patch_dir} at {out_res}px.")

    written, skipped, errors = 0, 0, 0
    with h5py.File(x_path, "r") as fx:
        X = fx["x"]   # (N, 96, 96, 3) uint8
        for i in tqdm(indices.tolist(), unit="patch"):
            out_path = os.path.join(patch_dir, f"pcam_{split}_{i:05d}.png")
            if os.path.exists(out_path):
                skipped += 1
                continue
            try:
                arr = np.array(X[i]).astype(np.uint8)
                img = Image.fromarray(arr)
                if out_res != arr.shape[0]:
                    img = img.resize((out_res, out_res), Image.LANCZOS)
                img.save(out_path)
                written += 1
            except (OSError, ValueError) as e:
                errors += 1
                if errors <= 10:
                    print(f"[preprocess_pcam] error at idx {i}: {e}")
