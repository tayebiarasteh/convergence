"""
data_loader/preprocess_histo_pcam.py
Created on June 15, 2026

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
from Inference.resume_utils import (MissingInput, append_status, cell_is_done, check_build_params,
                                    claim_cell, release_claim, write_done_flag,
                                    status_path, write_build_params, write_csv_atomic)


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
    return np.sort(sel)


def main_preprocess_pcam(global_config_path: str):
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    hcfg   = cfg["histo"]
    pcfg   = hcfg["pcam"]

    if not pcfg.get("enabled", True):
        return

    seed       = int(cfg.get("seed", 42))
    out_res    = int(hcfg.get("output_resolution", 224))
    split      = pcfg.get("split", "test")
    h5_dir     = pcfg["h5_dir"]
    patch_dir  = os.path.join(h5_dir, pcfg.get("patches_subdir", "patches"))
    cap        = pcfg.get("extract_cap_per_class", None)

    x_path = os.path.join(h5_dir, f"camelyonpatch_level_2_split_{split}_x.h5")
    y_path = os.path.join(h5_dir, f"camelyonpatch_level_2_split_{split}_y.h5")
    if not (os.path.exists(x_path) and os.path.exists(y_path)):
        raise MissingInput(
            f"[preprocess_pcam] the PCam H5 files are absent in {h5_dir}; expected "
            f"camelyonpatch_level_2_split_{split}_x.h5 and _y.h5.")

    os.makedirs(patch_dir, exist_ok=True)
    status = status_path(cfg, "preprocess_pcam")
    if cell_is_done(patch_dir):
        return
    if not claim_cell(patch_dir, "pcam", owner="preprocess_pcam"):
        return
    if cell_is_done(patch_dir):
        release_claim(patch_dir, "pcam")
        return
    stamp = os.path.join(patch_dir, ".pcam_build_params.json")
    build = {"split": split, "resolution": out_res, "cap": cap, "seed": seed}
    if not check_build_params(stamp, build, owner="preprocess_pcam"):
        pass

    with h5py.File(y_path, "r") as fy:
        y = np.array(fy["y"]).reshape(-1).astype(int)
    indices = _select_indices(y, cap, seed)

    written, skipped, errors = 0, 0, 0
    with h5py.File(x_path, "r") as fx:
        X = fx["x"]
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

    if errors:
        raise RuntimeError(f"[preprocess_pcam] {errors} patch(es) failed to render; the extraction "
                           f"is incomplete and no done flag is written.")
    write_build_params(stamp, build)
    write_done_flag(patch_dir, {"written": written, "skipped": skipped, **build})
    release_claim(patch_dir, "pcam")
    append_status(status, f"pcam patches written={written} skipped={skipped} errors={errors}")
