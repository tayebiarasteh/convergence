"""
data_loader/build_quilt_pool.py

Builds the Quilt-1M pool manifest for the E5 histopathology image-text arm.

Quilt-1M provides ~1M histopathology image-caption pairs mined from educational
sources. No image preprocessing is applied here: encoders' own processors resize
images at inference time. This builder reads the lookup CSV, verifies that the
image files exist on disk, applies an optional pair cap, and writes the manifest.

The manifest stores image_key as the path relative to Convergence.quilt.root so
the loader can resolve absolute paths portably. The report_text column holds the
caption inline.

Run:
    python -m data_loader.build_quilt_pool

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, read_csv_defensively,
)


def main_build_quilt_pool(global_config_path: str) -> str:
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    qcfg   = cfg["quilt"]

    if not qcfg.get("enabled", True):
        print("[build_quilt_pool] Quilt disabled in config; nothing to do.")
        return ""

    lookup_csv   = qcfg["lookup_csv"]
    root         = qcfg["root"]
    images_sub   = qcfg.get("images_subdir", "images")
    out_csv      = qcfg["pool_manifest_csv"]
    max_pairs    = qcfg.get("max_pairs", None)
    seed         = int(cfg.get("seed", 42))

    if not os.path.exists(lookup_csv):
        raise FileNotFoundError(
            f"[build_quilt_pool] Lookup CSV not found: {lookup_csv}"
        )

    print(f"[build_quilt_pool] Reading {lookup_csv}")
    lut = read_csv_defensively(lookup_csv)
    print(f"  {len(lut)} rows in lookup table | columns: {list(lut.columns)}")

    # Detect path and caption columns robustly (column names vary by Quilt release)
    path_candidates    = ["image_path", "path", "file_path", "filename", "image"]
    caption_candidates = ["caption", "text", "description", "label"]

    path_col = next((c for c in path_candidates if c in lut.columns), None)
    cap_col  = next((c for c in caption_candidates if c in lut.columns), None)

    if path_col is None:
        raise KeyError(
            f"[build_quilt_pool] Cannot find image path column in lookup CSV. "
            f"Available: {list(lut.columns)}. "
            f"Expected one of: {path_candidates}"
        )
    if cap_col is None:
        raise KeyError(
            f"[build_quilt_pool] Cannot find caption column in lookup CSV. "
            f"Available: {list(lut.columns)}. "
            f"Expected one of: {caption_candidates}"
        )

    print(f"  Using path_col='{path_col}', caption_col='{cap_col}'")

    # Drop rows with missing path or caption
    before = len(lut)
    lut = lut[lut[path_col].notna() & lut[cap_col].notna()].copy()
    print(f"  {before - len(lut)} rows dropped (missing path or caption); "
          f"{len(lut)} remain.")

    # Optional cap
    if max_pairs is not None and len(lut) > int(max_pairs):
        lut = lut.sample(n=int(max_pairs), random_state=seed)
        print(f"  Capped to {len(lut)} pairs.")

    # image_key: path relative to root (normalize so it's portable)
    def _to_rel(p: str) -> str:
        p = str(p).strip()
        # If the path already starts with root, strip it
        if p.startswith(root):
            p = os.path.relpath(p, root)
        # If it starts with images_sub, keep as-is
        return p

    lut["_image_key"] = lut[path_col].apply(_to_rel)

    # Verify existence (absolute path = root / image_key)
    def _exists(rel: str) -> bool:
        return os.path.exists(os.path.join(root, rel))

    exists_mask = lut["_image_key"].apply(_exists)
    n_miss = (~exists_mask).sum()
    if n_miss:
        print(f"  WARNING: {n_miss} image files not found under {root}; dropping.")
    lut = lut[exists_mask].copy()

    if lut.empty:
        raise RuntimeError("[build_quilt_pool] No valid Quilt pairs after filtering.")

    # Build manifest
    rows = pd.DataFrame({
        "case_id":      ["quilt__" + str(i) for i in lut.index],
        "dataset":      "quilt",
        "modality":     "histo",
        "split":        "train",
        "image_key":    lut["_image_key"].values,
        "image_subdir": np.nan,
        "report_text":  lut[cap_col].values,
    })

    rows = finalize_manifest(rows, meta_cols=["report_text"])
    assert_unique_case_ids(rows)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows.to_csv(out_csv, index=False)
    print(f"[build_quilt_pool] {len(rows)} pairs -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    main_build_quilt_pool(
        "/home/homesOnMaster/sarasteh/Documents/Repositories/convergence/config/config.yaml"
    )
