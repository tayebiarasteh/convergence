"""
data_loader/build_quilt_pool.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively, assert_unique_case_ids


def main_build_quilt_pool(global_config_path: str) -> str:
    cfg   = read_config(global_config_path)["Convergence"]
    qcfg  = cfg["quilt"]

    if not qcfg.get("enabled", False):
        print("[build_quilt_pool] Quilt disabled in config; nothing to do.")
        return ""

    lookup_csv = qcfg["lookup_csv"]
    if not os.path.exists(lookup_csv):
        raise FileNotFoundError(
            f"[build_quilt_pool] Quilt lookup CSV not found: {lookup_csv}. "
            f"Set Convergence.quilt.lookup_csv to your downloaded file."
        )

    root       = qcfg["root"]
    images_sub = qcfg.get("images_subdir", "")
    image_col  = qcfg.get("image_col", "image_path")
    caption_col= qcfg.get("caption_col", "caption")
    split_col  = qcfg.get("split_col", "split")
    want_split = qcfg.get("split", "train")
    cap        = qcfg.get("cases_cap", None)
    verify     = qcfg.get("verify_image_exists", True)

    df = read_csv_defensively(lookup_csv)

    # Validate the configured columns exist; fail clearly if not.
    for col, key in [(image_col, "image_col"), (caption_col, "caption_col")]:
        if col not in df.columns:
            raise KeyError(
                f"Column '{col}' (quilt.{key}) not in lookup. "
                f"Available columns: {list(df.columns)}. Fix the *_col keys in config."
            )

    # Filter to the requested split if a split column is present.
    if split_col and split_col in df.columns and want_split:
        before = len(df)
        df = df[df[split_col].astype(str) == str(want_split)].copy()
    else:
        print("no split filter applied (column absent or split null).")

    # Drop rows with empty image or caption.
    df = df[df[image_col].notna() & df[caption_col].notna()].copy()
    df = df[df[caption_col].astype(str).str.strip() != ""]

    if cap is not None and len(df) > int(cap):
        df = df.sample(n=int(cap), random_state=cfg.get("seed", 42)).reset_index(drop=True)
        print(f"[build_quilt_pool] capped to {cap} pairs.")

    rows = []
    missing = 0
    for _, r in df.iterrows():
        fname = str(r[image_col]).strip()
        # image_key is relative to root: <images_subdir>/<fname>
        image_key = os.path.join(images_sub, fname) if images_sub else fname
        if verify:
            full = os.path.join(root, image_key)
            if not os.path.exists(full):
                missing += 1
                continue
        # case_id: a stable id from the image filename stem
        stem = os.path.splitext(os.path.basename(fname))[0]
        rows.append({
            "case_id":      f"quilt__{stem}",
            "dataset":      "quilt",
            "modality":     "histo",
            "split":        want_split,
            "image_key":    image_key,
            "image_subdir": "",
            "report_text":  str(r[caption_col]).strip(),
        })

    if missing:
        print(f"[build_quilt_pool] {missing} rows skipped (image file not found on disk).")

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise RuntimeError(
            "[build_quilt_pool] No usable Quilt pairs were found. Check root, "
            "images_subdir, and that the downloaded images match the lookup."
        )

    # Drop duplicate case_ids (same image stem appearing twice), keep first.
    pool = pool.drop_duplicates(subset=["case_id"]).reset_index(drop=True)
    assert_unique_case_ids(pool)

    out_csv = qcfg["pool_manifest_csv"]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pool.to_csv(out_csv, index=False)
    return out_csv
