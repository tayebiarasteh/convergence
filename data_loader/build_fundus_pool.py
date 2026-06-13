"""
data_loader/build_fundus_pool.py

Builds the fundus embedding-pool manifest from APTOS-2019 and Messidor-2.

Both sources provide referable DR (diabetic retinopathy) grades. Binary
presence of referable DR (grade >= referable_dr_min) is stored as the
primary label. Images are already preprocessed to 224px under
preprocessed224/<source_tag>/. No pixel work is performed here.

The fundus pool anchors the cross-modality specialist convergence arm:
RETFound (fundus-specialist) is expected to converge with other medical
encoders on fundus content more than general encoders converge across
unrelated modalities (derm, mammo).

Run:
    python -m data_loader.build_fundus_pool

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, balanced_presence_sample,
    finalize_manifest, new_rng, read_csv_defensively,
)


_LABEL_COLS_FUNDUS = ["referable_dr"]
_META_COLS_FUNDUS  = ["dr_grade", "source_tag"]


def _resolve_fundus_path(image_root: str, source_tag: str,
                          image_key: str, resolution: int = 224) -> str:
    res = "preprocessed224" if resolution == 224 else "preprocessed"
    return os.path.join(image_root, res, source_tag, image_key)


def _build_aptos(fcfg: dict, scfg: dict, image_root: str,
                 cap: int, seed: int) -> pd.DataFrame:
    master_csv  = scfg["master_csv"]
    source_tag  = scfg.get("source_tag", "aptos")
    id_col      = scfg.get("id_col", "id_code")
    label_col   = scfg.get("label_col", "diagnosis")
    thr         = int(fcfg.get("referable_dr_min", 2))

    if not os.path.exists(master_csv):
        print(f"[build_fundus_pool/aptos] CSV not found: {master_csv}; skipping.")
        return pd.DataFrame()

    df = read_csv_defensively(master_csv)
    df["_label"] = (pd.to_numeric(df[label_col], errors="coerce") >= thr).astype(float)
    df["_grade"] = pd.to_numeric(df[label_col], errors="coerce")

    # Verify images exist at 224px
    def _path(row):
        fname = f"{row[id_col]}.png"
        return _resolve_fundus_path(image_root, source_tag, fname)

    exists = df.apply(lambda r: os.path.exists(_path(r)), axis=1)
    n_miss = (~exists).sum()
    if n_miss:
        print(f"[build_fundus_pool/aptos] {n_miss} images missing; dropping.")
    df = df[exists].copy()

    df = balanced_presence_sample(df, "_label", cap, seed)

    rows = pd.DataFrame({
        "case_id":      [f"fundus__aptos__{row[id_col]}" for _, row in df.iterrows()],
        "dataset":      "fundus_aptos",
        "modality":     "fundus",
        "split":        "train",
        "image_key":    df[id_col].apply(lambda x: f"{x}.png").values,
        "image_subdir": source_tag,
        "referable_dr": df["_label"].values,
        "dr_grade":     df["_grade"].values,
        "source_tag":   source_tag,
    })
    print(f"[build_fundus_pool/aptos] {len(rows)} rows.")
    return rows.reset_index(drop=True)


def _build_messidor(fcfg: dict, scfg: dict, image_root: str,
                    cap: int, seed: int) -> pd.DataFrame:
    master_csv   = scfg["master_csv"]
    source_tag   = scfg.get("source_tag", "messidor")
    id_col       = scfg.get("id_col", "image_id")
    label_col    = scfg.get("label_col", "adjudicated_dr_grade")
    gradable_col = scfg.get("gradable_col", "adjudicated_gradable")
    thr          = int(fcfg.get("referable_dr_min", 2))

    if not os.path.exists(master_csv):
        print(f"[build_fundus_pool/messidor] CSV not found: {master_csv}; skipping.")
        return pd.DataFrame()

    df = read_csv_defensively(master_csv)

    # Drop ungradable images
    if gradable_col in df.columns:
        before = len(df)
        df = df[df[gradable_col] == 1].copy()
        print(f"[build_fundus_pool/messidor] Dropped {before - len(df)} ungradable images.")

    df["_label"] = (pd.to_numeric(df[label_col], errors="coerce") >= thr).astype(float)
    df["_grade"] = pd.to_numeric(df[label_col], errors="coerce")

    # Verify images (image_id already includes extension for Messidor)
    def _path(row):
        return _resolve_fundus_path(image_root, source_tag, str(row[id_col]))

    exists = df.apply(lambda r: os.path.exists(_path(r)), axis=1)
    n_miss = (~exists).sum()
    if n_miss:
        print(f"[build_fundus_pool/messidor] {n_miss} images missing; dropping.")
    df = df[exists].copy()

    df = balanced_presence_sample(df, "_label", cap, seed)

    rows = pd.DataFrame({
        "case_id":      [f"fundus__messidor__{row[id_col]}" for _, row in df.iterrows()],
        "dataset":      "fundus_messidor",
        "modality":     "fundus",
        "split":        "test",
        "image_key":    df[id_col].astype(str).values,
        "image_subdir": source_tag,
        "referable_dr": df["_label"].values,
        "dr_grade":     df["_grade"].values,
        "source_tag":   source_tag,
    })
    print(f"[build_fundus_pool/messidor] {len(rows)} rows.")
    return rows.reset_index(drop=True)


def main_build_fundus_pool(global_config_path: str) -> str:
    params    = read_config(global_config_path)
    cfg       = params["Convergence"]
    fcfg      = cfg["fundus"]

    if not fcfg.get("enabled", True):
        print("[build_fundus_pool] Fundus disabled in config; nothing to do.")
        return ""

    image_root = fcfg["image_root"]
    out_csv    = fcfg["pool_manifest_csv"]
    cap        = int(fcfg.get("cases_per_source", 2000))
    seed       = int(cfg.get("seed", 42))

    parts: List[pd.DataFrame] = []
    sources = fcfg.get("sources", {})

    if "aptos" in sources:
        parts.append(_build_aptos(fcfg, sources["aptos"], image_root, cap, seed))
    if "messidor" in sources:
        parts.append(_build_messidor(fcfg, sources["messidor"], image_root, cap, seed))

    parts = [p for p in parts if not p.empty]
    if not parts:
        raise RuntimeError("[build_fundus_pool] No fundus rows produced.")

    pool = pd.concat(parts, ignore_index=True)
    pool = finalize_manifest(pool,
                             label_cols=_LABEL_COLS_FUNDUS,
                             meta_cols=_META_COLS_FUNDUS)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pool.to_csv(out_csv, index=False)
    print(f"[build_fundus_pool] {len(pool)} total rows -> {out_csv}")
    return out_csv


if __name__ == "__main__":
    main_build_fundus_pool(
        "/home/homesOnMaster/sarasteh/Documents/Repositories/convergence/config/config.yaml"
    )
