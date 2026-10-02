"""
data_loader/build_fundus_pool.py
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
    assert_unique_case_ids, balanced_presence_sample,
    finalize_manifest, read_csv_defensively,
)
from Inference.resume_utils import write_csv_atomic, MissingInput, append_status, status_path
from data_loader.build_utils import manifest_exists_and_valid, write_manifest


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
        return pd.DataFrame()

    df = read_csv_defensively(master_csv)
    df["_label"] = (pd.to_numeric(df[label_col], errors="coerce") >= thr).astype(float, copy=False)
    df["_grade"] = pd.to_numeric(df[label_col], errors="coerce")

    def _path(row):
        fname = f"{row[id_col]}.png"
        return _resolve_fundus_path(image_root, source_tag, fname)

    exists = df.apply(lambda r: os.path.exists(_path(r)), axis=1)
    n_miss = (~exists).sum()
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
        return pd.DataFrame()

    df = read_csv_defensively(master_csv)

    if gradable_col in df.columns:
        before = len(df)
        df = df[df[gradable_col] == 1].copy()

    df["_label"] = (pd.to_numeric(df[label_col], errors="coerce") >= thr).astype(float, copy=False)
    df["_grade"] = pd.to_numeric(df[label_col], errors="coerce")

    def _path(row):
        return _resolve_fundus_path(image_root, source_tag, str(row[id_col]))

    exists = df.apply(lambda r: os.path.exists(_path(r)), axis=1)
    n_miss = (~exists).sum()
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
    return rows.reset_index(drop=True)


def main_build_fundus_pool(global_config_path: str) -> str:
    params    = read_config(global_config_path)
    cfg       = params["Convergence"]
    fcfg      = cfg["fundus"]

    if not fcfg.get("enabled", True):
        return ""

    image_root = fcfg["image_root"]
    out_csv    = fcfg["pool_manifest_csv"]
    _expected = {"source": "fundus", "seed": int(cfg.get("seed", 42))}
    if manifest_exists_and_valid(out_csv, _expected, owner="fundus_pool"):
        return out_csv
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
        raise MissingInput("[build_fundus_pool] no fundus source produced rows; check that the "
                           "APTOS and Messidor master lists are present.")

    pool = pd.concat(parts, ignore_index=True)
    pool = finalize_manifest(pool,
                             label_cols=_LABEL_COLS_FUNDUS,
                             meta_cols=_META_COLS_FUNDUS)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    write_csv_atomic(pool, out_csv)
    return out_csv
