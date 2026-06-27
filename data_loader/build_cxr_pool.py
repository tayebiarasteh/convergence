"""
data_loader/build_cxr_pool.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, finalize_manifest, read_csv_defensively,
)
from data_loader.cxr_harmonization import (
    CANONICAL_CXR_FINDINGS, CXR_SITES, EXTENDED_MAPS, IMAGE_KEY_COL,
    IMAGE_SUBDIR_COL, LABEL_MAPS, LABEL_POLICY, VIEW_COL, VIEW_KEEP,
    resolve_cxr_image_path,
)


_META_COLS: List[str] = [
    "subject_id", "study_id", "age", "sex",
    "race", "ethnicity", "insurance",
    "view", "report_rel_path",
]



def _binarize_series(
    s: pd.Series,
    positive: int,
    negatives: set,
    excludes: set,
) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(np.nan, index=s.index, dtype=float)
    out[num == positive] = 1.0
    for nc in negatives:
        out[(out.isna()) & (num == nc)] = 0.0
    # exclude_codes remain NaN (already initialised to NaN)
    return out


def _decode_canonical_labels(site: str, df: pd.DataFrame) -> pd.DataFrame:
    policy = LABEL_POLICY[site]
    pos = int(policy["positive_code"])
    negs = {int(c) for c in policy["negative_codes"]}
    excl = {int(c) for c in policy["exclude_codes"]}

    result: Dict[str, pd.Series] = {}
    for canonical in CANONICAL_CXR_FINDINGS:
        native_cols = [nc for nc, c in LABEL_MAPS[site].items()
                       if c == canonical and nc in df.columns]
        if not native_cols:
            result[canonical] = pd.Series(np.nan, index=df.index, dtype=float)
            continue
        decoded = [_binarize_series(df[nc], pos, negs, excl) for nc in native_cols]
        if len(decoded) == 1:
            result[canonical] = decoded[0]
        else:
            result[canonical] = pd.concat(decoded, axis=1).max(axis=1)

    return pd.DataFrame(result, index=df.index)


def _decode_extended_labels(site: str, df: pd.DataFrame) -> pd.DataFrame:
    emap = EXTENDED_MAPS.get(site, {})
    if not emap:
        return pd.DataFrame(index=df.index)

    policy = LABEL_POLICY[site]
    pos = int(policy["positive_code"])
    negs = {int(c) for c in policy["negative_codes"]}

    result: Dict[str, pd.Series] = {}
    for native_col, ext_name in emap.items():
        if native_col not in df.columns:
            continue
        result[ext_name] = _binarize_series(df[native_col], pos, negs, set())

    return pd.DataFrame(result, index=df.index)



def _case_id(site: str, image_key: str) -> str:
    k = str(image_key).strip()
    if site == "mimic":
        # stem of the filename: 02aa804e-...-4e384014
        stem = k.rsplit("/", 1)[-1].replace(".jpg", "")
        return f"mimic__{stem}"
    if site == "chexpert":
        # strip 'CheXpert-v1.0/', normalize separators, strip extension
        rel = k.replace("CheXpert-v1.0/", "").replace("/", "_")
        if rel.lower().endswith(".jpg"):
            rel = rel[:-4]
        return f"chexpert__{rel}"
    if site == "nih_cxr14":
        stem = k.rsplit("/", 1)[-1]          # already short, e.g. 00000001_000.png
        stem = stem.rsplit(".", 1)[0]
        return f"nih_cxr14__{stem}"
    if site == "padchest":
        stem = k.rsplit(".", 1)[0] if "." in k else k
        return f"padchest__{stem}"
    # vindr_cxr, vindr_pcxr: image_id is already a short UUID
    return f"{site}__{k}"



def _apply_view_filter(df: pd.DataFrame, site: str) -> pd.DataFrame:
    keep = VIEW_KEEP[site]
    col = VIEW_COL[site]
    if keep is None or col is None or col not in df.columns:
        return df
    before = len(df)
    df = df[df[col].isin(keep)].copy()
    print(f"  [{site}] view filter ({keep}): {before} -> {len(df)} rows")
    return df



def _verify_images(
    df: pd.DataFrame,
    site: str,
    image_root: str,
    resolution: int = 224,
) -> pd.DataFrame:

    def _exists(row: pd.Series) -> bool:
        try:
            path = resolve_cxr_image_path(
                dataset=site,
                image_root=image_root,
                image_key=str(row["image_key"]),
                resolution=resolution,
                split=str(row["split"]) if "split" in row and str(row.get("split")) not in ("", "nan") else None,
                image_subdir=str(row["image_subdir"])
                if "image_subdir" in row and str(row.get("image_subdir")) not in ("", "nan")
                else None,
            )
            return os.path.exists(path)
        except (KeyError, TypeError):
            return False

    mask = df.apply(_exists, axis=1)
    n_missing = (~mask).sum()
    if n_missing:
        print(f"  [{site}] WARNING: {n_missing} images missing at {resolution}px; "
              f"dropping from manifest.")
    return df[mask].copy()



def _join_chexpert_plus(df: pd.DataFrame, plus_cfg: dict) -> pd.DataFrame:
    """Left-join race, ethnicity, insurance from CheXpert Plus onto the
    CheXpert master list. Joined on jpg_rel_path. Missing or non-matching rows
    receive NaN for the equity columns."""
    plus_csv = plus_cfg.get("chexpert_plus_csv")
    if not plus_csv or not os.path.exists(plus_csv):
        print("  [chexpert] CheXpert Plus CSV not found; race/ethnicity/insurance "
              "will be NaN. Set Convergence.cxr.sites.chexpert.chexpert_plus_csv.")
        df["race"] = np.nan
        df["ethnicity"] = np.nan
        df["insurance"] = np.nan
        return df

    join_col = plus_cfg.get("report_join_col", "jpg_rel_path")
    keep_cols = [join_col,
                 plus_cfg.get("race_col", "race"),
                 plus_cfg.get("ethnicity_col", "ethnicity"),
                 plus_cfg.get("insurance_col", "insurance_type")]
    plus = read_csv_defensively(plus_csv, usecols=keep_cols)
    plus = plus.rename(columns={
        plus_cfg.get("race_col", "race"): "race",
        plus_cfg.get("ethnicity_col", "ethnicity"): "ethnicity",
        plus_cfg.get("insurance_col", "insurance_type"): "insurance",
    })
    plus = plus.drop_duplicates(subset=[join_col])

    df = df.merge(plus[[join_col, "race", "ethnicity", "insurance"]],
                  on=join_col, how="left")
    n_matched = df["race"].notna().sum()
    print(f"  [chexpert] CheXpert Plus joined: {n_matched}/{len(df)} rows matched.")
    return df



def _extract_metadata(site: str, df: pd.DataFrame) -> pd.DataFrame:
    """Extract the standardized metadata columns from each site's master list.
    Always returns a DataFrame with exactly _META_COLS columns, with NaN where
    a site does not carry a field."""
    def _col(name: str, fallback=np.nan) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series(fallback, index=df.index)

    if site == "mimic":
        return pd.DataFrame({
            "subject_id":    _col("subject_id").astype(str),
            "study_id":      _col("study_id").astype(str),
            "age":           pd.to_numeric(_col("age"), errors="coerce"),
            "sex":           _col("gender"),
            "race":          np.nan,
            "ethnicity":     np.nan,
            "insurance":     np.nan,
            "view":          _col("view"),
            "report_rel_path": _col("report_rel_path"),
        }, index=df.index)

    if site == "chexpert":
        # CheXpert gender column uses 'Female'/'Male'; harmonize to 'F'/'M'
        sex_raw = _col("gender")
        sex_s   = sex_raw.map({"Female": "F", "Male": "M"}).where(
                      sex_raw.isin(["Female", "Male"]), sex_raw
                  )
        view_s = _col("AP_PA").fillna(_col("view"))
        return pd.DataFrame({
            "subject_id":    _col("subject_id").astype(str),
            "study_id":      np.nan,
            "age":           pd.to_numeric(_col("age"), errors="coerce"),
            "sex":           sex_s,
            "race":          _col("race"),
            "ethnicity":     _col("ethnicity"),
            "insurance":     _col("insurance"),
            "view":          view_s,
            "report_rel_path": np.nan,
        }, index=df.index)

    if site == "vindr_cxr":
        # age=0 is a known VinDr-CXR encoding for missing/unavailable age; treat as NaN
        age_raw = pd.to_numeric(_col("age"), errors="coerce")
        age_raw = age_raw.replace(0.0, np.nan)
        return pd.DataFrame({
            "subject_id":    np.nan,
            "study_id":      np.nan,
            "age":           age_raw,
            "sex":           _col("gender"),
            "race":          np.nan, "ethnicity": np.nan, "insurance": np.nan,
            "view":          np.nan,
            "report_rel_path": np.nan,
        }, index=df.index)

    if site == "nih_cxr14":
        return pd.DataFrame({
            "subject_id":    _col("patient_id").astype(str),
            "study_id":      np.nan,
            "age":           pd.to_numeric(_col("age"), errors="coerce"),
            "sex":           _col("gender"),
            "race":          np.nan, "ethnicity": np.nan, "insurance": np.nan,
            "view":          _col("view_position"),
            "report_rel_path": np.nan,
        }, index=df.index)

    if site == "padchest":
        return pd.DataFrame({
            "subject_id":    _col("PatientID").astype(str),
            "study_id":      _col("StudyID").astype(str),
            "age":           pd.to_numeric(_col("age"), errors="coerce"),
            "sex":           _col("gender"),
            "race":          np.nan, "ethnicity": np.nan, "insurance": np.nan,
            "view":          _col("view"),
            "report_rel_path": np.nan,
        }, index=df.index)

    if site == "vindr_pcxr":
        return pd.DataFrame({
            "subject_id":    np.nan, "study_id": np.nan,
            "age":           np.nan, "sex":       np.nan,
            "race":          np.nan, "ethnicity": np.nan, "insurance": np.nan,
            "view":          np.nan,
            "report_rel_path": np.nan,
        }, index=df.index)

    raise ValueError(f"unknown site '{site}'")



def _load_site(
    site: str,
    scfg: dict,
    cfg_cxr: dict,
    verify: bool,
    cap: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Load, filter, harmonize, and verify one CXR site into the pool schema.

    Returns a DataFrame with CORE_COLUMNS + _META_COLS + canonical + extended
    label columns, all on a fresh RangeIndex."""
    master_csv  = scfg["master_csv"]
    image_root  = scfg["image_root"]
    split_col   = scfg.get("split_col", "split")
    key_col     = IMAGE_KEY_COL[site]
    subdir_col  = IMAGE_SUBDIR_COL[site]

    print(f"\n[build_cxr_pool] Loading {site} from {master_csv}")
    df = read_csv_defensively(master_csv)
    print(f"  [{site}] master list: {len(df)} rows, {len(df.columns)} columns")

    # CheXpert Plus join (before view filter to preserve full join surface)
    if site == "chexpert":
        df = _join_chexpert_plus(df, scfg)

    # View filter
    df = _apply_view_filter(df, site)
    if df.empty:
        print(f"  [{site}] WARNING: empty after view filter.")
        return pd.DataFrame()

    # Optional per-site cap (config-level safety valve; null = no cap)
    if cap is not None and len(df) > cap:
        df = df.sample(n=cap, random_state=seed)
        print(f"  [{site}] capped to {cap} rows.")

    df = df.reset_index(drop=True)

    # Core columns
    key_series   = df[key_col].astype(str)
    def _subdir_str(v) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        s = str(v)
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
        except (ValueError, TypeError):
            pass
        return s

    subdir_series = df[subdir_col].apply(_subdir_str) if subdir_col and subdir_col in df.columns \
                    else pd.Series(np.nan, index=df.index)

    core = pd.DataFrame({
        "case_id":     [_case_id(site, k) for k in key_series],
        "dataset":     site,
        "modality":    "cxr",
        "split":       df[split_col].astype(str) if split_col in df.columns
                       else pd.Series("unknown", index=df.index),
        "image_key":   key_series,
        "image_subdir": subdir_series,
    }, index=df.index)

    # Metadata
    meta = _extract_metadata(site, df)

    # Labels
    canonical = _decode_canonical_labels(site, df)
    extended  = _decode_extended_labels(site, df)

    # Assemble
    out = pd.concat([core, meta, canonical, extended], axis=1)

    # Image existence check
    if verify:
        out = _verify_images(out, site, image_root)

    n_final = len(out)
    n_ext = len(extended.columns)
    print(f"  [{site}] done: {n_final} rows, {n_ext} extended-finding columns.")
    return out.reset_index(drop=True)



def main_build_cxr_pool(global_config_path: str) -> str:
    """Build the shared CXR embedding-pool manifest and return its path."""
    params   = read_config(global_config_path)
    cfg      = params["Convergence"]
    cfg_cxr  = cfg["cxr"]
    out_csv  = cfg_cxr["pool_manifest_csv"]
    verify   = bool(cfg_cxr.get("verify_image_exists", True))
    cap      = cfg_cxr.get("cap_per_site_view", None)
    seed     = int(cfg.get("seed", 42))

    site_dfs: List[pd.DataFrame] = []
    for site in CXR_SITES:
        scfg = cfg_cxr["sites"].get(site, {})
        if not scfg.get("enabled", True):
            print(f"\n[build_cxr_pool] {site} disabled; skipping.")
            continue
        site_df = _load_site(site, scfg, cfg_cxr, verify, cap, seed)
        if not site_df.empty:
            site_dfs.append(site_df)

    if not site_dfs:
        raise RuntimeError("[build_cxr_pool] No sites produced any rows.")

    # Concatenate and finalize
    pool = pd.concat(site_dfs, ignore_index=True)

    # Collect all extended finding names across sites in a stable order
    ext_names: List[str] = []
    seen: set = set()
    for emap in EXTENDED_MAPS.values():
        for name in emap.values():
            if name not in seen:
                ext_names.append(name)
                seen.add(name)

    label_cols = list(CANONICAL_CXR_FINDINGS) + sorted(ext_names)
    pool = finalize_manifest(pool, label_cols=label_cols, meta_cols=_META_COLS)

    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pool.to_csv(out_csv, index=False)

    n_sites = pool["dataset"].nunique()
    n_cases = len(pool)
    for site, grp in pool.groupby("dataset"):
        print(f"    {site}: {len(grp)}")
    return out_csv

