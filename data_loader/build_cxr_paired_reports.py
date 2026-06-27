"""
data_loader/build_cxr_paired_reports.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively


# Columns emitted for each source before union
_REPORT_COLS: List[str] = [
    "case_id", "dataset", "image_key", "split",
    "report_path", "report_text", "report_source",
]

_PLUS_SECTION_PAIRS = [
    ("section_findings", "section_impression"),
]
_PLUS_FALLBACK = "report"



def _collect_mimic_reports(
    pool: pd.DataFrame,
    mimic_scfg: dict,
) -> pd.DataFrame:
    image_root = mimic_scfg["image_root"]

    mimic_rows = pool[pool["dataset"] == "mimic"].copy()
    if mimic_rows.empty:
        return pd.DataFrame(columns=_REPORT_COLS)

    # Filter to rows with a valid report_rel_path
    has_report = (
        mimic_rows["report_rel_path"].notna() &
        (mimic_rows["report_rel_path"].astype(str).str.strip() != "") &
        (mimic_rows["report_rel_path"].astype(str).str.strip() != "nan")
    )
    mimic_rows = mimic_rows[has_report].copy()
    if mimic_rows.empty:
        return pd.DataFrame(columns=_REPORT_COLS)

    # Construct absolute report paths
    abs_paths = mimic_rows["report_rel_path"].apply(
        lambda rel: os.path.join(image_root, str(rel))
    )

    # Verify files exist; warn on missing
    exists_mask = abs_paths.apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing:
        print(f"MIMIC: {n_missing} report .txt files "
              f"not found on disk; these rows are excluded.")
    abs_paths = abs_paths[exists_mask]
    mimic_rows = mimic_rows[exists_mask]

    out = pd.DataFrame({
        "case_id":       mimic_rows["case_id"].values,
        "dataset":       "mimic",
        "image_key":     mimic_rows["image_key"].values,
        "split":         mimic_rows["split"].values,
        "report_path":   abs_paths.values,
        "report_text":   np.nan,
        "report_source": "mimic_report_file",
    })

    return out.reset_index(drop=True)



def _combine_sections(row: pd.Series, section_cols: List[str]) -> Optional[str]:
    parts = []
    for col in section_cols:
        val = str(row.get(col, "") or "").strip()
        if val and val.lower() not in ("nan", "none", ""):
            parts.append(val)
    return "\n".join(parts) if parts else None


def _collect_chexpert_reports(
    pool: pd.DataFrame,
    chexpert_scfg: dict,
) -> pd.DataFrame:
    plus_csv = chexpert_scfg.get("chexpert_plus_csv")
    if not plus_csv or not os.path.exists(plus_csv):
        print("CheXpert Plus CSV not found; "
              "no CheXpert report rows will be produced. "
              "Set Convergence.cxr.sites.chexpert.chexpert_plus_csv.")
        return pd.DataFrame(columns=_REPORT_COLS)

    chexpert_rows = pool[pool["dataset"] == "chexpert"].copy()
    if chexpert_rows.empty:
        return pd.DataFrame(columns=_REPORT_COLS)

    join_col = chexpert_scfg.get("report_join_col", "jpg_rel_path")

    # Load only the columns we need from CheXpert Plus (it is wide and large)
    section_cols_flat = [c for pair in _PLUS_SECTION_PAIRS for c in pair]
    usecols = [join_col] + section_cols_flat + [_PLUS_FALLBACK]
    usecols = [c for c in usecols if c]   # drop None if any
    plus = read_csv_defensively(plus_csv, usecols=usecols)

    # Deduplicate on join_col: keep first occurrence (earliest study order)
    plus = plus.drop_duplicates(subset=[join_col]).reset_index(drop=True)

    # Join to the pool CheXpert rows on image_key == jpg_rel_path
    merged = chexpert_rows[["case_id", "image_key", "split"]].merge(
        plus.rename(columns={join_col: "image_key"}),
        on="image_key",
        how="inner",
    )

    n_joined = len(merged)
    n_pool   = len(chexpert_rows)
    if n_joined < n_pool:
        print(f"CheXpert Plus join: "
              f"{n_joined}/{n_pool} pool rows matched.")

    if merged.empty:
        return pd.DataFrame(columns=_REPORT_COLS)

    # Build report_text: section pairs first, fallback to full report
    texts: List[Optional[str]] = []
    sources: List[str] = []
    for _, row in merged.iterrows():
        text = None
        for pair in _PLUS_SECTION_PAIRS:
            text = _combine_sections(row, list(pair))
            if text:
                sources.append("+".join(pair))
                break
        if not text:
            fb = str(row.get(_PLUS_FALLBACK, "") or "").strip()
            text = fb if fb and fb.lower() not in ("nan", "none", "") else None
            sources.append(_PLUS_FALLBACK if text else "none")
        texts.append(text)

    merged["report_text"]   = texts
    merged["report_source"] = sources

    # Drop rows with no usable text
    merged = merged[merged["report_text"].notna()].copy()

    out = pd.DataFrame({
        "case_id":       merged["case_id"].values,
        "dataset":       "chexpert",
        "image_key":     merged["image_key"].values,
        "split":         merged["split"].values,
        "report_path":   np.nan,
        "report_text":   merged["report_text"].values,
        "report_source": merged["report_source"].values,
    })

    return out.reset_index(drop=True)



def main_build_cxr_paired_reports(global_config_path: str) -> str:
    """Build the paired image-report manifest and return its path.

    Expects build_cxr_pool to have already run; reads the pool manifest rather
    than re-loading six master lists."""
    params     = read_config(global_config_path)
    cfg        = params["Convergence"]
    cfg_cxr    = cfg["cxr"]
    pool_csv   = cfg_cxr["pool_manifest_csv"]
    out_csv    = cfg_cxr["paired_reports_csv"]

    if not os.path.exists(pool_csv):
        raise FileNotFoundError(
            f"[build_cxr_paired_reports] Pool manifest not found: {pool_csv}. "
            f"Run build_cxr_pool first."
        )

    pool = read_csv_defensively(pool_csv, usecols=[
        "case_id", "dataset", "image_key", "split", "report_rel_path",
    ])

    mimic_scfg    = cfg_cxr["sites"]["mimic"]
    chexpert_scfg = cfg_cxr["sites"]["chexpert"]

    parts: List[pd.DataFrame] = []

    # Check which report sources are enabled in config
    report_sources = cfg_cxr.get("reports", {}).get("sources", ["mimic", "chexpert_plus"])

    if "mimic" in report_sources:
        mimic_df = _collect_mimic_reports(pool, mimic_scfg)
        if not mimic_df.empty:
            parts.append(mimic_df)

    if "chexpert_plus" in report_sources:
        chexpert_df = _collect_chexpert_reports(pool, chexpert_scfg)
        if not chexpert_df.empty:
            parts.append(chexpert_df)

    if not parts:
        raise RuntimeError("No report rows produced. "
                           "Check report source configs and CSV paths.")

    paired = pd.concat(parts, ignore_index=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    paired.to_csv(out_csv, index=False)

    for src, grp in paired.groupby("dataset"):
        print(f"    {src}: {len(grp)}")
    return out_csv

