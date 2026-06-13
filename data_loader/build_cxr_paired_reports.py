"""
data_loader/build_cxr_paired_reports.py

Builds the paired image-report manifest for the E2 vision-to-language
convergence experiment.

Report sources:
  MIMIC    Each study has a companion .txt radiology report on disk. The report
           path is stored in the CXR pool manifest under report_rel_path. This
           builder reads those paths (resolving against the MIMIC image_root)
           but does NOT read the file text at build time: the manifest stores
           the absolute report path, and the text-embedding stage reads the
           text at inference time. This avoids materializing potentially 100+MB
           of text into the CSV.

  CheXpert Plus provides inline report text per study joined on jpg_rel_path.
           This builder reads the cleaned sections (section_findings +
           section_impression). If both are empty it falls back to the full
           report column. The text is stored inline in report_text because it
           is already in memory and avoids a separate file-read step at
           inference.

Output schema:
  case_id        joins to the CXR pool manifest
  dataset        mimic | chexpert
  image_key      the raw path token (for join / audit)
  split
  report_path    absolute path to the .txt file (MIMIC only; NaN for CheXpert)
  report_text    inline report text (CheXpert Plus only; NaN for MIMIC)
  report_source  which column / file the text came from

The text-embedding stage uses report_path when non-null and report_text
otherwise, so the two sources are interchangeable from the embedder's
perspective.

Run:
    python -m data_loader.build_cxr_paired_reports

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

# CheXpert Plus text sections to try in order (first non-empty pair wins)
_PLUS_SECTION_PAIRS = [
    ("section_findings", "section_impression"),
]
_PLUS_FALLBACK = "report"


# ----- MIMIC ----------------------------------------------------------------

def _collect_mimic_reports(
    pool: pd.DataFrame,
    mimic_scfg: dict,
) -> pd.DataFrame:
    """Extract MIMIC report manifest rows from the pool manifest.

    The pool manifest already carries report_rel_path (absolute MIMIC image
    root + report_rel_path = the .txt file). No file reading happens here;
    the absolute path is stored for the text-embedding stage.

    Only rows with a non-null, non-empty report_rel_path are included. MIMIC
    typically has ~100% coverage for frontal studies."""
    image_root = mimic_scfg["image_root"]

    mimic_rows = pool[pool["dataset"] == "mimic"].copy()
    if mimic_rows.empty:
        print("[build_cxr_paired_reports] No MIMIC rows in pool manifest; skipping.")
        return pd.DataFrame(columns=_REPORT_COLS)

    # Filter to rows with a valid report_rel_path
    has_report = (
        mimic_rows["report_rel_path"].notna() &
        (mimic_rows["report_rel_path"].astype(str).str.strip() != "") &
        (mimic_rows["report_rel_path"].astype(str).str.strip() != "nan")
    )
    mimic_rows = mimic_rows[has_report].copy()
    if mimic_rows.empty:
        print("[build_cxr_paired_reports] No MIMIC rows have a report_rel_path; "
              "check that MIMIC pool was built with report_rel_path populated.")
        return pd.DataFrame(columns=_REPORT_COLS)

    # Construct absolute report paths
    abs_paths = mimic_rows["report_rel_path"].apply(
        lambda rel: os.path.join(image_root, str(rel))
    )

    # Verify files exist; warn on missing
    exists_mask = abs_paths.apply(os.path.exists)
    n_missing = (~exists_mask).sum()
    if n_missing:
        print(f"[build_cxr_paired_reports] MIMIC: {n_missing} report .txt files "
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

    print(f"[build_cxr_paired_reports] MIMIC: {len(out)} paired report rows.")
    return out.reset_index(drop=True)


# ----- CheXpert Plus --------------------------------------------------------

def _combine_sections(row: pd.Series, section_cols: List[str]) -> Optional[str]:
    """Concatenate non-empty section columns, separated by newline.
    Returns None if all sections are empty or NaN."""
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
    """Extract CheXpert Plus inline report text for the pool's CheXpert rows.

    Joins CheXpert Plus on jpg_rel_path (stored as image_key in the pool).
    Uses section_findings + section_impression; falls back to the full report
    column when both sections are empty. Rows with no usable text are excluded.
    """
    plus_csv = chexpert_scfg.get("chexpert_plus_csv")
    if not plus_csv or not os.path.exists(plus_csv):
        print("[build_cxr_paired_reports] CheXpert Plus CSV not found; "
              "no CheXpert report rows will be produced. "
              "Set Convergence.cxr.sites.chexpert.chexpert_plus_csv.")
        return pd.DataFrame(columns=_REPORT_COLS)

    chexpert_rows = pool[pool["dataset"] == "chexpert"].copy()
    if chexpert_rows.empty:
        print("[build_cxr_paired_reports] No CheXpert rows in pool manifest; skipping.")
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
        print(f"[build_cxr_paired_reports] CheXpert Plus join: "
              f"{n_joined}/{n_pool} pool rows matched.")

    if merged.empty:
        print("[build_cxr_paired_reports] CheXpert Plus: no rows after join.")
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

    print(f"[build_cxr_paired_reports] CheXpert Plus: {len(out)} paired report rows.")
    return out.reset_index(drop=True)


# ----- Orchestrator ---------------------------------------------------------

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

    print(f"[build_cxr_paired_reports] Reading pool manifest from {pool_csv}")
    pool = read_csv_defensively(pool_csv, usecols=[
        "case_id", "dataset", "image_key", "split", "report_rel_path",
    ])
    print(f"  Pool: {len(pool)} rows | "
          f"MIMIC: {(pool['dataset']=='mimic').sum()} | "
          f"CheXpert: {(pool['dataset']=='chexpert').sum()}")

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
        raise RuntimeError("[build_cxr_paired_reports] No report rows produced. "
                           "Check report source configs and CSV paths.")

    paired = pd.concat(parts, ignore_index=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    paired.to_csv(out_csv, index=False)

    print(f"\n[build_cxr_paired_reports] Paired report manifest -> {out_csv}")
    print(f"  {len(paired)} total paired rows")
    for src, grp in paired.groupby("dataset"):
        print(f"    {src}: {len(grp)}")
    print(f"  report_path non-null: {paired['report_path'].notna().sum()}")
    print(f"  report_text non-null: {paired['report_text'].notna().sum()}")
    return out_csv


if __name__ == "__main__":
    main_build_cxr_paired_reports(
        "/home/homesOnMaster/sarasteh/Documents/Repositories/convergence/config/config.yaml"
    )
