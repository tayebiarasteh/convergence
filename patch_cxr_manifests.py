"""
patch_cxr_manifests.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import numpy as np
import pandas as pd

from config.serde import read_config


def _patch_pool(pool_csv: str) -> None:
    if not os.path.exists(pool_csv):
        print(f"[patch] Pool manifest not found: {pool_csv} — skip.")
        return

    df      = pd.read_csv(pool_csv, low_memory=False)
    changed = {}

    if "sex" in df.columns and "dataset" in df.columns:
        m = df["dataset"] == "chexpert"
        n_f = int((df.loc[m, "sex"] == "Female").sum())
        n_m = int((df.loc[m, "sex"] == "Male").sum())
        if n_f or n_m:
            df.loc[m & (df["sex"] == "Female"), "sex"] = "F"
            df.loc[m & (df["sex"] == "Male"),   "sex"] = "M"
            changed["chexpert_sex"] = f"Female→F ({n_f}), Male→M ({n_m})"

    if "image_subdir" in df.columns and "dataset" in df.columns:
        m = df["dataset"] == "padchest"
        if m.any():
            def _int_str(v):
                if pd.isna(v):
                    return v
                try:
                    f = float(str(v))
                    return str(int(f)) if f == int(f) else str(v)
                except (ValueError, TypeError):
                    return str(v)

            before = df.loc[m, "image_subdir"].astype(str)
            df["image_subdir"] = df["image_subdir"].astype(object)
            df.loc[m, "image_subdir"] = df.loc[m, "image_subdir"].apply(_int_str)
            after  = df.loc[m, "image_subdir"].astype(str)
            n = int((before != after).sum())
            if n:
                changed["padchest_subdir"] = f"{n} rows (e.g. '1.0'→'1')"

    if "age" in df.columns and "dataset" in df.columns:
        m = df["dataset"] == "vindr_cxr"
        if m.any():
            n = int((df.loc[m, "age"] == 0.0).sum())
            if n:
                df.loc[m & (df["age"] == 0.0), "age"] = np.nan
                changed["vindr_cxr_age_zero"] = f"{n} rows → NaN"

    if changed:
        df.to_csv(pool_csv, index=False)
        print(f"  Saved {len(df)} rows.  Changes:")
        for k, v in changed.items():
            print(f"    {k}: {v}")
    else:
        print("  Already clean — no changes needed.")



def _patch_paired_reports(paired_csv: str, mimic_image_root: str) -> None:
    if not os.path.exists(paired_csv):
        print(f"[patch] Paired-reports manifest not found: {paired_csv} — skip.")
        return

    df = pd.read_csv(paired_csv, low_memory=False)

    if "report_path" not in df.columns:
        print("  No report_path column — skip.")
        return

    MARKER = "mimic-cxr-reports/"
    n_fixed = 0
    for i, val in df["report_path"].items():
        if pd.isna(val):
            continue
        s = str(val)
        if MARKER in s:
            rel     = s[s.index(MARKER):]
            correct = os.path.join(mimic_image_root, rel)
            if s != correct:
                df.at[i, "report_path"] = correct
                n_fixed += 1

    if n_fixed:
        df.to_csv(paired_csv, index=False)
        print(f"  Fixed {n_fixed} report_path values → prefix now {mimic_image_root}")
    else:
        print("  Already clean — no changes needed.")


def main_patch_cxr_manifests(global_config_path: str) -> None:
    cfg  = read_config(global_config_path)["Convergence"]
    pool_csv   = cfg["cxr"]["pool_manifest_csv"]
    paired_csv = cfg["cxr"]["paired_reports_csv"]
    mimic_root = cfg["cxr"]["sites"]["mimic"]["image_root"]

    _patch_pool(pool_csv)
    _patch_paired_reports(paired_csv, mimic_root)


if __name__ == "__main__":
    from main_convergence import GLOBAL_CONFIG_PATH
    main_patch_cxr_manifests(GLOBAL_CONFIG_PATH)