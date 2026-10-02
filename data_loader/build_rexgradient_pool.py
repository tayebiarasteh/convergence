"""
data_loader/build_rexgradient_pool.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import numpy as np
import pandas as pd

from Inference.resume_utils import (MissingInput, append_status, check_build_params,
                                    fingerprint_file, status_path, write_build_params,
                                    write_csv_atomic)
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively, write_manifest

import warnings
warnings.filterwarnings("ignore")


def main_build_rexgradient_pool(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    rx = cfg["rexgradient"]
    out_csv = rx["pool_manifest_csv"]
    status = status_path(cfg, "rexgradient_pool")
    if not rx.get("enabled", True):
        raise MissingInput("rexgradient.enabled is false; enable it before running main_build_rexgradient_pool.")

    params = {"master": fingerprint_file(rx["master_csv"]), "max_cases": int(rx["max_cases"]),
              "report_cols": list(rx["report_cols"])}
    if os.path.exists(out_csv) and not force and check_build_params(out_csv, params,
                                                                    owner="build_rexgradient_pool"):
        return out_csv

    if not os.path.exists(rx["master_csv"]):
        raise MissingInput(f"ReXGradient master list not found at {rx['master_csv']}; it is a "
                           f"gated download and is not produced by this pipeline.")
    df = read_csv_defensively(rx["master_csv"])
    need = ["ImagePath", "split", rx["patient_col"], rx["institution_col"],
            rx["manufacturer_col"], rx["view_col"]]
    report_cols = [c for c in rx["report_cols"] if c in df.columns]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise MissingInput(f"ReXGradient master is missing {missing}; the release layout changed.")

    n0 = len(df)
    if "ImageModality" in df.columns:
        df = df[df["ImageModality"].astype(str).isin(["CR", "DX", "DR"])]
    df = df[df[rx["view_col"]].astype(str).isin(["AP", "PA"])]

    pool = df[need + report_cols].copy().rename(columns={
        "ImagePath": "image_key", rx["patient_col"]: "subject_id",
        rx["institution_col"]: "institution", rx["manufacturer_col"]: "manufacturer",
        rx["view_col"]: "view"})
    pool["image_key"] = pool["image_key"].astype(str).str.replace(r"^(\.\./)+", "", regex=True)
    pool.insert(0, "case_id", "rexgradient__" + np.arange(len(pool)).astype(str))
    pool.insert(1, "dataset", "rexgradient")
    pool.insert(2, "modality", "cxr")
    pool["split"] = pool["split"].astype(str).str.lower().replace({"val": "valid"})
    if report_cols:
        pool["report_text"] = (df[report_cols].fillna("").astype(str)
                               .agg(" ".join, axis=1).str.strip())
        pool = pool.drop(columns=report_cols)
    pool = pool[pool["institution"].notna() & pool["manufacturer"].notna()].reset_index(drop=True)
    import ast

    def _plain(v: str) -> str:
        t = str(v).strip()
        if t.startswith("[") and t.endswith("]"):
            try:
                items = [str(x).strip() for x in ast.literal_eval(t)]
            except (ValueError, SyntaxError):
                items = [t.strip("[]").replace("'", "").replace('"', "").strip()]
            return items[0] if len(items) == 1 else "mixed: " + " + ".join(sorted(items))
        return t
    for col in ("manufacturer", "institution"):
        pool[col] = pool[col].map(_plain)

    cap = int(rx["max_cases"])
    if len(pool) > cap:
        rng = np.random.default_rng(int(cfg["seed"]))
        pats = pool["subject_id"].dropna().unique()
        rng.shuffle(pats)
        keep, taken = set(), 0
        for pt in pats:
            n = int((pool["subject_id"] == pt).sum())
            if taken + n > cap:
                continue
            keep.add(pt); taken += n
            if taken >= cap:
                break
        pool = pool[pool["subject_id"].isin(keep)].reset_index(drop=True)

    write_manifest(pool, out_csv, params, owner="rexgradient_pool")
    msg = (f"ReXGradient pool {len(pool)} rows, {pool.subject_id.nunique()} patients, "
           f"{pool.institution.nunique()} institutions, {pool.manufacturer.nunique()} manufacturers, "
           f"{int(pool.get('report_text', pd.Series(dtype=str)).astype(str).str.len().gt(0).sum())} with report text")
    append_status(status, msg)
    return out_csv
