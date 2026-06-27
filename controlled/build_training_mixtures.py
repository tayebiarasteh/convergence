"""
controlled/build_training_mixtures.py
Created on June 1, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

import warnings
warnings.filterwarnings("ignore")


def main_build_training_mixtures(global_config_path: str) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    ctrl    = cfg["controlled"]
    out_dir = os.path.join(ctrl["ckpts_dir"], "mixtures")
    os.makedirs(out_dir, exist_ok=True)

    seed = int(cfg.get("seed", 42))
    rng  = np.random.default_rng(seed)

    _build_cxr_mixtures(cfg, ctrl, out_dir, rng)
    _build_histo_mixtures(cfg, ctrl, out_dir, rng)

    return out_dir


def _build_cxr_mixtures(cfg: dict, ctrl: dict, out_dir: str, rng):
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    if not os.path.exists(pool_csv):
        print("[build_mixtures/cxr] Pool manifest not found; skipping CXR mixtures.")
        return
    pool = read_csv_defensively(pool_csv)

    mimic_train = pool[
        (pool["dataset"] == "mimic") & (pool["split"] == "train")
    ].copy().reset_index(drop=True)

    ssl_df = mimic_train[["case_id", "image_key", "dataset", "split"]].copy()
    ssl_df.to_csv(os.path.join(out_dir, "cxr_ssl_train.csv"), index=False)

    findings = [f for f in CANONICAL_CXR_FINDINGS
                if f != "no_finding" and f in mimic_train.columns]
    # Keep only rows with at least one positive label
    has_label = mimic_train[findings].apply(
        lambda r: (pd.to_numeric(r, errors="coerce") == 1.0).any(), axis=1
    )
    sup_df = mimic_train[has_label].copy()
    cols   = ["case_id", "image_key", "dataset", "split"] + findings
    sup_df[cols].to_csv(os.path.join(out_dir, "cxr_supervised_train.csv"), index=False)

    if "report_rel_path" in mimic_train.columns:
        has_report = mimic_train["report_rel_path"].notna() & \
                     (mimic_train["report_rel_path"].astype(str).str.strip() != "") & \
                     (mimic_train["report_rel_path"].astype(str) != "nan")
        txt_df = mimic_train[has_report][
            ["case_id", "image_key", "dataset", "split", "report_rel_path"]
        ].copy()
        txt_df.to_csv(os.path.join(out_dir, "cxr_image_text_train.csv"), index=False)

    if "subject_id" in mimic_train.columns:
        subjects = mimic_train["subject_id"].dropna().unique()
        rng.shuffle(subjects)
        half = len(subjects) // 2
        half1_ids = set(subjects[:half].astype(str))
        half2_ids = set(subjects[half:].astype(str))
        h1 = mimic_train[mimic_train["subject_id"].astype(str).isin(half1_ids)]
        h2 = mimic_train[mimic_train["subject_id"].astype(str).isin(half2_ids)]
        h1[cols].to_csv(os.path.join(out_dir, "cxr_disjoint_half1_train.csv"), index=False)
        h2[cols].to_csv(os.path.join(out_dir, "cxr_disjoint_half2_train.csv"), index=False)


def _build_histo_mixtures(cfg: dict, ctrl: dict, out_dir: str, rng):
    histo_pool_csv = cfg["histo"]["pool_manifest_csv"]
    if not os.path.exists(histo_pool_csv):
        print("Histo pool manifest not found; skipping.")
        return
    histo = read_csv_defensively(histo_pool_csv)

    # NCT-CRC training split
    nct_train = histo[
        (histo["dataset"] == "nct_crc") & (histo["split"] == "train")
    ].copy()

    nct_train[["case_id", "image_key", "dataset", "split"]].to_csv(
        os.path.join(out_dir, "histo_ssl_train.csv"), index=False
    )

    nct_train[["case_id", "image_key", "dataset", "split", "tissue_class"]].to_csv(
        os.path.join(out_dir, "histo_supervised_train.csv"), index=False
    )

    quilt_csv = cfg["quilt"]["pool_manifest_csv"]
    if cfg["quilt"].get("enabled", False) and os.path.exists(quilt_csv):
        quilt = read_csv_defensively(quilt_csv)
        cols  = [c for c in ["case_id", "image_key", "dataset", "split", "report_text"]
                 if c in quilt.columns]
        quilt[cols].to_csv(
            os.path.join(out_dir, "histo_image_text_train.csv"), index=False
        )
    else:
        print("Quilt-1M manifest not found or disabled; "
              "image-text cell skipped.")