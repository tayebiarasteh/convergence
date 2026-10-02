"""
controlled/build_training_mixtures.py
Created on June 16, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List

import numpy as np
import pandas as pd

from config.serde import read_config
from Inference.resume_utils import write_csv_atomic
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

import warnings
from Inference.resume_utils import MissingInput, append_status, status_path
from data_loader.build_utils import manifest_exists_and_valid, write_manifest
warnings.filterwarnings("ignore")


def main_build_training_mixtures(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "mixtures")
    ctrl    = cfg["controlled"]
    out_dir = os.path.join(ctrl["ckpts_dir"], "mixtures")
    os.makedirs(out_dir, exist_ok=True)

    from Inference.resume_utils import output_is_current, record_sources
    pool_srcs = [cfg["cxr"]["pool_manifest_csv"], cfg["histo"]["pool_manifest_csv"]]
    if (cfg.get("granularity", {}) or {}).get("enabled", True):
        pool_srcs.append(cfg["taix"]["pool_manifest_csv"])
    outs = [os.path.join(out_dir, f"{m}_common_train.csv")
            for m in ("cxr", "histo", "cxrh1", "cxrh2")]
    if (cfg.get("granularity", {}) or {}).get("enabled", True):
        outs.append(os.path.join(out_dir, "taix_common_train.csv"))
    if not force and all(output_is_current(o, pool_srcs, owner="mixtures") for o in outs):
        return out_dir

    seed = int(cfg.get("seed", 42))
    rng  = np.random.default_rng(seed)

    if not bool(ctrl.get("common_intersection", True)):
        raise ValueError("controlled.common_intersection is false; a per-objective "
                         "mixture gives the arms different images and different "
                         "optimization, which is the defect this key exists to forbid.")
    _build_cxr_mixtures(cfg, ctrl, out_dir, rng)
    _build_histo_mixtures(cfg, ctrl, out_dir, rng)
    if (cfg.get("granularity", {}) or {}).get("enabled", True):
        _build_taix_mixture(cfg, out_dir)

    for o in outs:
        if os.path.exists(o):
            record_sources(o, pool_srcs)
    append_status(status, f"training mixtures written to {out_dir}")
    return out_dir


def _build_cxr_mixtures(cfg: dict, ctrl: dict, out_dir: str, rng):
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    if not os.path.exists(pool_csv):
        raise MissingInput(f"the CXR pool manifest is absent at {pool_csv}; run main_build_cxr_pool first.")
    pool = read_csv_defensively(pool_csv)
    mimic_train = pool[(pool["dataset"] == "mimic") &
                       (pool["split"] == "train")].copy().reset_index(drop=True)

    findings = [f for f in CANONICAL_CXR_FINDINGS
                if f != "no_finding" and f in mimic_train.columns]
    has_label = mimic_train[findings].apply(
        lambda r: (pd.to_numeric(r, errors="coerce") == 1.0).any(), axis=1)
    has_report = pd.Series(False, index=mimic_train.index)
    if "report_rel_path" in mimic_train.columns:
        rp = mimic_train["report_rel_path"].astype(str)
        has_report = mimic_train["report_rel_path"].notna() & (rp.str.strip() != "") & (rp != "nan")

    usable = has_label & has_report
    common = mimic_train[usable].reset_index(drop=True)
    min_rows = int(ctrl.get("min_mixture_rows", 1000))
    if len(common) < min_rows:
        raise MissingInput(f"only {len(common)} MIMIC train cases carry both a label and a report, "
                           f"which is too few for a controlled comparison.")

    cols = ["case_id", "image_key", "dataset", "split", "subject_id"] + findings
    if "report_rel_path" in common.columns:
        cols.append("report_rel_path")
    out = os.path.join(out_dir, "cxr_common_train.csv")
    write_csv_atomic(common[[c for c in cols if c in common.columns]], out)

    if "subject_id" in mimic_train.columns:
        subjects = mimic_train["subject_id"].dropna().unique()
        rng.shuffle(subjects)
        half = len(subjects) // 2
        for tag, ids in (("cxrh1", subjects[:half]), ("cxrh2", subjects[half:])):
            sub = mimic_train[mimic_train["subject_id"].astype(str).isin(set(ids.astype(str)))]
            write_csv_atomic(sub[[c for c in cols if c in sub.columns]],
                             os.path.join(out_dir, f"{tag}_common_train.csv"))
    return out


def _build_taix_mixture(cfg: dict, out_dir: str) -> str:
    from controlled.matrix import LABEL_FORMS, taix_label_columns
    tx = cfg["taix"]
    pool_csv = tx["pool_manifest_csv"]
    if not os.path.exists(pool_csv):
        raise MissingInput(f"the TAIX pool manifest is absent at {pool_csv}; run main_build_taix_pool first.")
    pool = read_csv_defensively(pool_csv)
    train = pool[pool["split"].astype(str).str.lower() == "train"].reset_index(drop=True)
    if train.empty:
        raise MissingInput("the TAIX pool carries no train split; main_build_taix_pool preserves the released "
                           "splits, so an empty train split means the pool was built wrong.")

    all_cols, per_form = [], {}
    for form in LABEL_FORMS:
        cols = taix_label_columns(form, tx)
        missing = [c for c in cols if c not in train.columns]
        if missing:
            raise MissingInput(f"the TAIX pool is missing {missing} for label form '{form}'.")
        per_form[form] = cols
        all_cols += cols

    complete = train[all_cols].notna().all(axis=1)
    common = train[complete].reset_index(drop=True)
    if len(common) < int((cfg.get("controlled", {}) or {}).get("min_mixture_rows", 1000)):
        raise MissingInput(f"only {len(common)} TAIX train rows carry every form's columns, "
                           f"which is too few for a controlled comparison.")

    cap = int((cfg.get("granularity", {}) or {}).get("max_train_cases", 0))
    if cap and len(common) > cap:
        subs = common["subject_id"].dropna().astype(str).unique()
        rng = np.random.default_rng(int(cfg.get("seed", 42)))
        rng.shuffle(subs)
        keep, n = set(), 0
        for sid in subs:
            if n >= cap:
                break
            keep.add(sid)
            n += int((common["subject_id"].astype(str) == sid).sum())
        common = common[common["subject_id"].astype(str).isin(keep)].reset_index(drop=True)

    keep_cols = ["case_id", "image_key", "dataset", "split", "subject_id",
                 "reader_id"] + all_cols
    out = os.path.join(out_dir, "taix_common_train.csv")
    write_csv_atomic(common[[c for c in keep_cols if c in common.columns]], out)
    return out


def _build_histo_mixtures(cfg: dict, ctrl: dict, out_dir: str, rng):
    histo_pool_csv = cfg["histo"]["pool_manifest_csv"]
    if not os.path.exists(histo_pool_csv):
        raise MissingInput(f"the histo pool manifest is absent at {histo_pool_csv}; run main_build_histo_pool first.")
    histo = read_csv_defensively(histo_pool_csv)

    excluded = list((ctrl.get("modality_objective_exclusions", {}) or {}).get("histo", []))
    trained_here = [o for o in ctrl["objectives"] if o not in excluded]
    if "image_text" in trained_here:
        raise ValueError(
            "the histopathology arm cannot train an image-text objective on the same images as "
            "its siblings, because no captioned histopathology collection shares images with a "
            "labeled one. Exclude it in controlled.modality_objective_exclusions, or say in the "
            "protocol which shared captioned source is being used.")

    nct = histo[(histo["dataset"] == "nct_crc") &
                (histo["split"] == "train")].copy().reset_index(drop=True)
    if "tissue_class" not in nct.columns:
        raise MissingInput("the histo pool carries no tissue_class column, so the supervised "
                           "level has no target and no common intersection exists.")
    common = nct[nct["tissue_class"].notna() &
                 (nct["tissue_class"].astype(str).str.strip() != "")].reset_index(drop=True)
    if len(common) < int(ctrl.get("min_mixture_rows", 1000)):
        raise MissingInput(f"only {len(common)} NCT-CRC train patches carry a tissue class, "
                           f"which is too few for a controlled comparison.")

    cols = [c for c in ("case_id", "image_key", "dataset", "split", "tissue_class")
            if c in common.columns]
    out = os.path.join(out_dir, "histo_common_train.csv")
    write_csv_atomic(common[cols], out)
    return out
