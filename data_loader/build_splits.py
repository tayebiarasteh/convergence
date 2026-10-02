"""
data_loader/build_splits.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from Inference.resume_utils import (MissingInput, append_status, check_build_params,
                                    fingerprint_file, status_path, write_build_params,
                                    write_csv_atomic)
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _assert_patient_disjoint(df: pd.DataFrame, label: str) -> None:
    from data_loader.build_utils import assert_patient_disjoint
    assert_patient_disjoint(df, label)


def main_build_splits(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    sp = cfg["splits"]
    out_dir = sp["dir"]
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "case_splits.csv")
    anchor_csv = sp["anchor_dev_csv"]
    status = status_path(cfg, "splits")

    sources = {"cxr": cfg["cxr"]["pool_manifest_csv"]}
    if cfg.get("taix", {}).get("enabled"):
        sources["taix"] = cfg["taix"]["pool_manifest_csv"]
    if cfg.get("rexgradient", {}).get("enabled"):
        sources["rexgradient"] = cfg["rexgradient"]["pool_manifest_csv"]

    params = {"sources": {k: fingerprint_file(v) for k, v in sources.items()},
              "anchor_n": int(sp["anchor_n"]), "anchor_site": sp["anchor_source_site"],
              "enabled_sources": sorted(sources)}
    if os.path.exists(out_csv) and os.path.exists(anchor_csv) and not force:
        try:
            check_build_params(out_csv, params, owner="build_splits")
            return out_csv
        except Exception as e:
            pass

    frames: List[pd.DataFrame] = []
    for name, path in sources.items():
        if not os.path.exists(path):
            raise MissingInput(f"{name} pool manifest missing at {path}; build it before main_build_splits.")
        cols = ["case_id", "dataset", "split", "subject_id"]
        d = read_csv_defensively(path)
        for c in cols:
            if c not in d.columns:
                d[c] = np.nan
        frames.append(d[cols])
    allsp = pd.concat(frames, ignore_index=True)
    allsp["split"] = allsp["split"].astype(str).str.lower().replace({"val": "valid"})
    _assert_patient_disjoint(allsp, "released splits")

    holdout = set(sp["holdout_sites"])
    allsp["role"] = np.where(allsp["dataset"].isin(holdout), "external",
                             np.where(allsp["split"] == "train", "development", "internal_test"))
    write_csv_atomic(allsp, out_csv)
    write_build_params(out_csv, params)

    site = sp["anchor_source_site"]
    dev = allsp[(allsp["dataset"] == site) & (allsp["split"] == "train")]
    if len(dev) < int(sp["anchor_n"]):
        raise MissingInput(f"only {len(dev)} development cases at {site}, fewer than the "
                           f"{sp['anchor_n']} anchors requested.")
    pool = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    findings = [c for c in pool.columns if c in set(cfg["cxr"]["canonical_findings"])] \
        if "canonical_findings" in cfg.get("cxr", {}) else []
    rng = np.random.default_rng(int(cfg["seed"]))
    ids = dev["case_id"].astype(str).values
    if findings:
        merged = pool[pool["case_id"].astype(str).isin(set(ids))]
        per = max(1, int(sp["anchor_n"]) // max(1, len(findings)))
        picked: List[str] = []
        for f in findings:
            pos = merged[pd.to_numeric(merged[f], errors="coerce") == 1.0]["case_id"].astype(str).values
            if len(pos):
                picked.extend(rng.choice(pos, size=min(per, len(pos)), replace=False).tolist())
        picked = list(dict.fromkeys(picked))
        if len(picked) < int(sp["anchor_n"]):
            rest = [c for c in ids if c not in set(picked)]
            picked.extend(rng.choice(rest, size=int(sp["anchor_n"]) - len(picked), replace=False).tolist())
        anchors = picked[: int(sp["anchor_n"])]
    else:
        _ids = np.asarray(ids)
        anchors = _ids[rng.choice(len(_ids), size=int(sp["anchor_n"]), replace=False)].tolist()

    a = pd.DataFrame({"case_id": anchors})
    a["dataset"] = site
    a["role"] = "anchor_development"
    write_csv_atomic(a, anchor_csv)
    write_build_params(anchor_csv, params)

    leak = set(a["case_id"]) & set(allsp[allsp["role"] != "development"]["case_id"].astype(str))
    if leak:
        raise ValueError(f"[build_splits] {len(leak)} anchors are not development cases.")
    msg = (f"splits {len(allsp)} cases over {allsp.dataset.nunique()} sources; "
           f"{len(a)} anchors from {site} train, all development, stored by case id")
    append_status(status, msg)
    return out_csv
