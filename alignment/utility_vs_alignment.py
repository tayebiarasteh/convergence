"""
alignment/utility_vs_alignment.py
Created on August 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Inference.report_utils import add_fdr, report_metric, report_spearman
from Inference.resume_utils import (MissingInput, append_status, status_path, write_csv_atomic)
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _frame_utility(cfg: Dict) -> pd.DataFrame:
    fr = cfg.get("frame", {}) or {}
    path = fr.get("performance_csv", "")
    if not fr.get("enabled", False) or not path or not os.path.exists(path):
        return pd.DataFrame()
    from controlled.matrix import frame_run_id
    perf = read_csv_defensively(path)
    need = {"encoder", "backbone", "seed", "data_composition", "metric_name", "value", "subgroup",
            "mitigation"}
    if not need.issubset(perf.columns):
        raise MissingInput(f"FRAME's performance table lacks {sorted(need - set(perf.columns))}.")
    sel = perf[(perf["metric_name"] == "auroc_overall")
               & (perf["subgroup"].isna())
               & (perf["mitigation"] == "none")]
    g = sel.groupby(["encoder", "backbone", "seed", "data_composition"])["value"].mean()
    omap = dict(fr.get("objective_map", {}))
    rows = []
    for fid in fr.get("runs", []):
        parts = str(fid).split("__")
        if len(parts) != 6:
            continue
        _mod, obj, backbone, _init, comp, seed_tag = parts
        key = (obj, backbone, float(seed_tag.replace("seed", "")), comp)
        if key not in g.index:
            key = (obj, backbone, int(seed_tag.replace("seed", "")), comp)
            if key not in g.index:
                continue
        rows.append({"run_id": frame_run_id(fid, omap),
                     "utility_mean_raw": float(g.loc[key]) / 100.0,
                     "utility_source": "frame_delivered_table"})
    if not rows:
        raise MissingInput("FRAME's table carries no overall AUROC for any configured run.")
    return pd.DataFrame(rows)


def main_utility_vs_alignment(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    out_dir = os.path.join(cfg["alignment"]["results_base_dir"], "results_e13_utility")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "utility_vs_alignment.csv")
    status = status_path(cfg, "e13_utility")
    align_csv = os.path.join(ctrl["results_e5_dir"], "e5_alignment_table.csv")
    util_csv = os.path.join(ctrl["results_e5_dir"], "e5_cell_utility.csv")
    from Inference.resume_utils import output_is_current, record_sources
    _sources = [p for p in (align_csv, util_csv) if os.path.exists(p)]
    if not force and output_is_current(out_csv, _sources, owner="E13"):
        return out_csv

    from controlled.matrix import require_e5_family_complete
    require_e5_family_complete(global_config_path, align_csv, util_csv, owner="E13")
    align = read_csv_defensively(align_csv)

    long = pd.concat([
        align.rename(columns={"run_id_a": "run_id", "run_id_b": "other"})[
            ["run_id", "other", "modality", "mknn", "same_axis_level"]],
        align.rename(columns={"run_id_b": "run_id", "run_id_a": "other"})[
            ["run_id", "other", "modality", "mknn", "same_axis_level"]],
    ], ignore_index=True)
    per_cell = long.groupby("run_id").agg(
        alignment_mean_raw=("mknn", "mean"), n_partners=("mknn", "size"),
        modality=("modality", "first")).reset_index()
    within = long[long["same_axis_level"] == True].groupby("run_id")["mknn"].mean()
    per_cell["alignment_within_level_raw"] = per_cell["run_id"].map(within)

    util = pd.DataFrame()
    if os.path.exists(util_csv):
        u = read_csv_defensively(util_csv)
        u = u[u["target"] == "MEAN"]
        if not u.empty:
            util = u[["run_id", "utility_mean_raw"]].copy()
            util["utility_source"] = "measured_here"
    frame_u = _frame_utility(cfg)
    util = pd.concat([util, frame_u], ignore_index=True) if not frame_u.empty else util
    if util.empty:
        raise MissingInput("no cell carries a utility value; run main_converge_eval (which writes "
                           "e5_cell_utility.csv) or enable the frame block before E13.")
    util = util.drop_duplicates("run_id", keep="first")

    m = per_cell.merge(util, on="run_id", how="inner").dropna(
        subset=["alignment_mean_raw", "utility_mean_raw"])
    if len(m) < 6:
        raise MissingInput(f"only {len(m)} cells carry both an alignment and a utility value.")
    parts = m["run_id"].str.split("__", expand=True)
    m["arm"] = parts[0]
    m["axis_level"] = parts[1]
    m["backbone"] = parts[2]
    m["init"] = parts[3]
    m["seed"] = parts[4].str.replace("seed", "", regex=False).astype(int)
    write_csv_atomic(m, os.path.join(out_dir, "cell_alignment_and_utility.csv"))

    rows: List[Dict] = []
    for arm, g in m.groupby("arm"):
        if len(g) < 6:
            continue
        r = report_spearman(g["alignment_mean_raw"].values, g["utility_mean_raw"].values,
                            prefix="rho_alignment_utility")
        r.update({"arm": arm, "level": "cells", "n_cells": int(len(g)),
                  "n_levels": int(g["axis_level"].nunique())})
        rows.append(r)
        for axis, col in (("alignment", "alignment_mean_raw"), ("utility", "utility_mean_raw")):
            by = g.groupby("axis_level")[col]
            level_means = by.mean()
            seed_sd = g[g["init"] == "dinov3"].groupby(["axis_level", "backbone"])[col].std()
            rows.append({"arm": arm, "level": "axis_spread", "axis": axis,
                         "level_spread_raw": float(level_means.max() - level_means.min()),
                         "seed_spread_raw": float(seed_sd.mean()) if len(seed_sd) else float("nan"),
                         "level_means": ";".join(f"{k}={v:.4f}" for k, v in
                                                 level_means.sort_index().items()),
                         "n_levels": int(len(level_means)), "n_cells": int(len(g))})
    if not rows:
        raise MissingInput("E13 produced no rows.")
    df = pd.DataFrame(rows)
    if "p_raw" in df.columns:
        df = add_fdr(df, family_cols=["level"])
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources)
    append_status(status, f"E13 wrote {len(df)} rows over {m.run_id.nunique()} cells")
    return out_csv
