"""
controlled/matrix_contrasts.py
Created on September 23, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from Inference.report_utils import report_metric
from Inference.resume_utils import MissingInput, append_status, status_path, write_csv_atomic
from Inference.stats_utils import jackknife_over_units, jackknife_pairs_statistic

import warnings
warnings.filterwarnings("ignore")


def _parts(run_id: str) -> Dict[str, str]:
    p = str(run_id).split("__")
    return {"arm": p[0], "level": p[1], "backbone": p[2], "init": p[3], "seed": p[4]}


def _annotate(t: pd.DataFrame) -> pd.DataFrame:
    t = t.copy()
    for side in ("a", "b"):
        f = t[f"run_id_{side}"].map(_parts)
        for k in ("arm", "level", "backbone", "init", "seed"):
            t[f"{k}_{side}"] = f.map(lambda d, k=k: d[k])
    return t


def _group_row(contrast: str, setting: str, g: pd.DataFrame, metric: str = "mknn") -> Dict:
    v = pd.to_numeric(g[metric], errors="coerce").values
    row = report_metric(v, prefix="alignment", is_percent=True)
    jk = jackknife_over_units(v, g["run_id_a"].values, g["run_id_b"].values)
    row.update({"contrast": contrast, "setting": setting, "metric": metric,
                "n_pairs": int(len(g)), "n_cells": int(len(set(g.run_id_a) | set(g.run_id_b))),
                "min_raw": float(np.nanmin(v)) if len(v) else np.nan,
                "max_raw": float(np.nanmax(v)) if len(v) else np.nan,
                "crossed_std_raw": jk["std"], "crossed_ci_low_raw": jk["ci_lower"],
                "crossed_ci_high_raw": jk["ci_upper"], "crossed_n_cells": jk.get("n_units")})
    return row


def _difference_row(contrast: str, setting: str, a: pd.DataFrame, b: pd.DataFrame,
                    metric: str = "mknn") -> Dict:
    both = pd.concat([a.assign(_in_a=1.0), b.assign(_in_a=0.0)], ignore_index=True)
    v = pd.to_numeric(both[metric], errors="coerce").values
    flag = both["_in_a"].values

    def _diff(vals, fl):
        if (fl == 1).sum() == 0 or (fl == 0).sum() == 0:
            return np.nan
        return float(np.nanmean(vals[fl == 1]) - np.nanmean(vals[fl == 0]))

    jk = jackknife_pairs_statistic(both["run_id_a"].values, both["run_id_b"].values, [v, flag], _diff)
    return {"contrast": contrast, "setting": setting, "metric": metric, "kind": "difference",
            "difference_raw": _diff(v, flag), "difference_unit": "fraction_of_neighbors",
            "crossed_std_raw": jk["std"], "crossed_ci_low_raw": jk["ci_lower"],
            "crossed_ci_high_raw": jk["ci_upper"], "p_raw": jk["p_value"],
            "p_basis": "delete_one_cell_jackknife", "n_pairs_a": int(len(a)),
            "n_pairs_b": int(len(b)), "crossed_n_cells": jk.get("n_units")}


def _initialization_rows(t: pd.DataFrame) -> List[Dict]:
    rows = []
    e5 = t[(t["experiment"] == "E5") & (t["backbone_a"] == "vit_s") & (t["backbone_b"] == "vit_s")]
    for arm, g in e5.groupby("arm_a"):
        cross = g[g["init_a"] != g["init_b"]]
        if cross.empty:
            continue
        same_obj = cross[cross["level_a"] == cross["level_b"]]
        diff_obj = cross[cross["level_a"] != cross["level_b"]]
        rand = g[(g["init_a"] == "random") & (g["init_b"] == "random")]
        dino = g[(g["init_a"] == "dinov3") & (g["init_b"] == "dinov3")]
        seed_pairs = dino[(dino["level_a"] == dino["level_b"]) & (dino["seed_a"] != dino["seed_b"])]
        dino_cross = dino[dino["level_a"] != dino["level_b"]]
        setting = f"{arm}/vit_s"
        for name, sub in (("same_objective_across_initializations", same_obj),
                          ("different_objectives_across_initializations", diff_obj),
                          ("different_objectives_both_random", rand),
                          ("same_objective_seed_pairs_both_dinov3", seed_pairs),
                          ("different_objectives_both_dinov3", dino_cross)):
            if len(sub):
                rows.append(_group_row(name, setting, sub))
        if len(same_obj) and len(diff_obj):
            rows.append(_difference_row("same_minus_different_objective_across_initializations",
                                        setting, same_obj, diff_obj))
    return rows


def _disjoint_rows(t: pd.DataFrame) -> List[Dict]:
    d = t[t["experiment"] == "E5_disjoint"]
    if d.empty:
        return []
    same_half = d["arm_a"] == d["arm_b"]
    same_seed = d["seed_a"] == d["seed_b"]
    rows = []
    for name, sub in (("within_half_different_seed", d[same_half & ~same_seed]),
                      ("across_halves_different_seed", d[~same_half & ~same_seed]),
                      ("across_halves_same_seed", d[~same_half & same_seed])):
        if len(sub):
            r = _group_row(name, "cxr_disjoint/vit_s", sub)
            r["pairs"] = ";".join(f"{a}|{b}={v:.4f}" for a, b, v in
                                  zip(sub.run_id_a, sub.run_id_b, sub.mknn))
            rows.append(r)
    return rows


def _content_rows(t: pd.DataFrame) -> List[Dict]:
    f = t[(t["arm_a"] == "frame_cxr") & (t["backbone_a"] == t["backbone_b"])]
    f = f[f["level_a"].str.startswith("image_text") & f["level_b"].str.startswith("image_text")]
    rows = []
    for bb, g in f.groupby("backbone_a"):
        setting = f"frame_cxr/{bb}"
        nat = g[(g["level_a"] == "image_text") & (g["level_b"] == "image_text")
                & (g["seed_a"] != g["seed_b"])]
        if len(nat):
            rows.append(_group_row("natural_seed_pairs", setting, nat))
        for cond in sorted({lv for lv in set(g.level_a) | set(g.level_b) if lv != "image_text"}):
            vs = g[((g["level_a"] == cond) & (g["level_b"] == "image_text"))
                   | ((g["level_b"] == cond) & (g["level_a"] == "image_text"))]
            for name, sub in ((f"{cond}_vs_natural_different_seed", vs[vs["seed_a"] != vs["seed_b"]]),
                              (f"{cond}_vs_natural_same_seed", vs[vs["seed_a"] == vs["seed_b"]])):
                if len(sub):
                    rows.append(_group_row(name, setting, sub))
    return rows


def _retention_rows(t: pd.DataFrame, init: pd.DataFrame) -> List[Dict]:
    rows = []
    same = t[(t["init_a"] == "dinov3") & (t["init_b"] == "dinov3")
             & (t["backbone_a"] == t["backbone_b"]) & (t["level_a"] == t["level_b"])
             & (t["seed_a"] != t["seed_b"])]
    seed_agree = same.groupby(["arm_a", "backbone_a", "level_a"])["mknn"].mean()
    for (arm, bb, level), g in init.groupby(["arm_group", "backbone", "axis_level"]):
        v = pd.to_numeric(g["mknn"], errors="coerce").values
        row = report_metric(v, prefix="alignment", is_percent=True)
        row.update({"contrast": "cell_to_initialization", "setting": f"{arm}/{bb}",
                    "level": level, "metric": "mknn", "n_cells": int(len(g)),
                    "min_raw": float(np.nanmin(v)), "max_raw": float(np.nanmax(v)),
                    "seed_pair_agreement_raw": float(seed_agree.get((arm, bb, level), np.nan))})
        rows.append(row)
    return rows


def main_matrix_contrasts(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    out_dir = cfg["controlled"]["results_e5_dir"]
    out_csv = os.path.join(out_dir, "matrix_contrasts.csv")
    align_csv = os.path.join(out_dir, "e5_alignment_table.csv")
    util_csv = os.path.join(out_dir, "e5_cell_utility.csv")
    init_csv = os.path.join(out_dir, "cell_to_init_alignment.csv")
    status = status_path(cfg, "matrix_contrasts")
    from Inference.regimes import regime_extra, stale_reason
    from controlled.matrix import require_e5_family_complete
    from Inference.resume_utils import output_is_current, record_sources
    require_e5_family_complete(global_config_path, align_csv, util_csv, owner="main_matrix_contrasts")
    if not os.path.exists(init_csv) or stale_reason(init_csv):
        raise MissingInput(stale_reason(init_csv) or "the cell-to-initialization table is absent; "
                           "main_init_alignment writes it.")
    extra = regime_extra("matrix_contrasts")
    if not force and output_is_current(out_csv, [align_csv, init_csv], owner="main_matrix_contrasts", extra=extra):
        return out_csv
    t = _annotate(read_csv_defensively(align_csv))
    init = read_csv_defensively(init_csv)
    init["arm_group"] = init["run_id"].map(lambda r: _parts(r)["arm"])
    rows = (_initialization_rows(t) + _disjoint_rows(t) + _content_rows(t)
            + _retention_rows(t, init))
    if not rows:
        raise MissingInput("the alignment table has none of the pairs the contrasts need.")
    df = pd.DataFrame(rows)
    from Inference.report_utils import add_fdr
    df = add_fdr(df, family_cols=None)
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, [align_csv, init_csv], extra=extra)
    for _, r in df.iterrows():
        val = r["difference_raw"] if r.get("kind") == "difference" else r["alignment_mean_raw"]
    append_status(status, f"main_matrix_contrasts: {len(df)} contrast rows")
    return out_csv
