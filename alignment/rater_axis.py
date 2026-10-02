"""
alignment/rater_axis.py
Created on August 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd

from Inference.report_utils import add_fdr, report_spearman
from Inference.resume_utils import (MissingInput, append_status, status_path, write_csv_atomic)
from Inference.stats_utils import exceedance_p
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _reader_effect(grades: np.ndarray, readers: np.ndarray, n_perm: int, seed: int) -> Dict:
    keep = np.isfinite(grades)
    g, r = grades[keep], readers[keep]
    if len(g) < 200 or len(np.unique(r)) < 5:
        return {}
    total = float(((g - g.mean()) ** 2).sum())
    if total <= 0:
        return {}

    def _between(vals, labels):
        s = 0.0
        m = vals.mean()
        for lab in np.unique(labels):
            v = vals[labels == lab]
            s += len(v) * (v.mean() - m) ** 2
        return s / total

    obs = _between(g, r)
    rng = np.random.RandomState(seed)
    draws = np.array([_between(g, r[rng.permutation(len(r))]) for _ in range(n_perm)])
    return {"reader_variance_share_raw": obs,
            "reference_mean_raw": float(draws.mean()),
            "excess_over_reference_raw": obs - float(draws.mean()),
            "p_raw": exceedance_p(obs, draws), "p_fdr": float("nan"),
            "n_cases": int(len(g)), "n_readers": int(len(np.unique(r)))}


def main_rater_axis(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln = cfg["alignment"]
    out_dir = os.path.join(aln["results_base_dir"], "results_e15_rater")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "rater_axis.csv")
    status = status_path(cfg, "e15_rater")
    pool_csv = cfg["taix"]["pool_manifest_csv"]
    _ladder_csv = os.path.join(aln["results_base_dir"], "results_e9_granularity",
                               "granularity_ladder_metrics.csv")
    from Inference.resume_utils import output_is_current, record_sources
    _sources = [p for p in (pool_csv, _ladder_csv) if os.path.exists(p)]
    if not force and output_is_current(out_csv, _sources, owner="E15"):
        return out_csv

    if not os.path.exists(pool_csv):
        raise MissingInput("the TAIX pool manifest is absent; run main_build_taix_pool before E15.")
    man = read_csv_defensively(pool_csv)
    if "reader_id" not in man.columns:
        raise MissingInput("the TAIX pool carries no reader_id, so the rater axis has no axis.")
    graded = [c for c in man.columns if c.startswith("grade__")]
    if not graded:
        raise MissingInput("the TAIX pool carries no graded columns.")

    n_perm = int(cfg["stats"]["n_perm"])
    seed = int(cfg["seed"])
    readers = man["reader_id"].astype(str).values
    rows: List[Dict] = []
    for g in graded:
        eff = _reader_effect(pd.to_numeric(man[g], errors="coerce").values, readers, n_perm, seed)
        if not eff:
            continue
        eff.update({"finding": g.replace("grade__", ""), "level": "reader_effect",
                    "exploratory": True,
                    "confound": "reader identity is not randomized against case mix, calendar "
                                "time or shift"})
        rows.append(eff)
    if not rows:
        raise MissingInput("E15 produced no reader-effect rows.")
    eff_df = pd.DataFrame(rows)

    lad = os.path.join(aln["results_base_dir"], "results_e9_granularity",
                       "granularity_ladder_metrics.csv")
    joined = pd.DataFrame()
    if os.path.exists(lad):
        lad_df = read_csv_defensively(lad)
        sev = lad_df[lad_df["step"] == "severity"]
        if not sev.empty and "auroc_mean_raw" in sev.columns:
            enc_side = sev.groupby("finding")["auroc_mean_raw"].mean().rename(
                "encoder_severity_auroc_raw").reset_index()
            joined = eff_df.merge(enc_side, on="finding", how="inner").dropna(
                subset=["reader_variance_share_raw", "encoder_severity_auroc_raw"])
    link_rows: List[Dict] = []
    if len(joined) >= 5:
        r = report_spearman(joined["reader_variance_share_raw"].values,
                            joined["encoder_severity_auroc_raw"].values,
                            prefix="rho_reader_vs_encoder")
        r.update({"level": "reader_effect_vs_encoder_severity", "n_findings": int(len(joined)),
                  "exploratory": True})
        link_rows.append(r)
        write_csv_atomic(joined, os.path.join(out_dir, "reader_effect_vs_encoder.csv"))

    df = add_fdr(pd.concat([eff_df, pd.DataFrame(link_rows)], ignore_index=True)
                 if link_rows else eff_df, family_cols=["level"])
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources)
    append_status(status, f"E15 wrote {len(df)} rows over {eff_df.finding.nunique()} findings "
                          f"and {int(eff_df.n_readers.max())} readers")
    return out_csv
