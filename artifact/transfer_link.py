"""
artifact/transfer_link.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Inference.report_utils import (add_fdr, effective_p, report_partial_spearman,
                                    report_slope, report_spearman)
from Inference.resume_utils import (MissingInput, append_status, status_path, write_csv_atomic)
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def chance_corrected(auroc, oracle) -> np.ndarray:
    t = pd.to_numeric(pd.Series(auroc), errors="coerce").values
    o = pd.to_numeric(pd.Series(oracle), errors="coerce").values
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(o > 0.5, (t - 0.5) / (o - 0.5), np.nan)


def _pair_key(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    df = df.copy()
    df["pair_lo"] = np.minimum(df[a].astype(str), df[b].astype(str))
    df["pair_hi"] = np.maximum(df[a].astype(str), df[b].astype(str))
    return df


def _link_row(m: pd.DataFrame, outcome: str, with_oracle: bool) -> Dict:
    from scipy.stats import spearmanr
    from Inference.stats_utils import jackknife_pairs_statistic
    x, y = m["alignment"].values, m[outcome].values
    row = report_spearman(x, y, prefix="rho")
    row["p_naive_raw"] = row.pop("p_raw")
    row.pop("p_fdr", None)
    sl = report_slope(x, y, prefix="slope")
    for k, v in sl.items():
        row[k if k.startswith("slope") else f"slope_{k}"] = v
    if with_oracle and "oracle" in m.columns and m["oracle"].notna().sum() >= 10:
        pc = report_partial_spearman(x, y, m["oracle"].values, prefix="rho_partial_oracle")
        for k, v in pc.items():
            row[k if k.startswith("rho_partial") else f"rho_partial_oracle_{k}"] = v
    jk = jackknife_pairs_statistic(m["pair_lo"].values, m["pair_hi"].values, [x, y],
                                   lambda a, b: spearmanr(a, b).statistic)
    row.update({"rho_crossed_std_raw": jk["std"], "rho_crossed_ci_low_raw": jk["ci_lower"],
                "rho_crossed_ci_high_raw": jk["ci_upper"], "crossed_n_units": jk["n_units"],
                "p_raw": jk["p_value"], "p_fdr": float("nan"),
                "p_basis": "delete_one_unit_jackknife", "outcome": outcome,
                "n_pairs": int(len(m))})
    return row


def _retention_by_pair(df: pd.DataFrame, a: str, b: str, auroc: str, oracle: str,
                       ratio: str) -> pd.DataFrame:
    d = _pair_key(df[df[a] != df[b]], a, b)
    d["retention_chance_corrected"] = chance_corrected(d[auroc], d[oracle])
    return d.groupby(["pair_lo", "pair_hi"]).agg(
        retention=(ratio, "mean"), retention_chance_corrected=("retention_chance_corrected", "mean"),
        oracle=(oracle, "mean")).reset_index()


def main_transfer_link(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln = cfg["alignment"]
    base = aln["results_base_dir"]
    out_dir = os.path.join(base, "results_e12_link")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "alignment_vs_retention.csv")
    status = status_path(cfg, "e12_link")
    probe_csv = os.path.join(base, "results_e7_artifact", "cross_encoder_probe.csv")
    align_csv = os.path.join(base, "results_e1_alignment", "pairwise_alignment.csv")
    e6_csv = os.path.join(base, "results_e6_fracture", "per_finding_alignment.csv")
    stitch_csv = os.path.join(base, "results_e7_artifact", "stitching.csv")
    ctl_csv = os.path.join(cfg["controlled"]["results_e5_dir"], "controlled_transfer.csv")
    e5_csv = os.path.join(cfg["controlled"]["results_e5_dir"], "e5_alignment_table.csv")
    primary_metric = str(aln["h4_alignment_metric"])
    link_metrics = [str(m) for m in aln["h4_link_metrics"]]
    if primary_metric not in link_metrics:
        link_metrics = [primary_metric] + link_metrics
    primary_budget = int(aln.get("primary_anchors", 1024))
    from Inference.regimes import regime_extra, stale_reason
    from Inference.resume_utils import output_is_current, record_sources
    for p, stage in ((probe_csv, "main_universal_probe"), (align_csv, "main_convergence_map"), (e6_csv, "main_fracture_law")):
        if not os.path.exists(p):
            raise MissingInput(f"{os.path.basename(p)} is absent; run {stage} before E12.")
    why = stale_reason(e6_csv)
    if why:
        raise MissingInput(f"[E12] {why}; the finding-level link waits for main_fracture_law.")
    _sources = [p for p in (probe_csv, align_csv, e6_csv, stitch_csv, ctl_csv, e5_csv)
                if os.path.exists(p)]
    extra = {**regime_extra("e12_link"), "primary_metric": primary_metric,
             "primary_budget": primary_budget, "link_metrics": link_metrics}
    if not force and output_is_current(out_csv, _sources, owner="E12", extra=extra):
        return out_csv

    probe = read_csv_defensively(probe_csv)
    align = read_csv_defensively(align_csv)
    need_probe = {"enc_train", "enc_eval", "retention", "finding", "auroc_xenc", "oracle_auroc"}
    if not need_probe.issubset(probe.columns):
        raise MissingInput(f"cross_encoder_probe.csv lacks {sorted(need_probe - set(probe.columns))}")
    align = align[(align["pool"].astype(str) == "cxr_pool")
                  & (align["comparison_type"].astype(str) == "within_cxr")
                  & align["metric"].astype(str).isin(link_metrics)]
    align = _pair_key(align, "encoder_a", "encoder_b")
    obj = None
    if {"objective_train", "objective_eval"}.issubset(probe.columns):
        pk = _pair_key(probe, "enc_train", "enc_eval")
        pk["objective_pair"] = ["__".join(sorted((str(a), str(b))))
                                for a, b in zip(pk["objective_train"], pk["objective_eval"])]
        obj = pk.drop_duplicates(["pair_lo", "pair_hi"])[["pair_lo", "pair_hi", "objective_pair"]]

    rows: List[Dict] = []
    outcomes = ("retention", "retention_chance_corrected")
    st_pairs = None
    if os.path.exists(stitch_csv):
        st = read_csv_defensively(stitch_csv)
        st_pairs = _retention_by_pair(st[st["method"].astype(str) == "affine"], "enc_src",
                                      "enc_tgt", "auroc", "oracle_auroc", "retention")
    for metric, sub in align.groupby("metric"):
        a = sub.groupby(["pair_lo", "pair_hi"])["value"].mean().rename("alignment").reset_index()
        for budget, pgrp in probe.groupby("n_anchors"):
            r = _retention_by_pair(pgrp, "enc_train", "enc_eval", "auroc_xenc", "oracle_auroc",
                                   "retention")
            m = a.merge(r, on=["pair_lo", "pair_hi"], how="inner")
            if obj is not None:
                m = m.merge(obj, on=["pair_lo", "pair_hi"], how="left")
            if len(m) < 10:
                continue
            is_primary = str(metric) == primary_metric
            for outcome in outcomes:
                mm = m.dropna(subset=["alignment", outcome])
                row = _link_row(mm, outcome, with_oracle=is_primary)
                row.update({"route": "anchor", "level": "all_pairs", "metric": str(metric),
                            "n_anchors": int(budget)})
                rows.append(row)
                if obj is not None and is_primary:
                    for op, g in mm.groupby("objective_pair"):
                        if len(g) >= 10:
                            row = _link_row(g, outcome, with_oracle=False)
                            row.update({"route": "anchor", "level": "objective_pair",
                                        "metric": str(metric), "n_anchors": int(budget),
                                        "objective_pair": str(op)})
                            rows.append(row)
        if st_pairs is not None:
            m = a.merge(st_pairs, on=["pair_lo", "pair_hi"], how="inner")
            if len(m) >= 10:
                for outcome in outcomes:
                    row = _link_row(m.dropna(subset=["alignment", outcome]), outcome,
                                    with_oracle=False)
                    row.update({"route": "stitching", "level": "all_pairs", "metric": str(metric)})
                    rows.append(row)

    if os.path.exists(ctl_csv) and os.path.exists(e5_csv):
        ct = read_csv_defensively(ctl_csv)
        e5 = _pair_key(read_csv_defensively(e5_csv), "run_id_a", "run_id_b")
        for metric in [c for c in ("mknn", "cknna") if c in e5.columns]:
            a5 = e5.groupby(["pair_lo", "pair_hi"])[metric].mean().rename("alignment").reset_index()
            for group, g in ct.groupby("group"):
                r = _retention_by_pair(g, "enc_train", "enc_eval", "auroc_xenc_raw",
                                       "oracle_auroc_raw", "retention_raw")
                m = a5.merge(r, on=["pair_lo", "pair_hi"], how="inner")
                for label, mm in ((str(group), m),
                                  (f"{group}_dinov3_cells",
                                   m[~m.pair_lo.str.contains("__random__")
                                     & ~m.pair_hi.str.contains("__random__")])):
                    if len(mm) < 10 or (label != str(group) and len(mm) == len(m)):
                        continue
                    for outcome in outcomes:
                        row = _link_row(mm.dropna(subset=["alignment", outcome]), outcome,
                                        with_oracle=False)
                        row.update({"route": "controlled_anchor", "level": label,
                                    "metric": metric, "n_anchors": int(g["n_anchors"].iloc[0])})
                        rows.append(row)

    e6 = read_csv_defensively(e6_csv)
    acol = "alignment_mean_raw" if "alignment_mean_raw" in e6.columns else "alignment"
    if {"finding", acol}.issubset(e6.columns) and len(e6):
        pr = probe[probe["n_anchors"] == primary_budget].copy()
        pr["retention_chance_corrected"] = chance_corrected(pr["auroc_xenc"], pr["oracle_auroc"])
        r_f = pr.groupby("finding").agg(retention=("retention", "mean"),
                                        retention_chance_corrected=("retention_chance_corrected",
                                                                    "mean")).reset_index()
        m_f = e6[["finding", acol]].rename(columns={acol: "alignment"}).merge(
            r_f, on="finding", how="inner")
        for outcome in outcomes:
            mf = m_f.dropna(subset=["alignment", outcome])
            if len(mf) < 6:
                continue
            row = report_spearman(mf["alignment"].values, mf[outcome].values, prefix="rho")
            row.update({"route": "anchor", "metric": "per_finding_mknn_positive_cases",
                        "level": "finding_level", "n_anchors": primary_budget,
                        "outcome": outcome, "p_basis": "findings_as_units",
                        "n_findings": int(len(mf)),
                        "findings": ";".join(sorted(mf["finding"].astype(str)))})
            rows.append(row)

    if not rows:
        raise MissingInput("no encoder pair has both an alignment value and a retention value.")
    df = add_fdr(pd.DataFrame(rows), family_cols=["route", "level", "metric", "outcome"])

    primary = ((df["route"] == "anchor") & (df["level"] == "all_pairs")
               & (df["metric"] == primary_metric) & (df["n_anchors"] == primary_budget)
               & (df["outcome"] == "retention"))
    added = (((df["route"] == "stitching") & (df["level"] == "all_pairs")
              & (df["metric"] == primary_metric))
             | ((df["route"] == "controlled_anchor") & (df["level"] == "cxr")
                & (df["metric"] == "mknn")))
    df["h4_role"] = np.where(primary, "primary", np.where(added, "added", ""))
    eff = df.apply(effective_p, axis=1)
    df["meets_h4_rule"] = primary & (eff < 0.05) & (df["rho_estimate_raw"] > 0) \
        & (pd.to_numeric(df.get("slope_r_squared"), errors="coerce") > 0)
    if int(primary.sum()) != 1:
        raise MissingInput(f"[E12] {int(primary.sum())} rows match the pre-specified H4 row "
                           f"({primary_metric}, {primary_budget} anchors); E1 or main_universal_probe is short of "
                           f"that statistic or budget.")
    verdict = bool(df["meets_h4_rule"].any())
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources, extra=extra)
    prow = df[primary].iloc[0]
    msg = (f"E12 link over {int(prow['n_pairs'])} encoder pairs on the chest pool: rho "
           f"{prow['rho_estimate_raw']:.3f}, delete-one-encoder p {prow['p_raw']:.3g} "
           f"(naive p {prow['p_naive_raw']:.3g}); alignment predicts retention = {verdict}")
    append_status(status, msg)
    return out_csv
