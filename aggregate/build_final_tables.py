"""
aggregate/build_final_tables.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from Inference.report_utils import add_fdr

import warnings
warnings.filterwarnings("ignore")


def _safe_read(path: str) -> Optional[pd.DataFrame]:
    if path and os.path.exists(path):
        try:
            return read_csv_defensively(path)
        except Exception as e:
            print(f"[aggregate] Could not read {path}: {e}")
    return None


def _results_base(cfg: dict) -> str:
    return cfg["alignment"]["results_base_dir"]


def _map_auroc_cols(df: pd.DataFrame, prefix: str = "auroc") -> pd.DataFrame:
    df = df.copy()
    if f"{prefix}_mean" in df.columns:
        df = df.rename(columns={
            f"{prefix}_mean": "value_mean", f"{prefix}_std": "value_std",
            f"{prefix}_ci_low": "value_ci_low", f"{prefix}_ci_high": "value_ci_high",
            f"{prefix}_mean_raw": "value_raw",
        })
    elif prefix in df.columns:
        df = df.rename(columns={prefix: "value_mean"})
        df["value_mean"] = (df["value_mean"] * 100).round(1)
    return df



def build_alignment_table(cfg: dict) -> pd.DataFrame:
    base = _results_base(cfg)
    rows: List[pd.DataFrame] = []

    e1 = _safe_read(os.path.join(base, "results_e1_alignment", "pairwise_alignment.csv"))
    if e1 is not None and not e1.empty:
        e1 = e1.rename(columns={"encoder_a": "unit_a", "encoder_b": "unit_b",
                                 "pool": "modality_context", "value": "value_raw"})
        e1["experiment"]   = "E1"
        e1["sub_analysis"] = e1.get("comparison_type", "within_cxr")
        e1["k"]            = e1["metric"].str.extract(r"k(\d+)").astype(float)
        e1["value_mean"]   = (e1["value_raw"] * 100).round(1)
        rows.append(e1[["experiment","sub_analysis","comparison_type","unit_a",
                        "unit_b","modality_context","metric","k","n",
                        "value_raw","value_mean"]])

    wvc = _safe_read(os.path.join(base, "results_e1_alignment", "within_vs_cross.csv"))
    if wvc is not None and not wvc.empty:
        wvc = wvc.rename(columns={"comparison_type": "sub_analysis"})
        wvc["experiment"]      = "E1"
        wvc["comparison_type"] = "within_vs_cross_summary"
        rows.append(wvc)

    wvct = _safe_read(os.path.join(base, "results_e1_alignment", "within_vs_cross_tests.csv"))
    if wvct is not None and not wvct.empty:
        wvct["experiment"]      = "E1"
        wvct["comparison_type"] = "within_vs_cross_test"
        wvct["sub_analysis"]    = "within_vs_cross_test"
        rows.append(wvct)

    e2 = _safe_read(os.path.join(base, "results_e2_vla", "cross_modal_alignment.csv"))
    if e2 is not None and not e2.empty:
        e2_parts = []
        for met in ("mknn", "cknna"):
            if met not in e2.columns:
                continue
            part = e2[["image_encoder", "text_encoder", "n", "k"]].copy()
            part["metric"]        = met
            part["value_raw"]     = e2[met]
            part["value_mean"]    = (e2[met] * 100).round(1)
            part["value_std"]     = (e2.get(f"{met}_std") * 100).round(1) \
                                    if f"{met}_std" in e2.columns else float("nan")
            part["value_ci_low"]  = (e2.get(f"{met}_ci_low") * 100).round(1) \
                                    if f"{met}_ci_low" in e2.columns else float("nan")
            part["value_ci_high"] = (e2.get(f"{met}_ci_high") * 100).round(1) \
                                    if f"{met}_ci_high" in e2.columns else float("nan")
            e2_parts.append(part)
        if e2_parts:
            e2_long = pd.concat(e2_parts, ignore_index=True)
            e2_long["experiment"]       = "E2"
            e2_long["sub_analysis"]     = "vision_language"
            e2_long["comparison_type"]  = "cross_modal"
            e2_long = e2_long.rename(columns={"image_encoder": "unit_a",
                                              "text_encoder": "unit_b"})
            e2_long["modality_context"] = "cxr_to_report"
            rows.append(e2_long)

    e5 = _safe_read(os.path.join(cfg["controlled"]["results_e5_dir"],
                                 "e5_alignment_table.csv"))
    if e5 is not None and not e5.empty:
        id_cols = ["run_id_a","run_id_b","modality","objective_a","objective_b",
                   "backbone_a","backbone_b","init_a","init_b",
                   "same_objective","same_seed","n"]
        id_cols = [c for c in id_cols if c in e5.columns]
        e5_parts = []
        for met in ("mknn", "cknna"):
            if met not in e5.columns:
                continue
            part = e5[id_cols].copy()
            part["metric"]        = met
            part["value_raw"]     = e5[met]
            part["value_mean"]    = (e5[met] * 100).round(1)
            part["value_std"]     = (e5.get(f"{met}_std") * 100).round(1) \
                                    if f"{met}_std" in e5.columns else float("nan")
            part["value_ci_low"]  = (e5.get(f"{met}_ci_lower") * 100).round(1) \
                                    if f"{met}_ci_lower" in e5.columns else float("nan")
            part["value_ci_high"] = (e5.get(f"{met}_ci_upper") * 100).round(1) \
                                    if f"{met}_ci_upper" in e5.columns else float("nan")
            e5_parts.append(part)
        if e5_parts:
            e5_long = pd.concat(e5_parts, ignore_index=True)
            e5_long["experiment"]       = "E5"
            e5_long["sub_analysis"]     = "controlled_driver"
            e5_long["comparison_type"]  = (e5_long["objective_a"] + "_vs_"
                                           + e5_long["objective_b"])
            e5_long = e5_long.rename(columns={"run_id_a": "unit_a", "run_id_b": "unit_b",
                                              "modality": "modality_context"})
            rows.append(e5_long)

    e5s = _safe_read(os.path.join(cfg["controlled"]["results_e5_dir"], "e5_summary.csv"))
    if e5s is not None and not e5s.empty:
        e5s["experiment"]      = "E5"
        e5s["sub_analysis"]    = "controlled_driver_summary"
        e5s["comparison_type"] = "objective_pair_mean"
        if "modality" in e5s.columns:
            e5s = e5s.rename(columns={"modality": "modality_context"})
        rows.append(e5s)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)

    # FDR within (experiment, comparison_type, metric) families where p exists
    if "p_raw" in out.columns:
        out = add_fdr(out, family_cols=["experiment","comparison_type","metric"])
    return out



def build_structure_table(cfg: dict) -> pd.DataFrame:
    base = _results_base(cfg)
    rows: List[pd.DataFrame] = []

    e3 = _safe_read(os.path.join(base, "results_e3_ontology", "manifold_vs_ontology.csv"))
    if e3 is not None and not e3.empty:
        e3 = e3.rename(columns={
            "reference": "sub_analysis",
            "mantel_estimate": "stat_estimate", "mantel_ci_low": "stat_ci_low",
            "mantel_ci_high": "stat_ci_high", "mantel_estimate_raw": "stat_estimate_raw",
        })
        e3["experiment"]  = "E3"
        e3["metric"]      = "mantel_r_vs_reference"
        e3["stat_type"]   = "spearman_correlation"
        rows.append(e3)

    trip = _safe_read(os.path.join(base, "results_e3_ontology", "triplet_accuracy.csv"))
    if trip is not None and not trip.empty:
        trip = trip.rename(columns={
            "encoder": "unit_a",
            "triplet_acc_mean": "value_mean", "triplet_acc_std": "value_std",
            "triplet_acc_ci_low": "value_ci_low", "triplet_acc_ci_high": "value_ci_high",
            "triplet_acc_mean_raw": "value_raw",
        })
        trip["experiment"]   = "E3"
        trip["sub_analysis"] = "triplet_accuracy"
        trip["metric"]       = "triplet_prediction_accuracy"
        rows.append(trip)

    fits = _safe_read(os.path.join(base, "results_e4_scaling", "scaling_law_fits.csv"))
    if fits is not None and not fits.empty:
        # Spearman rho and slope are statistical measures on their natural scale:
        # estimate + CI + p, NOT percent, NO mean/std. Keep in stat_* columns.
        fits = fits.rename(columns={
            "axis": "sub_analysis",
            "spearman_estimate": "stat_estimate", "spearman_ci_low": "stat_ci_low",
            "spearman_ci_high": "stat_ci_high",
            "spearman_estimate_raw": "stat_estimate_raw",
            "spearman_p_raw": "p_raw", "spearman_p_fdr": "p_fdr",
        })
        fits["experiment"] = "E4"
        fits["metric"]     = "residual_vs_axis"
        fits["stat_type"]  = "spearman_correlation"
        rows.append(fits)

    sdat = _safe_read(os.path.join(base, "results_e4_scaling", "scaling_law_data.csv"))
    if sdat is not None and not sdat.empty:
        sdat = sdat.rename(columns={"encoder": "unit_a",
                                     "residual": "residual_raw",
                                     "auroc": "value_mean"})
        sdat["experiment"]   = "E4"
        sdat["sub_analysis"] = "per_encoder_data"
        sdat["metric"]       = "downstream_auroc"
        sdat["value_mean"]   = (sdat["value_mean"] * 100).round(1)  # AUROC->percent
        rows.append(sdat)

    resid = _safe_read(os.path.join(cfg["alignment"]["consensus_dir"], "encoder_residuals.csv"))
    if resid is not None and not resid.empty:
        resid = resid.rename(columns={"encoder": "unit_a"})
        resid["experiment"]   = "E4"
        resid["sub_analysis"] = "encoder_consensus_residual"
        resid["metric"]       = "procrustes_distance_to_consensus"
        rows.append(resid)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "p_raw" in out.columns:
        out = add_fdr(out, family_cols=["experiment","sub_analysis"])
    return out


def build_fracture_table(cfg: dict) -> pd.DataFrame:
    base = _results_base(cfg)
    rows: List[pd.DataFrame] = []

    pf = _safe_read(os.path.join(base, "results_e6_fracture", "per_finding_alignment.csv"))
    if pf is not None and not pf.empty:
        # report_metric columns: alignment_mean/_std/_ci_low/_ci_high (percent) +
        # alignment_mean_raw. Map to the standard value_* schema.
        pf = pf.rename(columns={
            "finding": "unit_a",
            "alignment_mean": "value_mean", "alignment_std": "value_std",
            "alignment_ci_low": "value_ci_low", "alignment_ci_high": "value_ci_high",
            "alignment_mean_raw": "value_raw",
            "learnability": "learnability_auroc"})
        pf["experiment"]   = "E6"
        pf["sub_analysis"] = "per_finding_alignment"
        pf["metric"]       = "mknn_per_finding"
        if "learnability_auroc" in pf.columns:
            pf["learnability_auroc"] = (pf["learnability_auroc"] * 100).round(1)
        rows.append(pf)

    fr = _safe_read(os.path.join(base, "results_e6_fracture", "fracture_regression.csv"))
    if fr is not None and not fr.empty:
        fr = fr.rename(columns={
            "spearman_estimate": "stat_estimate", "spearman_ci_low": "stat_ci_low",
            "spearman_ci_high": "stat_ci_high", "spearman_estimate_raw": "stat_estimate_raw",
            "spearman_p_raw": "p_raw",
        })
        fr["experiment"]   = "E6"
        fr["sub_analysis"] = "fracture_regression"
        fr["metric"]       = "alignment_vs_log_prevalence"
        fr["stat_type"]    = "spearman_correlation"
        rows.append(fr)

    demo = _safe_read(os.path.join(base, "results_e6_fracture", "per_demographic_alignment.csv"))
    if demo is not None and not demo.empty:
        demo = demo.rename(columns={
            "group": "unit_a", "demographic": "sub_analysis",
            "alignment_mean": "value_mean", "alignment_std": "value_std",
            "alignment_ci_low": "value_ci_low", "alignment_ci_high": "value_ci_high",
            "alignment_mean_raw": "value_raw"})
        demo["experiment"] = "E6"
        demo["metric"]     = "mknn_per_subgroup"
        rows.append(demo)

    demo_t = _safe_read(os.path.join(base, "results_e6_fracture", "per_demographic_tests.csv"))
    if demo_t is not None and not demo_t.empty:
        demo_t = demo_t.rename(columns={"demographic": "sub_analysis"})
        demo_t["experiment"] = "E6"
        demo_t["metric"]     = "subgroup_permutation_test"
        rows.append(demo_t)

    gold = _safe_read(os.path.join(base, "results_e6_fracture", "gold_label_fracture.csv"))
    if gold is not None and not gold.empty:
        gold = gold.rename(columns={"finding": "unit_a",
                                     "gold_alignment": "value_mean"})
        gold["experiment"]   = "E6"
        gold["sub_analysis"] = "gold_label_fracture"
        gold["metric"]       = "mknn_per_finding_gold"
        gold["value_mean"]   = (gold["value_mean"] * 100).round(1)
        rows.append(gold)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "p_raw" in out.columns:
        out = add_fdr(out, family_cols=["experiment","sub_analysis"])
    return out



def build_artifact_table(cfg: dict) -> pd.DataFrame:
    base    = _results_base(cfg)
    art_dir = os.path.join(base, "results_e7_artifact")
    rows: List[pd.DataFrame] = []

    ce = _safe_read(os.path.join(art_dir, "cross_encoder_probe.csv"))
    if ce is not None and not ce.empty:
        ce = _map_auroc_cols(ce, "auroc")
        ce = ce.rename(columns={"finding": "unit_a", "enc_train": "unit_b",
                                 "enc_eval": "unit_c",
                                 "oracle_auroc": "oracle_mean",
                                 "retention": "retention_mean"})
        ce["experiment"]   = "E7"
        ce["sub_analysis"] = "cross_encoder_probe"
        ce["metric"]       = "auroc_transfer"
        for c in ("oracle_mean","retention_mean"):
            if c in ce.columns:
                ce[c] = (ce[c] * 100).round(1)
        rows.append(ce)

    cs = _safe_read(os.path.join(art_dir, "cross_site_probe.csv"))
    if cs is not None and not cs.empty:
        cs = _map_auroc_cols(cs, "auroc")
        cs = cs.rename(columns={"finding": "unit_a", "site_train": "unit_b",
                                 "site_eval": "unit_c",
                                 "oracle_auroc": "oracle_mean",
                                 "retention": "retention_mean"})
        cs["experiment"]   = "E7"
        cs["sub_analysis"] = "cross_site_probe"
        cs["metric"]       = "auroc_transfer"
        for c in ("oracle_mean","retention_mean"):
            if c in cs.columns:
                cs[c] = (cs[c] * 100).round(1)
        rows.append(cs)

    st = _safe_read(os.path.join(art_dir, "stitching.csv"))
    if st is not None and not st.empty:
        st = _map_auroc_cols(st, "auroc")
        st = st.rename(columns={"finding": "unit_a", "enc_src": "unit_b",
                                 "enc_tgt": "unit_c",
                                 "retention": "retention_mean", "method": "method"})
        st["experiment"]   = "E7"
        st["sub_analysis"] = "stitching"
        st["metric"]       = "auroc_stitched"
        if "retention_mean" in st.columns:
            st["retention_mean"] = (st["retention_mean"] * 100).round(1)
        rows.append(st)

    drift = _safe_read(os.path.join(art_dir, "drift_detector_auroc.csv"))
    if drift is not None and not drift.empty:
        drift = _map_auroc_cols(drift, "auroc")
        drift = drift.rename(columns={"comparison": "sub_analysis",
                                       "ood_site": "unit_a"})
        drift["experiment"] = "E7"
        drift["metric"]     = "drift_auroc"
        rows.append(drift)

    rv = _safe_read(os.path.join(art_dir, "reader_validation.csv"))
    if rv is not None and not rv.empty:
        rv = _map_auroc_cols(rv, "auroc")
        rv["experiment"]   = "E7"
        rv["sub_analysis"] = "reader_validation"
        rv["metric"]       = "deviation_vs_distrust_auroc"
        rows.append(rv)

    agree = _safe_read(cfg["reader_study"].get("agreement_csv", ""))
    if agree is not None and not agree.empty:
        agree["experiment"]   = "E7"
        agree["sub_analysis"] = "inter_rater_agreement"
        rows.append(agree)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True, sort=False)
    if "p_raw" in out.columns:
        out = add_fdr(out, family_cols=["experiment","sub_analysis","metric"])
    return out



def build_theory_table(cfg: dict) -> pd.DataFrame:
    th_dir = cfg["theory"]["results_e8_dir"]
    rows: List[pd.DataFrame] = []

    syn = _safe_read(os.path.join(th_dir, "synthetic_alignment.csv"))
    if syn is not None and not syn.empty:
        syn_long = syn.melt(
            id_vars=["alpha","seed","pair_type"],
            value_vars=[c for c in ("mknn","procrustes") if c in syn.columns],
            var_name="metric", value_name="value_raw",
        )
        syn_long["experiment"]   = "E8"
        syn_long["sub_analysis"] = "synthetic_alignment"
        # mknn -> percent; procrustes is a distance, keep one-decimal raw scale
        syn_long["value_mean"] = syn_long.apply(
            lambda r: round(r["value_raw"] * 100, 1) if r["metric"] == "mknn"
            else round(r["value_raw"], 1), axis=1
        )
        rows.append(syn_long)

    # Per (alpha, pair_type) mean/std/CI over seeds (standardized report_metric).
    summ = _safe_read(os.path.join(th_dir, "informativeness_summary.csv"))
    if summ is not None and not summ.empty:
        summ = summ.rename(columns={
            "mknn_mean": "value_mean", "mknn_std": "value_std",
            "mknn_ci_low": "value_ci_low", "mknn_ci_high": "value_ci_high",
            "mknn_mean_raw": "value_raw"})
        summ["experiment"]   = "E8"
        summ["sub_analysis"] = "informativeness_summary"
        summ["metric"]       = "mknn_over_seeds"
        rows.append(summ)

    # Alignment-gap curve (sup_vs_sup minus ssl_vs_ssl) with paired-diff CI + p.
    curve = _safe_read(os.path.join(th_dir, "informativeness_curve.csv"))
    if curve is not None and not curve.empty:
        curve = curve.rename(columns={
            "alignment_gap_mean": "value_mean", "alignment_gap_std": "value_std",
            "alignment_gap_ci_low": "value_ci_low",
            "alignment_gap_ci_high": "value_ci_high",
            "alignment_gap_mean_raw": "value_raw"})
        curve["experiment"]   = "E8"
        curve["sub_analysis"] = "informativeness_curve"
        curve["metric"]       = "alignment_gap_sup_minus_ssl"
        rows.append(curve)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True, sort=False)



def main_build_final_tables(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    out_dir = os.path.join(cfg["outputs_root"], "final_tables")
    os.makedirs(out_dir, exist_ok=True)

    builders = {
        "convergence_alignment.csv": build_alignment_table,
        "convergence_structure.csv": build_structure_table,
        "fracture_robustness.csv":   build_fracture_table,
        "deployable_artifact.csv":   build_artifact_table,
        "theory_synthetic.csv":      build_theory_table,
    }

    from tqdm import tqdm
    for fname, builder in tqdm(list(builders.items()),
                               desc="[aggregate] final tables", unit="table"):
        path = os.path.join(out_dir, fname)
        # Per-table resume: skip a table that already exists unless force=True.
        if os.path.exists(path) and not force:
            print(f"[aggregate] {fname}: exists, skip (force=True to rebuild).")
            continue
        try:
            df = builder(cfg)
        except Exception as e:
            print(f"[aggregate] {fname}: builder failed: {e}")
            df = pd.DataFrame()
        df.to_csv(path, index=False)
        print(f"[aggregate] {fname}: {len(df)} rows -> {path}")

    print(f"\n[aggregate] All five final tables written to {out_dir}")
    return out_dir

