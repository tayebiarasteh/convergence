"""
alignment/scaling_law.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from tqdm import tqdm
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import linregress

from alignment.linear_probe import per_finding_auroc, mean_auroc
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
from Inference.stats_utils import bootstrap_slope, bootstrap_spearman, N_BOOT

import warnings
warnings.filterwarnings("ignore")


def main_scaling_law(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    emb_cfg = cfg["embeddings"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e4_scaling")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "scaling_law_data.csv")
    if os.path.exists(out_csv) and not force:
        return out_dir

    # Load consensus residuals
    residuals_csv = os.path.join(aln_cfg["consensus_dir"], "encoder_residuals.csv")
    if not os.path.exists(residuals_csv):
        return out_dir
    residuals_df = read_csv_defensively(residuals_csv)
    residual_map: Dict[str, float] = dict(
        zip(residuals_df["encoder"], residuals_df["residual"])
    )

    manifest  = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    emb_dir   = emb_cfg["output_dir"]
    seed      = int(cfg.get("seed", 42))

    # Build train/test split from manifest (use existing split column)
    if "split" in manifest.columns:
        train_mask = (manifest["split"] == "train").values
        test_mask  = (manifest["split"] == "test").values | \
                     (manifest["split"] == "valid").values
    else:
        rng = np.random.RandomState(seed)
        idx_all = np.arange(len(manifest))
        rng.shuffle(idx_all)
        cut = int(0.8 * len(idx_all))
        train_mask = np.zeros(len(manifest), dtype=bool)
        test_mask  = np.zeros(len(manifest), dtype=bool)
        train_mask[idx_all[:cut]] = True
        test_mask[idx_all[cut:]]  = True

    # Build labels dict from manifest
    labels = {
        f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float)
        for f in CANONICAL_CXR_FINDINGS if f in manifest.columns
    }

    # Collect per-encoder data
    panel_img = cfg["encoder_panel"]["image"]
    rows = []
    for enc_name, spec in tqdm(list(panel_img.items()),
                               desc="[E4] per-encoder", unit="enc"):
        if "ladder" not in spec.get("roles", []) and "core" not in spec.get("roles", []):
            continue
        npz_path = os.path.join(emb_dir, enc_name, "cxr_pool.npz")
        if not os.path.exists(npz_path):
            continue

        d   = np.load(npz_path, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32)
        # Align embeddings to manifest row order
        enc_ids = d["case_ids"].astype(str)
        man_ids = manifest["case_id"].astype(str).values
        id_to_row = {c: i for i, c in enumerate(man_ids)}
        enc_id_to_row = {c: i for i, c in enumerate(enc_ids)}
        shared_ids = [c for c in man_ids if c in enc_id_to_row]
        if len(shared_ids) < 200:
            continue
        man_idx = [id_to_row[c] for c in shared_ids]
        enc_idx = [enc_id_to_row[c] for c in shared_ids]

        emb_aligned      = emb[enc_idx]
        labels_aligned   = {f: v[man_idx] for f, v in labels.items()}
        train_sub        = train_mask[man_idx]
        test_sub         = test_mask[man_idx]

        auroc_dict = per_finding_auroc(emb_aligned, labels_aligned, train_sub, test_sub)
        auroc      = mean_auroc(auroc_dict)

        residual = residual_map.get(enc_name, float("nan"))
        params_m = float(spec.get("params_m", float("nan")))
        year     = float(spec.get("year", float("nan")))

        rows.append({
            "encoder":   enc_name,
            "residual":  round(residual, 6),
            "params_m":  params_m,
            "log_params": round(float(np.log10(params_m)), 4) if params_m > 0 else float("nan"),
            "auroc":     round(auroc, 4),
            "year":      year,
        })

    df = pd.DataFrame(rows).dropna(subset=["residual"])
    df.to_csv(out_csv, index=False)

    # Fit scaling laws
    _fit_scaling(df, out_dir)
    return out_dir


def _fit_scaling(df: pd.DataFrame, out_dir: str):
    from Inference.report_utils import report_slope, report_spearman, add_fdr
    axes = [
        ("log_params", "log10(params_m)"),
        ("auroc",      "downstream_auroc"),
        ("year",       "release_year"),
    ]
    rows = []
    for col, label in axes:
        sub = df[df[col].notna() & df["residual"].notna()]
        if len(sub) < 4:
            continue
        x, y = sub[col].values, sub["residual"].values
        slope_rep = report_slope(x, y, prefix="slope")        # estimate+CI+p+R^2
        rho_rep   = report_spearman(x, y, prefix="spearman")  # estimate+CI+p
        row = {"axis": label, "n_encoders": int(len(sub))}
        # slope (statistical measure, natural scale): estimate/CI + p + R^2
        for k in ("slope_estimate", "slope_ci_low", "slope_ci_high",
                  "slope_estimate_raw", "r_squared", "intercept_raw"):
            row[k] = slope_rep.get(k)
        row["slope_p_raw"] = slope_rep.get("p_raw")
        # spearman (statistical measure, natural scale): estimate/CI + p
        for k in ("spearman_estimate", "spearman_ci_low", "spearman_ci_high",
                  "spearman_estimate_raw"):
            row[k] = rho_rep.get(k)
        row["spearman_p_raw"] = rho_rep.get("p_raw")
        rows.append(row)
    fits = pd.DataFrame(rows)
    if not fits.empty:
        fits = add_fdr(fits.rename(columns={"slope_p_raw": "p_raw"}),
                       family_cols=None).rename(
            columns={"p_raw": "slope_p_raw", "p_fdr": "slope_p_fdr",
                     "significant_fdr05": "slope_significant_fdr05"})
        fits = add_fdr(fits.rename(columns={"spearman_p_raw": "p_raw"}),
                       family_cols=None).rename(
            columns={"p_raw": "spearman_p_raw", "p_fdr": "spearman_p_fdr",
                     "significant_fdr05": "spearman_significant_fdr05"})
    fits.to_csv(os.path.join(out_dir, "scaling_law_fits.csv"), index=False)