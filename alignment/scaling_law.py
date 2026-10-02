"""
alignment/scaling_law.py
Created on June 21, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import hashlib
import json
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
from Inference.resume_utils import append_status, status_path, write_csv_atomic, MissingInput
warnings.filterwarnings("ignore")


def _mean_panel_alignment(align_csv: str, metric: str) -> pd.DataFrame:
    al = read_csv_defensively(align_csv)
    al = al[(al["pool"].astype(str) == "cxr_pool") & (al["metric"].astype(str) == metric)
            & (al["comparison_type"].astype(str) == "within_cxr")]
    al = al[["encoder_a", "encoder_b", "value"]].copy()
    al["value"] = pd.to_numeric(al["value"], errors="coerce")
    return al.dropna()


def _encoder_means(pairs: pd.DataFrame, drop: str = None) -> Dict[str, float]:
    p = pairs if drop is None else pairs[(pairs.encoder_a != drop) & (pairs.encoder_b != drop)]
    long = pd.concat([p[["encoder_a", "value"]].rename(columns={"encoder_a": "enc"}),
                      p[["encoder_b", "value"]].rename(columns={"encoder_b": "enc"})])
    return long.groupby("enc")["value"].mean().to_dict()


def _crossed_spearman(pairs: pd.DataFrame, axis: Dict[str, float]) -> Dict:
    from scipy.stats import norm, spearmanr

    def _rho(drop=None):
        means = _encoder_means(pairs, drop)
        encs = [e for e in means if e != drop and np.isfinite(axis.get(e, np.nan))]
        if len(encs) < 4:
            return np.nan
        return float(spearmanr([means[e] for e in encs], [axis[e] for e in encs]).statistic)

    point = _rho()
    encs = sorted(set(pairs.encoder_a) | set(pairs.encoder_b))
    reps = np.array([v for v in (_rho(e) for e in encs) if np.isfinite(v)])
    out = {"rho_crossed_std_raw": np.nan, "rho_crossed_ci_low_raw": np.nan,
           "rho_crossed_ci_high_raw": np.nan, "p_crossed_raw": np.nan,
           "crossed_n_encoders": int(reps.size)}
    if reps.size >= 3 and np.isfinite(point):
        se = float(np.sqrt((reps.size - 1) / reps.size * np.sum((reps - reps.mean()) ** 2)))
        out.update({"rho_crossed_std_raw": se, "rho_crossed_ci_low_raw": point - 1.96 * se,
                    "rho_crossed_ci_high_raw": point + 1.96 * se,
                    "p_crossed_raw": float(2 * norm.sf(abs(point) / se)) if se > 0 else np.nan})
    return out


def main_scaling_law(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "scaling")
    from alignment.consensus import consensus_state
    state, reason = consensus_state(cfg, global_config_path)
    if state == "pending":
        raise MissingInput(f"[E4] {reason}; E4 runs once main_build_consensus has finished.")
    consensus_ok = state == "ready"
    aln_cfg = cfg["alignment"]
    pools = aln_cfg["e4_pools"]
    roles = list(aln_cfg["e4_roles"])
    emb_cfg = cfg["embeddings"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e4_scaling")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "scaling_law_data.csv")
    residuals_csv = os.path.join(aln_cfg["consensus_dir"], "encoder_residuals.csv")
    align_csv = os.path.join(aln_cfg["results_base_dir"], "results_e1_alignment",
                             "pairwise_alignment.csv")
    if not os.path.exists(align_csv):
        raise MissingInput("[E4] the E1 pairwise table is absent; run main_convergence_map before E4.")
    from Inference.regimes import regime_extra
    from Inference.resume_utils import output_is_current, record_sources
    _sources = ([residuals_csv] if consensus_ok else []) + [cfg["cxr"]["pool_manifest_csv"], align_csv]
    _pm = {n: [float(cfg["encoder_panel"]["image"][n].get("params_m", 0) or 0),
               float(cfg["encoder_panel"]["image"][n].get("year", 0) or 0)]
           for n in sorted(cfg["encoder_panel"]["image"])
           if set(cfg["encoder_panel"]["image"][n].get("roles", [])) & set(roles)}
    _x_axis = hashlib.sha256(json.dumps(_pm, sort_keys=True).encode()).hexdigest()[:16]
    panel_metric = str(aln_cfg.get("h4_alignment_metric", "mknn_at_pool_cap"))
    _params = {"roles": sorted(roles), "pools": sorted(pools),
               "params_m": _pm, "x_axis": _x_axis, "consensus_state": state,
               "panel_metric": panel_metric, **regime_extra("e4")}
    if not force and output_is_current(out_csv, _sources, owner="E4", extra=_params):
        _width_control(cfg, out_csv, out_dir, force)
        return out_dir
    from Inference.resume_utils import claim_unit, heartbeat_claim, release_claim
    if not claim_unit(out_dir, "scaling"):
        return out_dir
    if not force and output_is_current(out_csv, _sources, owner="E4", extra=_params):
        release_claim(out_dir, "scaling")
        return out_dir
    residual_map: Dict[str, float] = {}
    if consensus_ok:
        residuals_df = read_csv_defensively(residuals_csv)
        residual_map = dict(zip(residuals_df["encoder"], residuals_df["residual"]))
    panel_pairs = _mean_panel_alignment(align_csv, panel_metric)
    panel_means = _encoder_means(panel_pairs)

    manifest  = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    emb_dir   = emb_cfg["output_dir"]
    seed      = int(cfg.get("seed", 42))

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

    labels = {
        f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float, copy=False)
        for f in CANONICAL_CXR_FINDINGS if f in manifest.columns
    }

    panel_img = cfg["encoder_panel"]["image"]
    rows = []
    for enc_name, spec in tqdm(list(panel_img.items()),
                               desc="[E4] per-encoder", unit="enc"):
        if "ladder" not in spec.get("roles", []) and "core" not in spec.get("roles", []):
            continue
        heartbeat_claim(out_dir, "scaling")
        npz_path = os.path.join(emb_dir, enc_name, "cxr_pool.npz")
        if not os.path.exists(npz_path):
            continue

        d   = np.load(npz_path, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32, copy=False)
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
            "mean_panel_alignment_raw": panel_means.get(enc_name, float("nan")),
            "params_m":  params_m,
            "log_params": round(float(np.log10(params_m)), 4) if params_m > 0 else float("nan"),
            "auroc":     round(auroc, 4),
            "year":      year,
        })

    df = pd.DataFrame(rows)
    df = df[df["residual"].notna() | df["mean_panel_alignment_raw"].notna()]
    write_csv_atomic(df, out_csv)
    fits_csv = os.path.join(out_dir, "scaling_law_fits.csv")
    _fit_scaling(df, fits_csv, panel_pairs)
    record_sources(fits_csv, _sources, extra=_params)
    record_sources(out_csv, _sources, extra=_params)
    release_claim(out_dir, "scaling")
    _width_control(cfg, out_csv, out_dir, force)
    append_status(status, f"E4 written to {out_dir}")
    return out_dir


def _fit_scaling(df: pd.DataFrame, fits_csv: str, panel_pairs: pd.DataFrame):
    from Inference.report_utils import report_slope, report_spearman, add_fdr
    axes = [
        ("log_params", "log10(params_m)"),
        ("auroc",      "downstream_auroc"),
        ("year",       "release_year"),
    ]
    rows = []
    for outcome, ycol in (("consensus_residual", "residual"),
                          ("mean_panel_alignment", "mean_panel_alignment_raw")):
        for col, label in axes:
            sub = df[df[col].notna() & df[ycol].notna()]
            if len(sub) < 4:
                continue
            x, y = sub[col].values, sub[ycol].values
            slope_rep = report_slope(x, y, prefix="slope")
            rho_rep   = report_spearman(x, y, prefix="spearman")
            row = {"outcome": outcome, "axis": label, "n_encoders": int(len(sub))}
            for k in ("slope_estimate", "slope_ci_low", "slope_ci_high",
                      "slope_estimate_raw", "r_squared", "intercept_raw"):
                row[k] = slope_rep.get(k)
            row["slope_p_raw"] = slope_rep.get("p_raw")
            for k in ("spearman_estimate", "spearman_ci_low", "spearman_ci_high",
                      "spearman_estimate_raw"):
                row[k] = rho_rep.get(k)
            row["spearman_p_raw"] = rho_rep.get("p_raw")
            if outcome == "mean_panel_alignment":
                row.update(_crossed_spearman(panel_pairs, dict(zip(sub["encoder"], sub[col]))))
            rows.append(row)
    fits = pd.DataFrame(rows)
    if not fits.empty:
        fits = add_fdr(fits.rename(columns={"slope_p_raw": "p_raw"}),
                       family_cols=["outcome"]).rename(
            columns={"p_raw": "slope_p_raw", "p_fdr": "slope_p_fdr",
                     "significant_fdr05": "slope_significant_fdr05"})
        fits = add_fdr(fits.rename(columns={"spearman_p_raw": "p_raw"}),
                       family_cols=["outcome"]).rename(
            columns={"p_raw": "spearman_p_raw", "p_fdr": "spearman_p_fdr",
                     "significant_fdr05": "spearman_significant_fdr05"})
    write_csv_atomic(fits, fits_csv)


def _width_control(cfg: Dict, data_csv: str, out_dir: str, force: bool = False) -> None:
    from Inference.regimes import regime_extra
    from Inference.report_utils import add_fdr, report_partial_spearman, report_spearman
    from Inference.resume_utils import output_is_current, record_sources
    out_csv = os.path.join(out_dir, "scaling_law_width_control.csv")
    panel = cfg["encoder_panel"]["image"]
    dims = {n: int(panel[n].get("dim", 0) or 0) for n in sorted(panel)}
    extra = {**regime_extra("e4_width"), "dims": dims}
    if not force and output_is_current(out_csv, [data_csv], owner="E4/width", extra=extra):
        return
    df = read_csv_defensively(data_csv)
    df["log_width"] = [np.log10(dims[e]) if dims.get(e, 0) > 0 else np.nan for e in df["encoder"]]
    df = df[df["mean_panel_alignment_raw"].notna() & df["log_width"].notna()]
    y = df["mean_panel_alignment_raw"].values
    n_widths = int(np.unique(df["log_width"].values).size)
    rows = []
    for name, x, z in (("alignment_vs_log_width", df["log_width"].values, None),
                       ("log_params_vs_log_width", df["log_width"].values, None),
                       ("alignment_vs_log_params_given_width", df["log_params"].values, df["log_width"].values),
                       ("alignment_vs_year_given_width", df["year"].values, df["log_width"].values)):
        yy = df["log_params"].values if name == "log_params_vs_log_width" else y
        if z is None:
            rep = report_spearman(x, yy, prefix="rho")
        elif n_widths < 3:
            rep = report_partial_spearman(x[:0], yy[:0], z[:0], prefix="rho")
        else:
            rep = report_partial_spearman(x, yy, z, prefix="rho")
        rows.append({"analysis": name, "control": ("log10(embedding_width)" if z is not None else "none"),
                     "n_encoders": int(np.sum(np.isfinite(x) & np.isfinite(yy))), **rep})
    out = add_fdr(pd.DataFrame(rows), family_cols=["control"])
    write_csv_atomic(out, out_csv)
    record_sources(out_csv, [data_csv], extra=extra)
