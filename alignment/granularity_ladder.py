"""
alignment/granularity_ladder.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from tqdm import tqdm

from Inference.report_utils import add_fdr
from Inference.resume_utils import (MissingInput, append_status, heartbeat_claim,
                                    run_units_resumable, status_path, write_csv_atomic)
from Inference.stats_utils import exceedance_p
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")

STEPS = ("presence", "severity", "laterality")


def _patient_disjoint_split(subject_ids: np.ndarray, seed: int, frac: float = 0.7):
    subs = pd.unique(pd.Series(subject_ids).astype(str))
    rng = np.random.RandomState(seed)
    rng.shuffle(subs)
    train = set(subs[: max(1, int(frac * len(subs)))])
    tr = np.array([str(s) in train for s in subject_ids])
    return tr, ~tr


def _decodability(emb: np.ndarray, y: np.ndarray, subject_ids: np.ndarray, seed: int,
                  n_perm: int, max_train: int = 8000, max_test: int = 8000) -> Optional[Dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from Inference.stats_utils import bootstrap_auroc
    keep = np.isfinite(y) & np.isfinite(emb).all(axis=1)
    if keep.sum() < 200:
        return None
    emb, y, sid = emb[keep], y[keep], subject_ids[keep]
    tr, te = _patient_disjoint_split(sid, seed)
    rng = np.random.RandomState(seed)
    tr_idx = np.where(tr)[0]
    te_idx = np.where(te)[0]
    if len(tr_idx) > max_train:
        tr_idx = np.sort(rng.choice(tr_idx, max_train, replace=False))
    if len(te_idx) > max_test:
        te_idx = np.sort(rng.choice(te_idx, max_test, replace=False))
    if len(tr_idx) < 100 or len(te_idx) < 50:
        return None
    if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[te_idx])) < 2:
        return None
    sc = StandardScaler().fit(emb[tr_idx])
    Xtr, Xte = sc.transform(emb[tr_idx]), sc.transform(emb[te_idx])

    def _fit(labels):
        m = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                               random_state=seed).fit(Xtr, labels)
        return m, float(roc_auc_score(y[te_idx], m.predict_proba(Xte)[:, 1]))

    model, obs = _fit(y[tr_idx])
    scores = model.predict_proba(Xte)[:, 1]
    ci = bootstrap_auroc(y[te_idx], scores, cluster_ids=sid[te_idx])
    rp = np.random.RandomState(seed)
    draws = np.empty(int(n_perm), dtype=float)
    for i in range(int(n_perm)):
        yp = y[tr_idx][rp.permutation(len(tr_idx))]
        try:
            draws[i] = _fit(yp)[1]
        except ValueError:
            draws[i] = np.nan
    good = draws[np.isfinite(draws)]
    return {"auroc_raw": obs, "auroc_std_raw": ci["std"],
            "auroc_ci_lower_raw": ci["ci_lower"], "auroc_ci_upper_raw": ci["ci_upper"],
            "auroc_resample": ci.get("resample", "row"),
            "reference_mean_raw": float(np.mean(good)) if good.size else np.nan,
            "reference_ci_lower_raw": float(np.percentile(good, 2.5)) if good.size else np.nan,
            "reference_ci_upper_raw": float(np.percentile(good, 97.5)) if good.size else np.nan,
            "excess_over_reference_raw": obs - (float(np.mean(good)) if good.size else np.nan),
            "p_raw": exceedance_p(obs, good), "p_fdr": float("nan"),
            "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
            "n_positive": int((y == 1).sum()), "chance_raw": 0.5}


def _severity_matched_side_labels(man: pd.DataFrame, finding: str) -> Tuple[np.ndarray, np.ndarray]:
    rc, lc = f"side__{finding}__right", f"side__{finding}__left"
    gr, gl = f"grade__{finding}_right", f"grade__{finding}_left"
    right = pd.to_numeric(man.get(rc), errors="coerce")
    left = pd.to_numeric(man.get(lc), errors="coerce")
    if right is None or left is None:
        return np.array([]), np.array([])
    only_r = ((right == 1.0) & (left == 0.0)).values
    only_l = ((left == 1.0) & (right == 0.0)).values
    if gr in man.columns and gl in man.columns:
        grade = np.where(only_r, pd.to_numeric(man[gr], errors="coerce").values,
                         np.where(only_l, pd.to_numeric(man[gl], errors="coerce").values, np.nan))
    else:
        grade = np.where(only_r | only_l, 1.0, np.nan)
    idx = np.arange(len(man))
    keep = np.zeros(len(man), dtype=bool)
    rng = np.random.RandomState(0)
    for g in np.unique(grade[np.isfinite(grade)]):
        r_idx = idx[only_r & (grade == g)]
        l_idx = idx[only_l & (grade == g)]
        n = min(len(r_idx), len(l_idx))
        if n == 0:
            continue
        keep[rng.choice(r_idx, n, replace=False)] = True
        keep[rng.choice(l_idx, n, replace=False)] = True
    y = np.where(only_r, 1.0, np.where(only_l, 0.0, np.nan))
    y[~keep] = np.nan
    return y, grade


def main_granularity_ladder(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln = cfg["alignment"]
    out_dir = os.path.join(aln["results_base_dir"], "results_e9_granularity")
    os.makedirs(out_dir, exist_ok=True)
    status = status_path(cfg, "e9_granularity")
    partial_dir = os.path.join(out_dir, "partials")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(partial_dir)

    pool_name = aln["e9_pools"][0]
    manifest_csv = cfg["taix"]["pool_manifest_csv"]
    if not os.path.exists(manifest_csv):
        raise MissingInput("the TAIX pool manifest is absent; run main_build_taix_pool before E9.")
    man = read_csv_defensively(manifest_csv)
    if "subject_id" not in man.columns:
        raise MissingInput("the TAIX pool carries no subject_id, so no patient-disjoint split "
                           "is possible and every decodability score would be inflated.")
    seed = int(cfg["seed"])
    n_perm = int(cfg["stats"]["n_perm_structural"])
    gr = cfg.get("granularity", {}) or {}
    max_train = int(gr.get("probe_max_train", 8000))
    max_test = int(gr.get("probe_max_test", 8000))

    from encoders.panel import list_encoder_names
    from encoders.cache_utils import load_embedding_cache
    encoders = list_encoder_names(global_config_path, roles=["core"])
    graded = [c for c in man.columns if c.startswith("grade__")]
    sides = sorted({c.split("__")[1] for c in man.columns if c.startswith("side__")})

    def compute_unit(unit: str) -> List[Dict]:
        emb, ids = load_embedding_cache(cfg, unit, pool_name)
        row_of = {str(c): i for i, c in enumerate(ids)}
        have = man["case_id"].astype(str).isin(row_of).values
        sub = man[have].reset_index(drop=True)
        idx = np.array([row_of[c] for c in sub["case_id"].astype(str)])
        X = emb[idx]
        sid = sub["subject_id"].astype(str).values
        rows: List[Dict] = []

        probes: List[Tuple[str, str, np.ndarray, str]] = []
        for f in [c for c in cfg["cxr"]["canonical_findings"] if c in sub.columns]:
            probes.append(("presence", f, pd.to_numeric(sub[f], errors="coerce").values,
                           "present against absent"))
        for g in graded:
            grade = pd.to_numeric(sub[g], errors="coerce").values
            probes.append(("severity", g.replace("grade__", ""),
                           np.where(grade >= 2, 1.0, np.where(grade == 1, 0.0, np.nan)),
                           "grade 2 or above against grade 1, positives only"))
        for f in sides:
            y, _grade = _severity_matched_side_labels(sub, f)
            if y.size:
                probes.append(("laterality", f, y,
                               "right-only against left-only at matched grade"))

        bar = tqdm(probes, desc=f"[E9] {unit}", unit="probe", leave=False)
        for step, finding, y, contrast in bar:
            bar.set_postfix_str(f"{step}/{finding}")
            heartbeat_claim(partial_dir, f"e9__{unit}")
            r = _decodability(X, y, sid, seed, n_perm, max_train, max_test)
            if not r:
                continue
            r.update({"encoder": unit, "step": step, "finding": finding, "contrast": contrast})
            rows.append(r)

        from scipy.stats import spearmanr
        rp = np.random.RandomState(seed)
        for g in tqdm(graded, desc=f"[E9] {unit} ordering", unit="finding", leave=False):
            heartbeat_claim(partial_dir, f"e9__{unit}")
            grade = pd.to_numeric(sub[g], errors="coerce").values
            fin = np.isfinite(grade) & np.isfinite(X).all(axis=1)
            healthy = fin & (grade == 0)
            positive = fin & (grade >= 1)
            if healthy.sum() < 50 or positive.sum() < 200 or len(np.unique(grade[positive])) < 3:
                continue
            c = X[healthy].mean(axis=0)
            c = c / max(float(np.linalg.norm(c)), 1e-9)
            d = 1.0 - X[positive] @ c
            gp = grade[positive]
            obs = float(spearmanr(gp, d).correlation)
            draws = np.empty(int(n_perm), dtype=float)
            for i in range(int(n_perm)):
                draws[i] = float(spearmanr(gp[rp.permutation(len(gp))], d).correlation)
            good = draws[np.isfinite(draws)]
            rows.append({"encoder": unit, "step": "severity_ordering",
                         "finding": g.replace("grade__", ""),
                         "contrast": "distance from the grade-0 centroid against the grade, "
                                     "positives only",
                         "rho_raw": obs,
                         "reference_mean_raw": float(np.mean(good)) if good.size else np.nan,
                         "p_raw": exceedance_p(obs, good), "p_fdr": float("nan"),
                         "n_positive_cases": int(positive.sum()), "n_grade0": int(healthy.sum()),
                         "reference_unit": "case_permutation", "kind": "statistic"})
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="e9", units=encoders,
        compute_unit=compute_unit, progress_desc="[E9] encoders",
        build_params={"steps": list(STEPS), "seed": seed, "n_perm": n_perm,
                      "statistic": "linear_probe_auroc_patient_disjoint"},
        use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("E9 produced no rows; extract TAIX embeddings (main_extract_image_embeddings) first.")
    df = add_fdr(df, family_cols=["step"])
    stats_df = df[df["step"] == "severity_ordering"].dropna(axis=1, how="all")
    if not stats_df.empty:
        write_csv_atomic(stats_df, os.path.join(out_dir, "granularity_ladder_statistics.csv"))
    df = df[df["step"] != "severity_ordering"].dropna(axis=1, how="all")
    from Inference.report_utils import fmt_pct
    rep = df.copy()
    for src, dst in (("auroc_raw", "auroc_mean"), ("auroc_std_raw", "auroc_std"),
                     ("auroc_ci_lower_raw", "auroc_ci_low"),
                     ("auroc_ci_upper_raw", "auroc_ci_high")):
        if src in rep.columns:
            rep[dst] = rep[src].map(fmt_pct)
    rep["auroc_mean_raw"] = rep["auroc_raw"]
    out_csv = os.path.join(out_dir, "granularity_ladder_metrics.csv")
    write_csv_atomic(rep, out_csv)

    summary = df.groupby("step").agg(
        n_rows=("auroc_raw", "size"), auroc_mean_raw=("auroc_raw", "mean"),
        excess_mean_raw=("excess_over_reference_raw", "mean"),
        share_above_reference_raw=("excess_over_reference_raw", lambda v: float((v > 0).mean())),
        share_significant_raw=("p_fdr", lambda v: float((pd.to_numeric(v, errors="coerce")
                                                         < 0.05).mean()))).reset_index()
    write_csv_atomic(summary, os.path.join(out_dir, "granularity_ladder_summary.csv"))
    append_status(status, f"E9 wrote {len(rep)} rows over {df.encoder.nunique()} encoders")
    return out_csv
