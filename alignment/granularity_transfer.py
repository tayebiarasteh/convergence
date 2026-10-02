"""
alignment/granularity_transfer.py
Created on September 1, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from Inference.report_utils import add_fdr
from Inference.resume_utils import (MissingInput, append_status, check_build_params,
                                    fingerprint_file, heartbeat_claim, run_units_resumable,
                                    status_path, write_build_params, write_csv_atomic,
                                    write_npz_atomic)
from Inference.stats_utils import exceedance_p
from alignment.granularity_ladder import _patient_disjoint_split, _severity_matched_side_labels

import warnings
warnings.filterwarnings("ignore")


def _fit_classifier(Xtr: np.ndarray, y: np.ndarray, sd: int):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                              random_state=sd).fit(Xtr, y)


def _step_labels(sub: pd.DataFrame, cfg: Dict) -> List[Tuple[str, str, str, np.ndarray]]:
    out = []
    for f in [c for c in cfg["cxr"]["canonical_findings"] if c in sub.columns]:
        out.append(("presence", f, "present against absent",
                    pd.to_numeric(sub[f], errors="coerce").values.astype(float)))
    for g in [c for c in sub.columns if c.startswith("grade__")]:
        grade = pd.to_numeric(sub[g], errors="coerce").values
        y = np.where(grade >= 2, 1.0, np.where(grade == 1, 0.0, np.nan))
        out.append(("severity", g.replace("grade__", ""),
                    "grade 2 or above against grade 1, positives only", y))
    for f in sorted({c.split("__")[1] for c in sub.columns if c.startswith("side__")}):
        y, _ = _severity_matched_side_labels(sub, f)
        out.append(("laterality", f, "right-only against left-only at matched grade",
                    np.asarray(y, dtype=float)))
    return out


def with_transfer_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    t = pd.to_numeric(df["auroc_transfer_raw"], errors="coerce")
    hi = pd.to_numeric(df["reference_ci_upper_raw"], errors="coerce")
    o = pd.to_numeric(df.get("auroc_oracle_raw"), errors="coerce")
    df["above_reference_p975"] = np.where(t.notna() & hi.notna(), t > hi, np.nan)
    df["retention_chance_corrected_raw"] = np.where(o > 0.5, (t - 0.5) / (o - 0.5), np.nan)
    return df


def transfer_summary(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("step").agg(
        n_pairs=("enc_train", "size"),
        share_above_reference_p975_raw=("above_reference_p975", lambda v: float(pd.to_numeric(v, errors="coerce").mean())),
        share_significant_fdr_raw=("p_fdr", lambda p: float((pd.to_numeric(p, errors="coerce") < 0.05).mean())),
        min_attainable_p=("n_perm", lambda n: float(1.0 / (pd.to_numeric(n, errors="coerce").max() + 1))),
        median_transfer_auroc_raw=("auroc_transfer_raw", "median"),
        median_oracle_auroc_raw=("auroc_oracle_raw", "median"),
        median_retention_raw=("retention_raw", "median"),
        median_retention_chance_corrected_raw=("retention_chance_corrected_raw", "median"),
    ).reset_index()


def _refresh_derived(out_csv: str, sources: List[str], out_dir: str) -> None:
    from Inference.regimes import regime_extra
    from Inference.resume_utils import output_is_current, record_sources
    df = read_csv_defensively(out_csv)
    if not {"above_reference_p975", "retention_chance_corrected_raw"}.issubset(df.columns):
        df = with_transfer_columns(df)
        write_csv_atomic(df, out_csv)
        record_sources(out_csv, sources)
    summary_csv = os.path.join(out_dir, "ladder_transfer_summary.csv")
    extra = regime_extra("e9_transfer_summary")
    if output_is_current(summary_csv, [out_csv], owner="E9/transfer", extra=extra):
        return
    summary = transfer_summary(df)
    write_csv_atomic(summary, summary_csv)
    record_sources(summary_csv, [out_csv], extra=extra)


def _ladder_rows(cfg: Dict, encoders: List[str], pool_name: str, man: pd.DataFrame, seed: int,
                 max_train: int, max_test: int):
    from encoders.cache_utils import load_embedding_cache
    common: Optional[set] = None
    caches: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for enc in encoders:
        emb, ids = load_embedding_cache(cfg, enc, pool_name)
        caches[enc] = (emb, ids)
        s = set(map(str, ids))
        common = s if common is None else (common & s)
    common = common & set(man["case_id"].astype(str))
    if not common or len(common) < 300:
        raise MissingInput(f"only {len(common or [])} TAIX cases are embedded by every core encoder.")
    sub = man[man["case_id"].astype(str).isin(common)].reset_index(drop=True)
    sid = sub["subject_id"].astype(str).values
    tr, te = _patient_disjoint_split(sid, seed)
    rng = np.random.RandomState(seed)
    tr_idx, te_idx = np.where(tr)[0], np.where(te)[0]
    if len(tr_idx) > max_train:
        tr_idx = np.sort(rng.choice(tr_idx, max_train, replace=False))
    if len(te_idx) > max_test:
        te_idx = np.sort(rng.choice(te_idx, max_test, replace=False))
    return sub, caches, tr_idx, te_idx, _step_labels(sub, cfg)


def _anchor_rows(cfg: Dict, enc: str, anchor_ids: List[str], out_dir: str) -> np.ndarray:
    from encoders.cache_utils import embedding_cache_path, load_embedding_cache
    path = os.path.join(out_dir, "anchors", f"{enc}.npz")
    src = embedding_cache_path(cfg, enc, "cxr_pool")
    expected = {"cxr_cache": fingerprint_file(src), "n_anchors": len(anchor_ids),
                "first_anchor": anchor_ids[0], "last_anchor": anchor_ids[-1]}
    if os.path.exists(path) and check_build_params(path, expected, owner="ladder_transfer"):
        return np.load(path)["embeddings"].astype(np.float32, copy=False)
    emb, ids = load_embedding_cache(cfg, enc, "cxr_pool")
    pos = {str(c): i for i, c in enumerate(ids)}
    missing = [c for c in anchor_ids if c not in pos]
    if missing:
        raise MissingInput(f"{len(missing)} anchor case ids are absent from {enc}'s chest cache.")
    arr = emb[np.array([pos[c] for c in anchor_ids])]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    write_npz_atomic(path, embeddings=arr.astype(np.float32, copy=False))
    write_build_params(path, expected)
    return arr


def main_granularity_transfer(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln = cfg["alignment"]
    out_dir = os.path.join(aln["results_base_dir"], "results_e9_granularity")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ladder_transfer.csv")
    status = status_path(cfg, "e9_ladder_transfer")
    partial_dir = os.path.join(out_dir, "partials_transfer")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(partial_dir)

    from Inference.resume_utils import output_is_current, record_sources
    pool_name = aln["e9_pools"][0]
    manifest_csv = cfg["taix"]["pool_manifest_csv"]
    if not os.path.exists(manifest_csv):
        raise MissingInput("the TAIX pool manifest is absent; run main_build_taix_pool before the ladder transfer.")
    man = read_csv_defensively(manifest_csv)
    if "subject_id" not in man.columns:
        raise MissingInput("the TAIX pool has no subject_id, so no patient-disjoint split exists.")
    anchors_csv = os.path.join(aln["consensus_dir"], "anchor_case_ids.csv")
    from alignment.consensus import read_anchor_case_ids
    n_anchors = int(aln.get("primary_anchors", 1024))
    anchor_ids = read_anchor_case_ids(anchors_csv)[:n_anchors]
    if len(anchor_ids) < n_anchors:
        raise MissingInput(f"the anchor file has {len(anchor_ids)} ids, fewer than the primary "
                           f"budget of {n_anchors}; run main_build_consensus first.")
    seed = int(cfg["seed"])
    n_perm = int(cfg["stats"].get("n_perm_transfer", 20))
    gr = cfg.get("granularity", {}) or {}
    max_train = int(gr.get("probe_max_train", 8000))
    max_test = int(gr.get("probe_max_test", 8000))

    from encoders.panel import list_encoder_names
    from encoders.cache_utils import embedding_cache_path
    encoders = list_encoder_names(global_config_path, roles=["core"])
    _sources = [p for p in ([manifest_csv, anchors_csv]
                            + [embedding_cache_path(cfg, e, pool_name) for e in encoders])
                if os.path.exists(p)]
    if not force and output_is_current(out_csv, _sources, owner="E9/transfer"):
        _refresh_derived(out_csv, _sources, out_dir)
        return out_dir

    sub, caches, tr_idx, te_idx, labels = _ladder_rows(cfg, encoders, pool_name, man, seed,
                                                       max_train, max_test)
    case_ids = sub["case_id"].astype(str).values

    def _relative(enc: str) -> np.ndarray:
        path = os.path.join(out_dir, "relative_taix", f"{enc}.npz")
        expected = {"anchors": fingerprint_file(anchors_csv), "n_anchors": n_anchors,
                    "n_rows": int(len(sub)), "seed": seed}
        if os.path.exists(path) and check_build_params(path, expected, owner="ladder_transfer"):
            return np.load(path)["rel"].astype(np.float32, copy=False)
        emb, ids = caches[enc]
        pos = {str(c): i for i, c in enumerate(ids)}
        X = emb[np.array([pos[c] for c in case_ids])]
        A = _anchor_rows(cfg, enc, anchor_ids, out_dir)
        rel = (X @ A.T).astype(np.float32, copy=False)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_npz_atomic(path, rel=rel)
        write_build_params(path, expected)
        return rel

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    _fit = _fit_classifier

    def compute_unit(enc_train: str) -> List[Dict]:
        rel_src = _relative(enc_train)
        rows: List[Dict] = []
        targets = {}
        for e in tqdm([e for e in encoders if e != enc_train],
                      desc=f"[E9/transfer] {enc_train[:20]} targets", unit="enc", leave=False):
            targets[e] = _relative(e)
            heartbeat_claim(partial_dir, f"e9_transfer__{enc_train}")
        for step, finding, contrast, y in tqdm(labels,
                                               desc=f"[E9/transfer] {enc_train[:20]} steps",
                                               unit="step", leave=False):
            heartbeat_claim(partial_dir, f"e9_transfer__{enc_train}")
            keep_tr = tr_idx[np.isfinite(y[tr_idx])]
            keep_te = te_idx[np.isfinite(y[te_idx])]
            if len(keep_tr) < 100 or len(keep_te) < 50:
                continue
            ytr, yte = y[keep_tr], y[keep_te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            sc = StandardScaler().fit(rel_src[keep_tr])
            Xtr = sc.transform(rel_src[keep_tr])
            model = _fit(Xtr, ytr, seed)
            rp = np.random.RandomState(seed)
            perm_models = []
            for _ in range(n_perm):
                try:
                    perm_models.append(_fit(Xtr, ytr[rp.permutation(len(ytr))], seed))
                except ValueError:
                    continue
            step_aurocs = []
            for enc_eval, rel_tgt in targets.items():
                Xte = sc.transform(rel_tgt[keep_te])
                obs = float(roc_auc_score(yte, model.predict_proba(Xte)[:, 1]))
                step_aurocs.append(obs)
                draws = np.array([float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
                                  for m in perm_models])
                rows.append({"enc_train": enc_train, "enc_eval": enc_eval, "step": step,
                             "finding": finding, "contrast": contrast,
                             "auroc_transfer_raw": obs,
                             "reference_mean_raw": float(draws.mean()) if draws.size else np.nan,
                             "reference_ci_upper_raw": (float(np.percentile(draws, 97.5))
                                                        if draws.size else np.nan),
                             "excess_over_reference_raw": obs - (float(draws.mean()) if draws.size else np.nan),
                             "p_raw": exceedance_p(obs, draws) if draws.size else np.nan,
                             "p_fdr": float("nan"), "n_perm": int(draws.size),
                             "n_train": int(len(keep_tr)), "n_test": int(len(keep_te)),
                             "n_positive_test": int((yte == 1).sum()), "n_anchors": n_anchors,
                             "kind": "transfer", "reference_unit": "label_permutation"})
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="e9_transfer",
        units=list(encoders), compute_unit=compute_unit,
        build_params={"seed": seed, "n_perm": n_perm, "n_anchors": n_anchors,
                      "anchors": fingerprint_file(anchors_csv), "n_rows": int(len(sub)),
                      "statistic": "transferred_probe_auroc_in_anchor_space_patient_disjoint"},
        progress_desc="[E9/transfer] source encoders", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("the ladder transfer produced no rows; every step was too small.")
    n_skipped = int(df.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(f"{n_skipped} of {len(encoders)} source encoders could not run, so the "
                           f"table would be short of pairs; the finished ones are cached and skip.")

    oracle_csv = os.path.join(out_dir, "ladder_transfer_oracle.csv")
    oracles: Dict[Tuple[str, str, str], float] = {}
    if os.path.exists(oracle_csv):
        prev = read_csv_defensively(oracle_csv)
        for _, r in prev.iterrows():
            oracles[(str(r["enc_eval"]), str(r["step"]), str(r["finding"]))] = float(r["auroc_oracle_raw"])
    orows = []
    for enc in tqdm(encoders, desc="[E9/transfer] oracles", unit="enc"):
        rel = _relative(enc)
        for step, finding, contrast, y in labels:
            key = (enc, step, finding)
            if key in oracles:
                orows.append({"enc_eval": enc, "step": step, "finding": finding,
                              "auroc_oracle_raw": oracles[key]})
                continue
            keep_tr = tr_idx[np.isfinite(y[tr_idx])]
            keep_te = te_idx[np.isfinite(y[te_idx])]
            if len(keep_tr) < 100 or len(keep_te) < 50:
                continue
            ytr, yte = y[keep_tr], y[keep_te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            sc = StandardScaler().fit(rel[keep_tr])
            m = _fit(sc.transform(rel[keep_tr]), ytr, seed)
            v = float(roc_auc_score(yte, m.predict_proba(sc.transform(rel[keep_te]))[:, 1]))
            oracles[key] = v
            orows.append({"enc_eval": enc, "step": step, "finding": finding, "auroc_oracle_raw": v})
        write_csv_atomic(pd.DataFrame(orows), oracle_csv)
    odf = pd.DataFrame(orows)
    df = df.merge(odf, on=["enc_eval", "step", "finding"], how="left")
    df["retention_raw"] = df["auroc_transfer_raw"] / df["auroc_oracle_raw"]
    df = with_transfer_columns(add_fdr(df, family_cols=["step"]))
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources)
    _refresh_derived(out_csv, _sources, out_dir)
    append_status(status, f"ladder transfer: {len(df)} rows over {df['enc_train'].nunique()} sources")
    return out_dir


def main_granularity_stitching(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln = cfg["alignment"]
    out_dir = os.path.join(aln["results_base_dir"], "results_e9_granularity")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "ladder_stitching.csv")
    summary_csv = os.path.join(out_dir, "ladder_stitching_summary.csv")
    status = status_path(cfg, "e9_ladder_stitching")
    partial_dir = os.path.join(out_dir, "partials_stitching")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(partial_dir)

    from Inference.regimes import regime_extra
    from artifact.stitching import _affine_map
    from Inference.resume_utils import output_is_current, record_sources
    pool_name = aln["e9_pools"][0]
    manifest_csv = cfg["taix"]["pool_manifest_csv"]
    if not os.path.exists(manifest_csv):
        raise MissingInput("the TAIX pool manifest is absent; run main_build_taix_pool before the ladder stitching.")
    man = read_csv_defensively(manifest_csv)
    if "subject_id" not in man.columns:
        raise MissingInput("the TAIX pool has no subject_id, so no patient-disjoint split exists.")
    seed = int(cfg["seed"])
    n_perm = int(cfg["stats"].get("n_perm_transfer", 20))
    gr = cfg.get("granularity", {}) or {}
    max_train = int(gr.get("probe_max_train", 8000))
    max_test = int(gr.get("probe_max_test", 8000))

    from encoders.panel import list_encoder_names
    from encoders.cache_utils import embedding_cache_path
    encoders = list_encoder_names(global_config_path, roles=["core"])
    _sources = [p for p in ([manifest_csv] + [embedding_cache_path(cfg, e, pool_name) for e in encoders])
                if os.path.exists(p)]
    extra = regime_extra("e9_stitching")
    if not force and output_is_current(summary_csv, _sources, owner="E9/stitching", extra=extra):
        return out_dir

    sub, caches, tr_idx, te_idx, labels = _ladder_rows(cfg, encoders, pool_name, man, seed,
                                                       max_train, max_test)
    case_ids = sub["case_id"].astype(str).values

    def _native(enc: str) -> np.ndarray:
        emb, ids = caches[enc]
        pos = {str(c): i for i, c in enumerate(ids)}
        return emb[np.array([pos[c] for c in case_ids])].astype(np.float64, copy=False)

    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    def compute_unit(enc_train: str) -> List[Dict]:
        X_t = _native(enc_train)
        fin_t = np.isfinite(X_t).all(axis=1)
        fits = []
        for step, finding, contrast, y in tqdm(labels, desc=f"[E9/stitching] {enc_train[:20]} fits",
                                               unit="step", leave=False):
            heartbeat_claim(partial_dir, f"e9_stitching__{enc_train}")
            keep_tr = tr_idx[np.isfinite(y[tr_idx]) & fin_t[tr_idx]]
            keep_te = te_idx[np.isfinite(y[te_idx]) & fin_t[te_idx]]
            if len(keep_tr) < 100 or len(keep_te) < 50:
                continue
            ytr, yte = y[keep_tr], y[keep_te]
            if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
                continue
            sc = StandardScaler().fit(X_t[keep_tr])
            Xtr = sc.transform(X_t[keep_tr])
            model = _fit_classifier(Xtr, ytr, seed)
            rp = np.random.RandomState(seed)
            perm_models = []
            for _ in range(n_perm):
                try:
                    perm_models.append(_fit_classifier(Xtr, ytr[rp.permutation(len(ytr))], seed))
                except ValueError:
                    continue
            oracle = float(roc_auc_score(yte, model.predict_proba(sc.transform(X_t[keep_te]))[:, 1]))
            fits.append((step, finding, contrast, keep_tr, keep_te, yte, sc, model, perm_models, oracle))
        rows: List[Dict] = []
        for enc_eval in tqdm([e for e in encoders if e != enc_train],
                             desc=f"[E9/stitching] {enc_train[:20]} mapped", unit="enc", leave=False):
            heartbeat_claim(partial_dir, f"e9_stitching__{enc_train}")
            X_s = _native(enc_eval)
            fin_s = np.isfinite(X_s).all(axis=1)
            fit_rows = tr_idx[fin_s[tr_idx] & fin_t[tr_idx]]
            if len(fit_rows) < X_s.shape[1] + 2:
                raise MissingInput(f"({enc_eval}, {enc_train}) share {len(fit_rows)} finite training "
                                   f"rows, fewer than the {X_s.shape[1] + 2} the map needs.")
            W, b = _affine_map(X_s[fit_rows], X_t[fit_rows])
            for step, finding, contrast, keep_tr, keep_te, yte, sc, model, perm_models, oracle in fits:
                ok = fin_s[keep_te]
                if ok.sum() < 50 or len(np.unique(yte[ok])) < 2:
                    continue
                Xte = sc.transform(X_s[keep_te[ok]] @ W + b)
                obs = float(roc_auc_score(yte[ok], model.predict_proba(Xte)[:, 1]))
                draws = np.array([float(roc_auc_score(yte[ok], m.predict_proba(Xte)[:, 1]))
                                  for m in perm_models])
                rows.append({"enc_train": enc_train, "enc_eval": enc_eval, "step": step,
                             "finding": finding, "contrast": contrast,
                             "auroc_transfer_raw": obs,
                             "reference_mean_raw": float(draws.mean()) if draws.size else np.nan,
                             "reference_ci_upper_raw": (float(np.percentile(draws, 97.5))
                                                        if draws.size else np.nan),
                             "excess_over_reference_raw": obs - (float(draws.mean()) if draws.size else np.nan),
                             "p_raw": exceedance_p(obs, draws) if draws.size else np.nan,
                             "p_fdr": float("nan"), "n_perm": int(draws.size),
                             "n_train": int(len(keep_tr)), "n_test": int(ok.sum()),
                             "n_positive_test": int((yte[ok] == 1).sum()),
                             "n_map_rows": int(len(fit_rows)), "auroc_oracle_raw": oracle,
                             "retention_raw": obs / oracle if oracle > 0 else np.nan,
                             "kind": "linear_map", "reference_unit": "label_permutation"})
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="e9_stitching",
        units=list(encoders), compute_unit=compute_unit,
        build_params={"seed": seed, "n_perm": n_perm, "n_rows": int(len(sub)),
                      "max_train": max_train, "max_test": max_test,
                      "statistic": "classifier_auroc_through_label_free_linear_map_patient_disjoint"},
        progress_desc="[E9/stitching] classifier encoders", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("the ladder stitching produced no rows; every step was too small.")
    n_skipped = int(df.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(f"{n_skipped} of {len(encoders)} classifier encoders could not run, so the "
                           f"table would be short of pairs; the finished ones are cached and skip.")
    df = with_transfer_columns(add_fdr(df, family_cols=["step"]))
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources, extra=extra)
    summary = transfer_summary(df)
    write_csv_atomic(summary, summary_csv)
    record_sources(summary_csv, _sources, extra=extra)
    append_status(status, f"ladder stitching: {len(df)} rows over {df['enc_train'].nunique()} classifier encoders")
    return out_dir
