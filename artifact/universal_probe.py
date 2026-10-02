"""
artifact/universal_probe.py
Created on June 23, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from alignment.linear_probe import eval_probe, eval_probe_scores, fit_probe, mean_auroc, per_finding_auroc
from artifact.relative_reps import load_relative_rep
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
from Inference.resume_utils import (MissingInput, append_status, clear_partial_dir,
                                    fingerprint_file, heartbeat_claim, print_projected_peak,
                                    run_units_resumable, status_path, write_csv_atomic)

import warnings
warnings.filterwarnings("ignore")


def refit_baseline(x_train, y_train, x_test, y_test, label_budget: int, seed: int,
                   n_draws: int = 20) -> Dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    n = min(int(label_budget), len(x_train))
    empty = {"mean": float("nan"), "sd": float("nan"), "n_draws": 0}
    if n < 20 or len(np.unique(y_test)) < 2:
        return empty
    pos = np.where(y_train == 1)[0]
    neg = np.where(y_train != 1)[0]
    if len(pos) < 2 or len(neg) < 2:
        return empty
    n_pos = int(round(n * len(pos) / len(y_train)))
    n_pos = min(max(n_pos, 2), len(pos), n - 2)
    n_neg = min(n - n_pos, len(neg))
    vals = []
    for _ in range(int(n_draws)):
        idx = np.concatenate([rng.choice(pos, size=n_pos, replace=False),
                              rng.choice(neg, size=n_neg, replace=False)])
        m = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced").fit(
            x_train[idx], y_train[idx])
        vals.append(float(roc_auc_score(y_test, m.predict_proba(x_test)[:, 1])))
    arr = np.asarray(vals, dtype=float)
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "n_draws": int(len(arr))}


def _objective_of(cfg: Dict, enc: str) -> str:
    return str(cfg["encoder_panel"]["image"].get(enc, {}).get("objective", "unknown"))


def _assert_patient_disjoint_split(manifest: pd.DataFrame) -> None:
    from data_loader.build_utils import assert_patient_disjoint
    assert_patient_disjoint(manifest, "chest pool manifest")


def main_universal_probe(global_config_path: str, force: bool = False) -> str:
    cfg      = read_config(global_config_path)["Convergence"]
    status   = status_path(cfg, "probe")
    aln_cfg  = cfg["alignment"]
    cons_dir = cfg["artifact"]["consensus_dir"]
    out_dir  = os.path.join(aln_cfg["results_base_dir"], "results_e7_artifact")
    os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(out_dir, "cross_encoder_probe.csv")

    manifest = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    findings = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]
    labels   = {f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float, copy=False)
                for f in findings}

    split_col  = manifest["split"].astype(str).str.lower()
    train_mask = (split_col == "train").values
    test_mask  = split_col.isin(["test"]).values
    if not train_mask.any() or not test_mask.any():
        raise MissingInput("the pool manifest carries no train/test split, so no patient-disjoint "
                           "evaluation is possible; run main_build_splits first.")
    _assert_patient_disjoint_split(manifest)

    from Inference.resume_utils import output_is_current, record_sources
    from encoders.cache_utils import embedding_cache_path
    from encoders.panel import list_encoder_names
    _core = list_encoder_names(global_config_path, roles=["core"])
    _sources = [q for q in ([cfg["cxr"]["pool_manifest_csv"]]
                            + [embedding_cache_path(cfg, e, "cxr_pool") for e in _core])
                if os.path.exists(q)]
    if not force and output_is_current(out_csv, _sources, owner="E12"):
        return out_dir

    seed = int(cfg.get("seed", 42))
    rng  = np.random.RandomState(seed)
    label_budgets = [int(b) for b in aln_cfg["label_budgets"]]
    primary_anchors = int(aln_cfg.get("primary_anchors", 1024))
    anchor_counts = sorted({int(n) for n in aln_cfg.get("anchor_sweep", [primary_anchors])}
                           | {primary_anchors})

    core_encs = _core

    probe_max_train = int(cfg["embeddings"].get("probe_max_train", 25000))
    probe_max_test  = int(cfg["embeddings"].get("probe_max_test", 20000))
    train_pool = np.where(train_mask)[0]
    test_pool  = np.where(test_mask)[0]
    if len(train_pool) > probe_max_train:
        train_pool = np.sort(rng.choice(train_pool, probe_max_train, replace=False))
    if len(test_pool) > probe_max_test:
        test_pool = np.sort(rng.choice(test_pool, probe_max_test, replace=False))

    kept_rows = np.sort(np.unique(np.concatenate([train_pool, test_pool])))
    row_to_compact = {int(r): i for i, r in enumerate(kept_rows)}
    train_local = np.array([row_to_compact[int(r)] for r in train_pool])
    test_local  = np.array([row_to_compact[int(r)] for r in test_pool])
    n_compact = len(kept_rows)
    train_mask_c = np.zeros(n_compact, dtype=bool); train_mask_c[train_local] = True
    test_mask_c  = np.zeros(n_compact, dtype=bool); test_mask_c[test_local]  = True
    labels_c = {f: v[kept_rows] for f, v in labels.items()}
    manifest_ids = manifest["case_id"].astype(str).values

    _rr_cache: Dict[tuple, Optional[np.ndarray]] = {}
    def _rr(enc: str, n_a: int):
        key = (enc, n_a)
        if key not in _rr_cache:
            full = load_relative_rep(enc, cons_dir, n_anchors=n_a, primary=primary_anchors)
            _rr_cache[key] = full[kept_rows] if full is not None else None
        return _rr_cache[key]

    _nat_cache: Dict[str, Optional[np.ndarray]] = {}
    def _native(enc: str):
        if enc not in _nat_cache:
            from encoders.cache_utils import load_embedding_cache
            emb, ids = load_embedding_cache(cfg, enc, "cxr_pool")
            pos = {str(c): i for i, c in enumerate(ids)}
            take = [pos.get(c, -1) for c in manifest_ids[kept_rows]]
            out = np.full((len(kept_rows), emb.shape[1]), np.nan, dtype=np.float32)
            ok = np.array([t >= 0 for t in take])
            out[ok] = emb[[t for t in take if t >= 0]]
            _nat_cache[enc] = out
        return _nat_cache[enc]

    n_draws = int(aln_cfg.get("refit_draws", 20))
    partials = os.path.join(out_dir, "probe_partials")
    if force:
        clear_partial_dir(partials)
    shared_params = {"seed": seed, "n_rows": int(n_compact), "findings": list(findings),
                     "manifest": fingerprint_file(cfg["cxr"]["pool_manifest_csv"]),
                     "probe_max_train": probe_max_train, "probe_max_test": probe_max_test}

    def _refit_unit(enc: str) -> List[Dict]:
        spaces = {"anchor": _rr(enc, primary_anchors), "native": _native(enc)}
        widths = {k: (v.shape[1] if v is not None else 0) for k, v in spaces.items()}
        out: List[Dict] = []
        for fname in tqdm(findings, desc=f"[E12] {enc[:20]} findings", unit="finding",
                          leave=False):
            heartbeat_claim(partials, f"e12_refit__{enc}")
            col = labels_c[fname]
            shown = {}
            for space, X in spaces.items():
                if X is None:
                    continue
                fin = np.isfinite(X).all(axis=1)
                tr = np.where(train_mask_c & np.isfinite(col) & fin)[0]
                te = np.where(test_mask_c & np.isfinite(col) & fin)[0]
                if len(tr) < 20 or len(te) < 10:
                    continue
                for b in label_budgets:
                    rb = refit_baseline(X[tr], np.asarray(col)[tr], X[te], np.asarray(col)[te],
                                        label_budget=b, seed=seed, n_draws=n_draws)
                    out.append({"encoder": enc, "finding": fname, "space": space,
                                "label_budget": int(b), "refit_mean": rb["mean"],
                                "refit_sd": rb["sd"], "refit_draws": int(rb["n_draws"]),
                                "n_train": int(len(tr)), "n_test": int(len(te))})
                shown[space] = (out[-1]["refit_mean"], len(tr), len(te))
            if shown:
                parts = ", ".join(f"{sp} {v:.3f}" for sp, (v, _, _) in shown.items())
                n_tr, n_te = list(shown.values())[0][1], list(shown.values())[0][2]
        _nat_cache.pop(enc, None)
        return out

    refit_df = run_units_resumable(
        partial_dir=partials, group="e12_refit", units=list(core_encs),
        compute_unit=_refit_unit,
        build_params={**shared_params, "budgets": list(label_budgets), "draws": n_draws,
                      "primary_anchors": primary_anchors},
        progress_desc="[E12] refit baselines", use_claims=True, status_file=status)
    if int(refit_df.attrs.get("n_skipped", 0)):
        raise MissingInput(f"{refit_df.attrs['n_skipped']} of {len(core_encs)} refit units could "
                           f"not run, so the baselines every transfer row is measured against "
                           f"would be short; the finished units are cached and skip.")
    refit: Dict[tuple, Dict] = {
        (str(r.encoder), str(r.finding), str(r.space), int(r.label_budget)):
            {"mean": float(r.refit_mean), "sd": float(r.refit_sd),
             "n_draws": int(r.refit_draws)}
        for r in refit_df.itertuples()} if not refit_df.empty else {}

    from sklearn.metrics import roc_auc_score
    widest = max(anchor_counts)
    native_w = max((int(v.get("dim", 0)) for v in cfg["encoder_panel"]["image"].values()), default=0)
    print_projected_peak("E12", {
        "relative representations at the widest anchor count":
            len(core_encs) * n_compact * widest * 4 / 2 ** 30,
        "native features": len(core_encs) * n_compact * native_w * 4 / 2 ** 30,
        "one full relative representation being read":
            len(manifest) * widest * 4 / 2 ** 30,
    }, note=f"{len(core_encs)} encoders over {n_compact} capped rows at "
            f"{sorted(anchor_counts)} anchors.")
    def _drop_other_anchor_counts(n_a: int) -> None:
        for key in [k for k in _rr_cache if k[1] != n_a]:
            _rr_cache.pop(key, None)

    def _oracle_unit(unit: str) -> List[Dict]:
        n_a, enc = int(unit.split("__")[0]), unit.split("__")[1]
        _drop_other_anchor_counts(n_a)
        rr = _rr(enc, n_a)
        if rr is None:
            raise MissingInput(f"no relative representation for {enc} at {n_a} anchors; "
                               f"run build_relative_reps after setting alignment.anchor_sweep.")
        per = per_finding_auroc(rr, labels_c, train_mask_c, test_mask_c)
        return [{"n_anchors": n_a, "encoder": enc, "finding": f, "oracle_auroc": float(v)}
                for f, v in per.items()]

    oracle_units = [f"{n_a}__{enc}" for n_a in anchor_counts for enc in core_encs]
    oracle_df = run_units_resumable(
        partial_dir=partials, group="e12_oracle", units=oracle_units,
        compute_unit=_oracle_unit,
        build_params={**shared_params, "anchor_counts": list(anchor_counts)},
        progress_desc="[E12] oracles", use_claims=True, status_file=status)
    oracle: Dict[tuple, float] = {}
    if not oracle_df.empty:
        for r in oracle_df.itertuples():
            oracle[(int(r.n_anchors), str(r.encoder), str(r.finding))] = float(r.oracle_auroc)

    def _transfer_unit(unit: str) -> List[Dict]:
        n_a, enc_train = int(unit.split("__")[0]), unit.split("__")[1]
        absent = [e for e in core_encs
                  if e != enc_train and not any((n_a, e, f) in oracle for f in findings)]
        if absent:
            raise MissingInput(f"{len(absent)} peer encoder(s) have no oracle at {n_a} anchors "
                               f"(first: {absent[:3]}); another job is still computing them.")
        _drop_other_anchor_counts(n_a)
        rr_train = _rr(enc_train, n_a)
        if rr_train is None:
            raise MissingInput(f"no relative representation for {enc_train} at {n_a} anchors.")
        peers = [e for e in core_encs if e != enc_train]
        probes = {}
        for fname in findings:
            col = labels_c[fname]
            tr = np.where(train_mask_c & np.isfinite(col))[0]
            tr = tr[np.isfinite(rr_train[tr, 0])]
            if len(tr) < 20 or len(np.unique(col[tr])) < 2:
                continue
            probes[fname] = fit_probe(rr_train[tr], col[tr])
        out: List[Dict] = []
        for enc_eval in tqdm(peers, desc=f"[E12] {enc_train[:18]}@{n_a} peers", unit="enc",
                             leave=False):
            heartbeat_claim(partials, f"e12_transfer__{unit}")
            rr_eval = _rr(enc_eval, n_a)
            n_before = len(out)
            for fname, (model, scaler) in probes.items():
                col = labels_c[fname]
                te = np.where(test_mask_c & np.isfinite(col))[0]
                te = te[np.isfinite(rr_eval[te, 0])]
                if len(te) < 10 or len(np.unique(col[te])) < 2:
                    continue
                yt, ys = eval_probe_scores(model, scaler, rr_eval[te], col[te])
                try:
                    auroc_xenc = float(roc_auc_score(yt, ys))
                except ValueError:
                    continue
                oracle_f = oracle.get((n_a, enc_eval, fname), float("nan"))
                row = {
                    "finding": fname, "enc_train": enc_train, "enc_eval": enc_eval,
                    "n_anchors": int(n_a),
                    "objective_train": _objective_of(cfg, enc_train),
                    "objective_eval": _objective_of(cfg, enc_eval),
                    "same_objective": _objective_of(cfg, enc_train) == _objective_of(cfg, enc_eval),
                    "oracle_auroc": round(oracle_f, 4),
                    "auroc_xenc": round(auroc_xenc, 4),
                    "retention": (round(auroc_xenc / oracle_f, 4)
                                  if np.isfinite(oracle_f) and oracle_f > 0 else float("nan")),
                    "n_test": int(len(te)),
                }
                for space in ("anchor", "native"):
                    for b in label_budgets:
                        rb = refit.get((enc_eval, fname, space, b),
                                       {"mean": float("nan"), "sd": float("nan"), "n_draws": 0})
                        v = rb["mean"]
                        row[f"refit_{space}_b{b}"] = (round(v, 4) if np.isfinite(v)
                                                      else float("nan"))
                        row[f"refit_{space}_b{b}_sd"] = (round(rb["sd"], 4) if np.isfinite(rb["sd"])
                                                         else float("nan"))
                        row[f"refit_{space}_b{b}_draws"] = int(rb["n_draws"])
                        row[f"transfer_minus_refit_{space}_b{b}"] = (
                            round(auroc_xenc - v, 4) if np.isfinite(v) else float("nan"))
                out.append(row)
            fresh = out[n_before:]
        return out

    transfer_units = [f"{n_a}__{enc}" for n_a in anchor_counts for enc in core_encs]
    df = run_units_resumable(
        partial_dir=partials, group="e12_transfer", units=transfer_units,
        compute_unit=_transfer_unit,
        build_params={**shared_params, "anchor_counts": list(anchor_counts),
                      "budgets": list(label_budgets)},
        progress_desc="[E12] transfer", use_claims=True, status_file=status)
    n_skipped = int(df.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(
            f"{n_skipped} of {len(transfer_units)} transfer units could not run because a peer's "
            f"oracle was absent, so the table would be short of encoder pairs. Every finished "
            f"unit is cached and skips; run this stage again once the oracle units are done.")
    if df.empty:
        raise MissingInput("E12 produced no transfer rows; run build_relative_reps before this stage.")
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, _sources)

    _paired_summaries(df, out_dir, label_budgets, primary_anchors)
    _cross_site_probe(cfg, global_config_path, findings, labels, manifest,
                      out_dir, primary_anchors, partials, status)
    append_status(status, f"E12 transfer written to {out_dir}")
    return out_dir


def _paired_summaries(df: pd.DataFrame, out_dir: str, budgets: List[int], primary: int) -> None:
    from Inference.report_utils import add_fdr, report_paired_diff, report_ratio
    rows: List[Dict] = []
    at_primary = df[df["n_anchors"] == primary]
    groupings = {"finding": ["finding"], "encoder_pair": ["enc_train", "enc_eval"],
                 "objective_pair": ["objective_train", "objective_eval"],
                 "anchor_budget": ["n_anchors"]}
    for level, cols in groupings.items():
        src = df if level == "anchor_budget" else at_primary
        for keys, grp in src.groupby(cols):
            keys = keys if isinstance(keys, tuple) else (keys,)
            base = {"level": level}
            base.update({c: k for c, k in zip(cols, keys)})
            r = report_ratio(grp["retention"].dropna().values, prefix="retention")
            r.update(base); r["n_rows"] = int(len(grp))
            rows.append(r)
            for space in ("anchor", "native"):
                for b in budgets:
                    col = f"refit_{space}_b{b}"
                    m = grp[["auroc_xenc", col]].dropna()
                    if len(m) < 5:
                        continue
                    d = report_paired_diff(m["auroc_xenc"].values, m[col].values,
                                           prefix="transfer_minus_refit")
                    d.update(base)
                    d.update({"comparator": f"refit_{space}", "label_budget": b,
                              "n_rows": int(len(m))})
                    rows.append(d)
    out = pd.DataFrame(rows)
    if "p_raw" in out.columns:
        out = add_fdr(out, family_cols=["level", "comparator"])
    write_csv_atomic(out, os.path.join(out_dir, "transfer_paired_summary.csv"))


def _cross_site_probe(cfg, global_config_path, findings, labels, manifest,
                      out_dir, primary_anchors, partials, status) -> None:
    from artifact.relative_reps import load_relative_rep
    from encoders.panel import list_encoder_names
    from Inference.report_utils import add_fdr, report_auroc, report_paired_diff
    from sklearn.metrics import roc_auc_score

    cons_dir = cfg["artifact"]["consensus_dir"]
    home = str(cfg["splits"]["anchor_source_site"])
    split = manifest["split"].astype(str).str.lower().values
    dataset = manifest["dataset"].astype(str).values
    encs = list_encoder_names(global_config_path, roles=["core"])

    def _unit(enc: str) -> List[Dict]:
        rows: List[Dict] = []
        rr = load_relative_rep(enc, cons_dir, n_anchors=primary_anchors, primary=primary_anchors)
        if rr is None:
            raise MissingInput(f"no relative representation for {enc}; run build_relative_reps before E12.")
        tr_mask = (dataset == home) & (split == "train")
        te_home = (dataset == home) & (split == "test")
        if tr_mask.sum() < 100 or te_home.sum() < 50:
            raise MissingInput(f"the anchor source site '{home}' has no usable train/test split.")
        for fname in tqdm(findings, desc=f"[E12] {enc[:20]} cross-site", unit="finding",
                          leave=False):
            heartbeat_claim(partials, f"e12_crosssite__{enc}")
            col = labels[fname]
            fin = np.isfinite(rr[:, 0])
            tr = np.where(tr_mask & np.isfinite(col) & fin)[0]
            if len(tr) < 20 or len(np.unique(col[tr])) < 2:
                continue
            model, scaler = fit_probe(rr[tr], col[tr])
            oh = np.where(te_home & np.isfinite(col) & fin)[0]
            if len(oh) < 20 or len(np.unique(col[oh])) < 2:
                continue
            oracle = eval_probe(model, scaler, rr[oh], col[oh])
            for site in sorted(set(dataset)):
                if site == home:
                    continue
                te = np.where((dataset == site) & np.isfinite(col) & fin)[0]
                if len(te) < 20 or len(np.unique(col[te])) < 2:
                    continue
                yt, ys = eval_probe_scores(model, scaler, rr[te], col[te])
                pat = (manifest["subject_id"].astype(str).values[te]
                       if "subject_id" in manifest.columns else None)
                rep = report_auroc(yt, ys, prefix="auroc", cluster_ids=pat)
                auroc = float(roc_auc_score(yt, ys))
                rep.update({"encoder": enc, "finding": fname, "site_train": home,
                            "site_eval": site, "oracle_auroc": round(oracle, 4),
                            "auroc_raw": auroc,
                            "retention": (round(auroc / oracle, 4)
                                          if np.isfinite(oracle) and oracle > 0 else float("nan")),
                            "n_test": int(len(te)), "space": "anchor_relative",
                            "consensus_dependent": False})
                rows.append(rep)
        return rows

    df = run_units_resumable(
        partial_dir=partials, group="e12_crosssite", units=list(encs), compute_unit=_unit,
        build_params={"home": home, "primary_anchors": primary_anchors,
                      "findings": list(findings),
                      "manifest": fingerprint_file(cfg["cxr"]["pool_manifest_csv"])},
        progress_desc="[E12] cross-site", use_claims=True, status_file=status)
    if int(df.attrs.get("n_skipped", 0)):
        raise MissingInput(f"{df.attrs['n_skipped']} of {len(encs)} cross-site units could not "
                           f"run; the finished ones are cached and skip.")
    rows = df.to_dict("records")
    if not rows:
        raise MissingInput("the cross-site probe produced no rows.")
    write_csv_atomic(df, os.path.join(out_dir, "cross_site_probe.csv"))

    summ: List[Dict] = []
    for site, grp in df.groupby("site_eval"):
        m = grp[["auroc_raw", "oracle_auroc"]].dropna()
        if len(m) < 5:
            continue
        d = report_paired_diff(m["auroc_raw"].values, m["oracle_auroc"].values,
                               prefix="site_minus_oracle")
        d.update({"site_eval": site, "n_rows": int(len(m)),
                  "n_findings": int(grp["finding"].nunique()),
                  "n_encoders": int(grp["encoder"].nunique())})
        summ.append(d)
    write_csv_atomic(add_fdr(pd.DataFrame(summ), family_cols=None),
                     os.path.join(out_dir, "cross_site_paired_summary.csv"))
