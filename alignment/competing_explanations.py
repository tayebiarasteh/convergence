"""
alignment/competing_explanations.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import itertools
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Inference.report_utils import add_fdr, report_metric, report_paired_diff, report_permutation_2
from Inference.resume_utils import (MissingInput, append_status, run_units_resumable,
                                    status_path, write_csv_atomic)
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _pool_manifest(cfg: Dict, pool: str) -> str:
    return {"cxr_pool": cfg["cxr"]["pool_manifest_csv"],
            "taix_pool": cfg["taix"]["pool_manifest_csv"],
            "rexgradient_pool": cfg["rexgradient"]["pool_manifest_csv"]}[pool]


def _matched_draw(rng, ids: np.ndarray, n: int) -> Optional[np.ndarray]:
    if len(ids) < n:
        return None
    return ids[rng.choice(len(ids), size=n, replace=False)]


def _align(ea, ia, eb, ib, ids, k):
    from alignment.metrics import mknn
    pa = {str(c): i for i, c in enumerate(ia)}
    pb = {str(c): i for i, c in enumerate(ib)}
    ka = [pa[c] for c in ids]
    kb = [pb[c] for c in ids]
    return mknn(ea[ka], eb[kb], k=k)


def _acquisition_arm(cfg, rex, ea, ia, eb, ib, common, matched_n, k, seed, unit) -> List[Dict]:
    man = rex.set_index(rex["case_id"].astype(str))
    shared = [c for c in common if c in man.index]
    if not shared:
        return []
    man = man.loc[shared]
    rng = np.random.default_rng(seed)
    matched_n = int(cfg["alignment"].get("e10_acquisition_n", matched_n))
    k = max(2, int(round(float(cfg["alignment"]["k_fraction"]) * matched_n)))
    within, across, names = [], [], []
    for mfr, grp in man.groupby("manufacturer"):
        sel = _matched_draw(rng, grp.index.values, matched_n)
        if sel is None:
            continue
        within.append(_align(ea, ia, eb, ib, sel, k))
        names.append(str(mfr))
    if len(within) < 2:
        return []
    all_ids = man.index.values
    for _ in range(len(within)):
        sel = _matched_draw(rng, all_ids, matched_n)
        if sel is None:
            break
        across.append(_align(ea, ia, eb, ib, sel, k))
    if len(across) < 2:
        return []
    r = report_permutation_2(np.array(within), np.array(across),
                            prefix="within_minus_across_manufacturer",
                            is_percent=True, tested="within_manufacturer")
    r.update({"pair": unit, "pool": "rexgradient_pool", "test": "acquisition",
              "n_manufacturers": len(within), "manufacturers": ";".join(sorted(names)),
              "matched_n": matched_n, "k": k})
    return [r]


def _require_harmonized_cache(cfg, enc: str, pool: str) -> None:
    from encoders.cache_utils import (HARMONIZED_PIPELINE, embedding_cache_path,
                                      harmonized_cache_name)
    from Inference.resume_utils import read_build_params
    path = embedding_cache_path(cfg, enc, harmonized_cache_name(pool, cfg))
    if not os.path.exists(path):
        raise MissingInput(f"no harmonized cache for ({enc}, {pool}); run main_extract_image_embeddings with "
                           f"embeddings.harmonized_preprocessing.enabled and "
                           f"alignment.e10_harmonized_pools naming this pool.")
    rec = read_build_params(path) or {}
    if rec.get("pipeline") != HARMONIZED_PIPELINE:
        raise MissingInput(f"the harmonized cache for ({enc}, {pool}) records pipeline "
                           f"{rec.get('pipeline')!r} and this code reads {HARMONIZED_PIPELINE!r}; "
                           f"re-run main_extract_image_embeddings for {enc} so the cache rebuilds under the current rule.")


def _harmonized_cache_ok(cfg, enc: str, pool: str) -> bool:
    try:
        _require_harmonized_cache(cfg, enc, pool)
        return True
    except MissingInput:
        return False


def _preprocessing_arm(cfg, ea, ia, eb, ib, a, b, pool, matched_n, k, seed, unit) -> List[Dict]:
    from encoders.cache_utils import harmonized_cache_name, load_embedding_cache
    hname = harmonized_cache_name(pool, cfg)
    for enc in (a, b):
        _require_harmonized_cache(cfg, enc, pool)
    ha, hia = load_embedding_cache(cfg, a, hname)
    hb, hib = load_embedding_cache(cfg, b, hname)
    common = sorted(set(map(str, ia)) & set(map(str, ib))
                    & set(map(str, hia)) & set(map(str, hib)))
    if len(common) < matched_n:
        return []
    rng = np.random.default_rng(seed)
    n_rep = max(4, int(len(common) // matched_n))
    nat, harm = [], []
    for _ in range(n_rep):
        sel = _matched_draw(rng, np.array(common), matched_n)
        nat.append(_align(ea, ia, eb, ib, sel, k))
        harm.append(_align(ha, hia, hb, hib, sel, k))
    r = report_paired_diff(np.array(harm), np.array(nat), prefix="harmonized_minus_native",
                           is_percent=True)
    r.update({"pair": unit, "pool": pool, "test": "preprocessing",
              "preprocessing": "harmonized_vs_native", "n_draws": n_rep,
              "native_mean_raw": float(np.mean(nat)),
              "harmonized_mean_raw": float(np.mean(harm)),
              "matched_n": matched_n, "k": k})
    return [r]


def _exposure_arm(cfg, ea, ia, eb, ib, common, pool, matched_n, k, seed, unit,
                  site_restrict=None) -> List[Dict]:
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    draws = {pool: np.array(sorted(common))}
    if site_restrict is not None:
        for site, ids in site_restrict.items():
            draws[f"{pool}:{site}"] = np.array(sorted(set(common) & set(ids)))
    unc = set((cfg.get("provenance", {}) or {}).get("uncontaminated_sources", []))
    for tag, ids in draws.items():
        sel = _matched_draw(rng, ids, matched_n)
        if sel is None:
            continue
        v = _align(ea, ia, eb, ib, sel, k)
        base = tag.split(":")[0].replace("_pool", "")
        rows.append({"pair": unit, "pool": tag, "test": "pretraining_exposure",
                     "preprocessing": "native", "alignment_raw": float(v),
                     "panel_saw_this_site": base not in unc,
                     "matched_n": matched_n, "k": k})
    return rows


def vendor_name(value) -> str:
    s = str(value).strip()
    if s.startswith("[") and s.endswith("]"):
        import ast
        try:
            parts = [str(p).strip() for p in ast.literal_eval(s) if str(p).strip()]
            return " + ".join(parts) if parts else "nan"
        except (ValueError, SyntaxError):
            return s.strip("[]").strip("'\" ")
    return s


def _scanner_probe(global_config_path: str, cfg: Dict, out_dir: str, seed: int) -> str:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from Inference.regimes import regime_extra
    from encoders.cache_utils import load_embedding_cache
    from encoders.panel import list_encoder_names
    from alignment.consensus import compute_relative_reps, read_anchor_case_ids
    from Inference.report_utils import report_accuracy
    from Inference.resume_utils import fingerprint_file, heartbeat_claim, record_sources

    out_csv = os.path.join(out_dir, "acquisition_predictability.csv")
    rex_csv = _pool_manifest(cfg, "rexgradient_pool")
    if not os.path.exists(rex_csv):
        raise MissingInput("the ReXGradient pool manifest is absent; run main_build_rexgradient_pool before E10.")
    man = read_csv_defensively(rex_csv)
    anchors_path = os.path.join(cfg["alignment"]["consensus_dir"], "anchor_case_ids.csv")
    anchor_ids = read_anchor_case_ids(anchors_path)

    targets = [t for t in ("manufacturer", "institution") if t in man.columns]
    if not targets:
        raise MissingInput("the ReXGradient pool carries neither manufacturer nor institution.")
    for t in targets:
        man[t] = man[t].map(vendor_name)
    partial_dir = os.path.join(out_dir, "partials_scanner")
    encoders = list_encoder_names(global_config_path, roles=["core"])

    def compute_unit(enc: str) -> List[Dict]:
        rows: List[Dict] = []
        heartbeat_claim(partial_dir, f"e10_scanner__{enc}")
        emb, ids = load_embedding_cache(cfg, enc, "rexgradient_pool")
        pos = {str(c): i for i, c in enumerate(ids)}
        sub = man[man["case_id"].astype(str).isin(pos)].reset_index(drop=True)
        if len(sub) < 500:
            return rows
        idx = np.array([pos[c] for c in sub["case_id"].astype(str)])
        X_native = emb[idx]

        n_primary = int(cfg["alignment"].get("primary_anchors", 1024))
        aemb, aids = load_embedding_cache(cfg, enc, "cxr_pool")
        apos = {str(c): i for i, c in enumerate(aids)}
        have = [c for c in anchor_ids[:n_primary] if c in apos]
        X_shared = None
        if len(have) >= 64:
            A = aemb[[apos[c] for c in have]]
            X_shared = (X_native @ A.T).astype(np.float32, copy=False)

        pcol = "subject_id" if "subject_id" in sub.columns else None
        rng = np.random.RandomState(seed)
        if pcol is not None and sub[pcol].notna().any():
            subs = sub[pcol].astype(str).unique()
            rng.shuffle(subs)
            train_pat = set(subs[: int(0.7 * len(subs))])
            tr = sub[pcol].astype(str).isin(train_pat).values
        else:
            tr = rng.random_sample(len(sub)) < 0.7
        te = ~tr

        for target in targets:
            y = sub[target].astype(str).values
            keep = pd.Series(y).notna().values & (pd.Series(y).astype(str) != "nan").values
            counts = pd.Series(y[keep & tr]).value_counts()
            classes = counts[counts >= 20].index
            m = keep & pd.Series(y).isin(classes).values
            if m.sum() < 200 or len(classes) < 2:
                continue
            for space, X in (("native", X_native), ("shared_anchor", X_shared)):
                if X is None:
                    continue
                heartbeat_claim(partial_dir, f"e10_scanner__{enc}")
                fin = np.isfinite(X).all(axis=1)
                a, b = m & tr & fin, m & te & fin
                if a.sum() < 100 or b.sum() < 50 or len(np.unique(y[a])) < 2:
                    continue
                sc = StandardScaler().fit(X[a])
                clf = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                                         random_state=seed).fit(sc.transform(X[a]), y[a])
                correct = (clf.predict(sc.transform(X[b])) == y[b]).astype(float)
                acc = float(correct.mean())
                base = float(pd.Series(y[b]).value_counts(normalize=True).max())
                pat = sub[pcol].astype(str).values[b] if pcol is not None else None
                rep = report_accuracy(correct, prefix="accuracy", cluster_ids=pat)
                rep.update({"encoder": enc, "target": target, "space": space,
                            "majority_baseline_raw": base,
                            "accuracy_above_baseline_raw": acc - base,
                            "n_classes": int(len(classes)), "classes": ";".join(sorted(classes)),
                            "n_train": int(a.sum()),
                            "n_test": int(b.sum()), "test": "acquisition_predictability",
                            "patient_disjoint": bool(pcol is not None)})
                rows.append(rep)
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="e10_scanner", units=list(encoders),
        compute_unit=compute_unit, progress_desc="[E10] scanner probe",
        build_params={"seed": seed, "anchors": fingerprint_file(anchors_path),
                      "manifest": fingerprint_file(rex_csv), **regime_extra("e10_scanner")},
        use_claims=True, status_file=status_path(cfg, "e10_competing"))
    if int(df.attrs.get("n_skipped", 0)):
        raise MissingInput(f"{df.attrs['n_skipped']} of {len(encoders)} encoders could not run "
                           f"the scanner probe; the finished ones are cached and skip.")
    if df.empty:
        raise MissingInput("the scanner probe produced no rows; extract the ReXGradient pool "
                           "(main_extract_image_embeddings) and build the anchors (main_build_consensus) first.")
    df = add_fdr(df, family_cols=["target", "space"])
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, [rex_csv, anchors_path], extra=regime_extra("e10_scanner"))
    return out_csv


def main_competing_explanations(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln = cfg["alignment"]
    out_dir = os.path.join(aln["results_base_dir"], "results_e10_competing")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "competing_explanations.csv")
    status = status_path(cfg, "e10_competing")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(os.path.join(out_dir, "partials"))

    from encoders.cache_utils import load_embedding_cache
    from encoders.panel import list_encoder_names
    encoders = list_encoder_names(global_config_path, roles=["core"])
    matched_n = int(aln["matched_n"])
    k = max(2, int(round(float(aln["k_fraction"]) * matched_n)))
    seed = int(cfg["seed"])
    harm_pools = set(aln.get("e10_harmonized_pools", []))
    units = [f"{a}__vs__{b}" for a, b in itertools.combinations(encoders, 2)]

    rex_csv = _pool_manifest(cfg, "rexgradient_pool")
    if not os.path.exists(rex_csv):
        raise MissingInput("the ReXGradient pool manifest is absent; run main_build_rexgradient_pool before E10.")
    rex = read_csv_defensively(rex_csv)
    cxr_man = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    home = str(cfg["splits"]["anchor_source_site"])
    home_ids = set(cxr_man.loc[cxr_man["dataset"].astype(str) == home,
                               "case_id"].astype(str))
    home_ap_ids = set()
    if "view" in cxr_man.columns:
        home_ap_ids = set(cxr_man.loc[(cxr_man["dataset"].astype(str) == home)
                                      & (cxr_man["view"].astype(str).str.upper() == "AP"),
                                      "case_id"].astype(str))

    harm_checks = [p for p in aln["e10_pools"] if p in harm_pools]

    def compute_unit(unit: str) -> List[Dict]:
        a, b = unit.split("__vs__")
        for hp in harm_checks:
            for enc in (a, b):
                _require_harmonized_cache(cfg, enc, hp)
        rows: List[Dict] = []
        for pool in aln["e10_pools"]:
            ea, ia = load_embedding_cache(cfg, a, pool)
            eb, ib = load_embedding_cache(cfg, b, pool)
            common = sorted(set(map(str, ia)) & set(map(str, ib)))
            if len(common) < matched_n:
                continue
            if pool == "rexgradient_pool":
                rows += _acquisition_arm(cfg, rex, ea, ia, eb, ib, common,
                                         matched_n, k, seed, unit)
                rows += _exposure_arm(cfg, ea, ia, eb, ib, common, pool,
                                      matched_n, k, seed, unit)
            else:
                restrict = None
                if pool == "cxr_pool":
                    restrict = {home: home_ids}
                    if home_ap_ids:
                        restrict[f"{home}_ap"] = home_ap_ids
                rows += _exposure_arm(cfg, ea, ia, eb, ib, common, pool,
                                      matched_n, k, seed, unit, site_restrict=restrict)
            if pool in harm_pools:
                rows += _preprocessing_arm(cfg, ea, ia, eb, ib, a, b, pool,
                                           matched_n, k, seed, unit)
        return rows

    not_current = sorted({enc for enc in encoders for hp in harm_checks
                          if not _harmonized_cache_ok(cfg, enc, hp)})

    df = run_units_resumable(
        partial_dir=os.path.join(out_dir, "partials"), group="e10", units=units,
        compute_unit=compute_unit, progress_desc="[E10] encoder pairs",
        build_params={"matched_n": matched_n, "k": k, "seed": seed,
                      "arms": ["preprocessing", "acquisition", "pretraining_exposure"]},
        use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("E10 produced no rows; extract the ReXGradient and TAIX pools first.")
    n_skipped = int(df.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(f"{n_skipped} of {len(units)} pairs could not run, so the correction "
                           f"family and the crossed interval would be computed over a subset. "
                           f"Fix the inputs named above and re-run this stage. The "
                           f"{len(units) - n_skipped} finished pairs are cached and skip.")
    df = add_fdr(df, family_cols=["test", "pool"])
    if "manufacturers" in df.columns:
        df["manufacturers"] = df["manufacturers"].map(
            lambda s: ";".join(vendor_name(x) for x in str(s).split(";")) if pd.notna(s) else s)
    import hashlib
    from Inference.resume_utils import output_is_current, record_sources
    rows_digest = hashlib.blake2b(pd.util.hash_pandas_object(df, index=False).values.tobytes(),
                                  digest_size=16).hexdigest()
    exposure_csv = os.path.join(out_dir, "exposure_paired_summary.csv")
    if (not force and os.path.exists(exposure_csv)
            and output_is_current(out_csv, [], owner="E10", extra={"rows_digest": rows_digest})):
        _scanner_probe(global_config_path, cfg, out_dir, seed)
        return out_csv
    write_csv_atomic(df, out_csv)

    exp = df[df["test"] == "pretraining_exposure"]
    if not exp.empty and "alignment_raw" in exp.columns:
        from Inference.report_utils import report_paired_diff
        from Inference.stats_utils import jackknife_over_units
        wide = exp.pivot_table(index="pair", columns="pool", values="alignment_raw",
                               aggfunc="first")
        seen_tags = [t for t in (f"cxr_pool:{home}_ap", f"cxr_pool:{home}", "cxr_pool")
                     if t in wide.columns]
        sums = []
        for unseen_tag, seen_tag in itertools.product(("taix_pool", "rexgradient_pool"), seen_tags):
            if unseen_tag not in wide.columns:
                continue
            m = wide[[unseen_tag, seen_tag]].dropna()
            if len(m) < 5:
                continue
            r = report_paired_diff(m[unseen_tag].values, m[seen_tag].values,
                                   prefix="unseen_minus_seen")
            enc_a = [p.split("__vs__")[0] for p in m.index]
            enc_b = [p.split("__vs__")[1] for p in m.index]
            jk = jackknife_over_units(m[unseen_tag].values - m[seen_tag].values,
                                      enc_a, enc_b)
            r.update({"unseen_site": unseen_tag, "seen_site": seen_tag,
                      "seen_view_matched": bool(seen_tag.endswith("_ap")),
                      "n_pairs": int(len(m)),
                      "crossed_std_raw": jk["std"],
                      "crossed_ci_low_raw": jk["ci_lower"],
                      "crossed_ci_high_raw": jk["ci_upper"],
                      "crossed_n_encoders": jk.get("n_units")})
            sums.append(r)
        if sums:
            write_csv_atomic(add_fdr(pd.DataFrame(sums), family_cols=None), exposure_csv)
    record_sources(out_csv, [], extra={"rows_digest": rows_digest})

    _scanner_probe(global_config_path, cfg, out_dir, seed)

    for arm in ("preprocessing", "acquisition", "pretraining_exposure"):
        n = int((df["test"] == arm).sum())
        if n == 0:
            raise MissingInput(
                f"E10's {arm} arm produced 0 rows, so the alternative it exists to rule out is "
                f"untested. Check alignment.e10_pools and e10_harmonized_pools.")
    append_status(status, f"E10 wrote {len(df)} rows over {df.pair.nunique()} encoder pairs")
    return out_csv
