"""
artifact/universal_probe.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import permutations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alignment.linear_probe import fit_probe, eval_probe, per_finding_auroc, mean_auroc
from artifact.relative_reps import load_relative_rep
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

import warnings
warnings.filterwarnings("ignore")


def main_universal_probe(global_config_path: str) -> str:
    cfg      = read_config(global_config_path)["Convergence"]
    aln_cfg  = cfg["alignment"]
    emb_dir  = cfg["embeddings"]["output_dir"]
    cons_dir = aln_cfg["consensus_dir"]
    out_dir  = os.path.join(aln_cfg["results_base_dir"], "results_e7_artifact")
    os.makedirs(out_dir, exist_ok=True)
    out_csv  = os.path.join(out_dir, "cross_encoder_probe.csv")
    # Panel-aware resume below (skip train-encoders already in final/partial CSV).

    manifest   = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    findings   = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]
    man_ids    = manifest["case_id"].astype(str).values
    labels     = {f: pd.to_numeric(manifest[f], errors="coerce").values.astype(float)
                  for f in findings}

    # Train/test split from manifest
    seed = int(cfg.get("seed", 42))
    rng  = np.random.RandomState(seed)
    idx  = rng.permutation(len(manifest))
    cut  = int(0.7 * len(idx))
    train_mask = np.zeros(len(manifest), dtype=bool)
    test_mask  = np.zeros(len(manifest), dtype=bool)
    train_mask[idx[:cut]] = True
    test_mask[idx[cut:]]  = True

    from encoders.image_encoders import list_encoder_names
    core_encs = list_encoder_names(global_config_path, roles=["core"])

    probe_max_train = int(cfg["embeddings"].get("probe_max_train", 50000))
    probe_max_test  = int(cfg["embeddings"].get("probe_max_test", 20000))

    # Precompute capped train/test index pools (shared across encoders/findings).
    train_pool = np.where(train_mask)[0]
    test_pool  = np.where(test_mask)[0]
    if len(train_pool) > probe_max_train:
        train_pool = np.sort(rng.choice(train_pool, probe_max_train, replace=False))
    if len(test_pool) > probe_max_test:
        test_pool = np.sort(rng.choice(test_pool, probe_max_test, replace=False))

    kept_rows = np.sort(np.unique(np.concatenate([train_pool, test_pool])))
    row_to_compact = {int(r): i for i, r in enumerate(kept_rows)}
    # Remap masks and labels into compact space.
    train_local = np.array([row_to_compact[int(r)] for r in train_pool])
    test_local  = np.array([row_to_compact[int(r)] for r in test_pool])
    n_compact = len(kept_rows)
    train_mask_c = np.zeros(n_compact, dtype=bool); train_mask_c[train_local] = True
    test_mask_c  = np.zeros(n_compact, dtype=bool); test_mask_c[test_local]  = True
    labels_c = {f: v[kept_rows] for f, v in labels.items()}   # compact labels

    _rr_cache: dict = {}
    def _get_rr(enc):
        if enc not in _rr_cache:
            full = load_relative_rep(enc, cons_dir)
            _rr_cache[enc] = full[kept_rows] if full is not None else None
        return _rr_cache[enc]

    from tqdm import tqdm
    partial_csv = os.path.join(out_dir, "cross_encoder_probe.partial.csv")
    done_train = set()
    prior_frames = []
    for src in (out_csv, partial_csv):
        if os.path.exists(src):
            try:
                pf = pd.read_csv(src)
                if "enc_train" in pf.columns:
                    done_train |= set(pf["enc_train"].unique())
                    prior_frames.append(pf)
            except Exception as e:
                print(f"could not read {os.path.basename(src)} ({e}).")
    if done_train:
        print(f"{len(done_train)} training-encoders already done; "
              f"computing only missing.")
    if prior_frames and not os.path.exists(partial_csv):
        pd.concat(prior_frames, ignore_index=True).to_csv(partial_csv, index=False)

    oracle_cache: dict = {}
    for enc in tqdm(core_encs, desc="[E7/probe] oracles", unit="enc"):
        rr = _get_rr(enc)
        if rr is None:
            continue
        oracle_cache[enc] = per_finding_auroc(rr, labels_c, train_mask_c, test_mask_c)

    for enc_train in tqdm(core_encs, desc="[E7/probe] train-enc", unit="enc"):
        if enc_train in done_train:
            continue
        rr_train = _get_rr(enc_train)
        if rr_train is None or enc_train not in oracle_cache:
            continue
        oracle = mean_auroc(oracle_cache[enc_train])

        rows_train = []
        from alignment.linear_probe import eval_probe_scores
        from sklearn.metrics import roc_auc_score

        probes = {}
        for fname in findings:
            col = labels_c[fname]
            tr_idx = np.where(train_mask_c & np.isfinite(col))[0]
            tr_valid = tr_idx[np.isfinite(rr_train[tr_idx, 0])]
            if len(tr_valid) < 20 or len(np.unique(col[tr_valid])) < 2:
                continue
            probes[fname] = fit_probe(rr_train[tr_valid], col[tr_valid])

        for enc_eval in core_encs:
            if enc_eval == enc_train:
                continue
            rr_eval = _get_rr(enc_eval)
            if rr_eval is None or enc_eval not in oracle_cache:
                continue
            oracle_dict = oracle_cache[enc_eval]

            # Evaluate the pre-fit probes in enc_eval's space (cheap: no refitting).
            for fname in findings:
                if fname not in probes:
                    continue
                col = labels_c[fname]
                te_idx = np.where(test_mask_c & np.isfinite(col))[0]
                te_valid = te_idx[np.isfinite(rr_eval[te_idx, 0])]
                if len(te_valid) < 10 or len(np.unique(col[te_valid])) < 2:
                    continue
                model, scaler = probes[fname]
                yt, ys = eval_probe_scores(model, scaler, rr_eval[te_valid], col[te_valid])
                try:
                    auroc_xenc = float(roc_auc_score(yt, ys))
                except ValueError:
                    auroc_xenc = float("nan")
                oracle_f = oracle_dict.get(fname, float("nan"))
                row = {
                    "finding":         fname,
                    "enc_train":       enc_train,
                    "enc_eval":        enc_eval,
                    "oracle_auroc":    round(oracle_f, 4),
                    "retention":       round(auroc_xenc / oracle_f, 4)
                                       if np.isfinite(oracle_f) and oracle_f > 0
                                       and np.isfinite(auroc_xenc) else float("nan"),
                    "auroc_xenc":      round(auroc_xenc, 4),
                }
                rows_train.append(row)
        if rows_train:
            hdr = not os.path.exists(partial_csv)
            pd.DataFrame(rows_train).to_csv(partial_csv, mode="a", header=hdr, index=False)
        done_train.add(enc_train)

    rows_ce_df = pd.read_csv(partial_csv) if os.path.exists(partial_csv) else pd.DataFrame()
    rows_ce_df.to_csv(out_csv, index=False)
    try:
        os.remove(partial_csv)
    except OSError:
        pass

    _cross_site_probe(cfg, findings, labels, manifest, cons_dir, out_dir, seed)

    gold_csv = cfg["reader_study"]["gold_csv"]
    if os.path.exists(gold_csv):
        _gold_probe(gold_csv, core_encs, cons_dir, out_dir, seed)

    return out_dir


def _cross_site_probe(cfg, findings, labels, manifest, cons_dir, out_dir, seed):
    consensus_path = os.path.join(cons_dir, "consensus.npy")
    if not os.path.exists(consensus_path):
        return
    cons = np.load(consensus_path).astype(np.float32)

    kept_path = os.path.join(cons_dir, "consensus_kept_case_idx.npy")
    if os.path.exists(kept_path):
        kept_idx = np.load(kept_path)
        if len(kept_idx) == cons.shape[0]:
            manifest = manifest.iloc[kept_idx].reset_index(drop=True)
            labels = {f: v[kept_idx] for f, v in labels.items()}
    man_ids = manifest["case_id"].astype(str).values

    mimic_mask   = (manifest["dataset"] == "mimic").values
    site_rows    = []
    site_names   = manifest["dataset"].unique()

    # Oracle = train on MIMIC, eval on MIMIC
    rng = np.random.RandomState(seed)
    mimic_idx = np.where(mimic_mask)[0]
    rng.shuffle(mimic_idx)
    cut = int(0.7 * len(mimic_idx))
    tr_mimic = np.zeros(len(manifest), dtype=bool)
    te_mimic = np.zeros(len(manifest), dtype=bool)
    tr_mimic[mimic_idx[:cut]] = True
    te_mimic[mimic_idx[cut:]] = True

    for fname in findings:
        col    = labels[fname]
        tr_idx = np.where(tr_mimic & np.isfinite(col) & np.isfinite(cons[:, 0]))[0]
        if len(tr_idx) < 20 or len(np.unique(col[tr_idx])) < 2:
            continue
        model, scaler = fit_probe(cons[tr_idx], col[tr_idx])
        # Oracle: eval on MIMIC test
        te_oracle = np.where(te_mimic & np.isfinite(col) & np.isfinite(cons[:, 0]))[0]
        oracle_auroc = eval_probe(model, scaler, cons[te_oracle], col[te_oracle]) \
                       if len(te_oracle) >= 10 else float("nan")

        for site in site_names:
            site_mask = (manifest["dataset"] == site).values
            te_idx    = np.where(site_mask & np.isfinite(col) & np.isfinite(cons[:, 0]))[0]
            if len(te_idx) < 10:
                continue
            from alignment.linear_probe import eval_probe_scores
            from Inference.report_utils import report_auroc
            yt, ys   = eval_probe_scores(model, scaler, cons[te_idx], col[te_idx])
            auroc_rep = report_auroc(yt, ys, prefix="auroc")
            auroc    = auroc_rep["auroc_mean_raw"]
            row = {
                "finding":        fname,
                "site_train":     "mimic",
                "site_eval":      site,
                "oracle_auroc":   round(oracle_auroc, 4),
                "retention":      round(auroc / oracle_auroc, 4)
                                  if np.isfinite(oracle_auroc) and oracle_auroc > 0
                                  and np.isfinite(auroc) else float("nan"),
            }
            row.update(auroc_rep)
            site_rows.append(row)

    if site_rows:
        pd.DataFrame(site_rows).to_csv(
            os.path.join(out_dir, "cross_site_probe.csv"), index=False
        )


def _gold_probe(gold_csv, encoders, cons_dir, out_dir, seed):
    gold = read_csv_defensively(gold_csv)
    if "finding" not in gold.columns or "label" not in gold.columns:
        return
    consensus_path = os.path.join(cons_dir, "consensus.npy")
    if not os.path.exists(consensus_path):
        return
    # minimal re-test on gold labels using consensus space
    gold_ids  = gold["case_id"].astype(str).values
    gold_labs = pd.to_numeric(gold["label"], errors="coerce").values
    gold.to_csv(os.path.join(out_dir, "gold_label_probe_raw.csv"), index=False)