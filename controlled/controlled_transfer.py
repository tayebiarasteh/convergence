"""
controlled/controlled_transfer.py
Created on August 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from Inference.report_utils import add_fdr, report_spearman
from Inference.resume_utils import (MissingInput, append_status, heartbeat_claim,
                                    run_units_resumable, status_path, write_csv_atomic)
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _cell_paths(cfg: Dict, run_id: str):
    emb_dir = os.path.join(cfg["controlled"]["results_e5_dir"], "embeddings")
    return (os.path.join(emb_dir, f"{run_id}.npz"),
            os.path.join(emb_dir, f"{run_id}__anchors.npz"))


def _load_npz(path: str):
    d = np.load(path, allow_pickle=True)
    return d["embeddings"].astype(np.float32, copy=False), d["case_ids"].astype(str)


def _relative_rep(cfg: Dict, run_id: str, n_anchors: int):
    emb_path, anc_path = _cell_paths(cfg, run_id)
    for p, what in ((emb_path, "held-out embeddings"), (anc_path, "anchor embeddings")):
        if not os.path.exists(p):
            raise MissingInput(f"no {what} for {run_id} at {p}; run main_converge_eval before main_controlled_transfer.")
    emb, ids = _load_npz(emb_path)
    anc, _aid = _load_npz(anc_path)
    rel = emb @ anc[:n_anchors].T
    return rel.astype(np.float32, copy=False), ids


def main_controlled_transfer(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    out_dir = ctrl["results_e5_dir"]
    out_csv = os.path.join(out_dir, "controlled_transfer.csv")
    link_csv = os.path.join(out_dir, "controlled_transfer_link.csv")
    status = status_path(cfg, "controlled_transfer")
    partial_dir = os.path.join(out_dir, "transfer_partials")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(partial_dir)

    n_anchors = int(cfg["alignment"].get("primary_anchors", 1024))
    seed = int(cfg.get("seed", 42))
    max_train = int(cfg["embeddings"].get("probe_max_train", 25000))
    max_test = int(cfg["embeddings"].get("probe_max_test", 20000))

    held_csv = os.path.join(out_dir, "heldout_cxr_manifest.csv")
    if not os.path.exists(held_csv):
        raise MissingInput(f"the chest held-out manifest is absent at {held_csv}; run main_converge_eval first.")
    man = read_csv_defensively(held_csv)
    from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
    findings = [f for f in CANONICAL_CXR_FINDINGS
                if f != "no_finding" and f in man.columns]
    split = man["split"].astype(str).str.lower().values
    fit_rows = split == "valid"
    ev_rows = split == "test"
    if fit_rows.sum() < 200 or ev_rows.sum() < 200:
        raise MissingInput(f"the held-out pool has {int(fit_rows.sum())} validation and "
                           f"{int(ev_rows.sum())} test rows; the transfer needs both.")
    if "subject_id" in man.columns:
        overlap = (set(man.loc[fit_rows, "subject_id"].astype(str))
                   & set(man.loc[ev_rows, "subject_id"].astype(str)))
        overlap.discard("nan")
        if overlap:
            raise ValueError(f"[controlled_transfer] {len(overlap)} patients sit in both the probe-fit and the "
                             f"probe-eval split; the released splits should keep them apart.")
    rng0 = np.random.RandomState(seed)

    def _cap_rows(mask: np.ndarray, cap: int) -> np.ndarray:
        idx = np.where(mask)[0]
        if len(idx) <= cap:
            return idx
        if "subject_id" in man.columns:
            subs = pd.unique(man.loc[idx, "subject_id"].astype(str))
            rng0.shuffle(subs)
            keep, taken = set(), 0
            counts = man.loc[idx, "subject_id"].astype(str).value_counts()
            for s in subs:
                n = int(counts.get(s, 0))
                if taken + n > cap:
                    continue
                keep.add(s); taken += n
                if taken >= cap:
                    break
            return idx[man.loc[idx, "subject_id"].astype(str).isin(keep).values]
        return np.sort(rng0.choice(idx, cap, replace=False))

    kept = np.sort(np.unique(np.concatenate([_cap_rows(fit_rows, max_train),
                                             _cap_rows(ev_rows, max_test)])))
    man = man.iloc[kept].reset_index(drop=True)
    split = man["split"].astype(str).str.lower().values
    fit_rows = split == "valid"
    ev_rows = split == "test"
    man_ids = man["case_id"].astype(str).values
    labels = {f: pd.to_numeric(man[f], errors="coerce").values.astype(float, copy=False)
              for f in findings}

    from controlled.matrix import (enumerate_disjoint_runs, enumerate_e5_runs,
                                   enumerate_frame_runs, pairing_group)
    chest = [r[0] for r in enumerate_e5_runs(global_config_path) if r[1] == "cxr"]
    chest += [r[0] for r in enumerate_frame_runs(global_config_path)]
    chest += [r[0] for r in enumerate_disjoint_runs(global_config_path)]
    ready = [rid for rid in chest
             if all(os.path.exists(p) for p in _cell_paths(cfg, rid))]
    if len(ready) < len(chest):
        missing = sorted(set(chest) - set(ready))
        raise MissingInput(f"{len(missing)} of {len(chest)} chest cells lack a held-out or an "
                           f"anchor cache (first: {missing[:3]}); train the controlled models and run main_converge_eval "
                           f"(with controlled.transfer.enabled) before this stage.")

    rel_cache: Dict[str, tuple] = {}

    def _rel(rid: str):
        if rid not in rel_cache:
            rel, ids = _relative_rep(cfg, rid, n_anchors)
            pos = {c: i for i, c in enumerate(ids)}
            take = np.array([pos.get(c, -1) for c in man_ids])
            out = np.full((len(man_ids), rel.shape[1]), np.nan, dtype=np.float32)
            ok = take >= 0
            out[ok] = rel[take[ok]]
            rel_cache[rid] = out
        return rel_cache[rid]

    oracle_probe: Dict[tuple, tuple] = {}

    def _fit_on(rid: str, f: str, row_mask: np.ndarray):
        from alignment.linear_probe import fit_probe
        col = labels[f]
        X = _rel(rid)
        m = row_mask & np.isfinite(col) & np.isfinite(X).all(axis=1)
        if m.sum() < 40 or len(np.unique(col[m])) < 2:
            return None
        return fit_probe(X[m], col[m])

    def _oracle(rid: str, f: str):
        key = (rid, f)
        if key not in oracle_probe:
            oracle_probe[key] = _fit_on(rid, f, fit_rows)
        return oracle_probe[key]

    def compute_unit(rid_train: str) -> List[Dict]:
        from alignment.linear_probe import eval_probe_scores
        from sklearn.metrics import roc_auc_score
        rows: List[Dict] = []
        group = pairing_group(rid_train)
        peers = [r for r in ready if r != rid_train and pairing_group(r) == group]
        probes = {}
        for f in tqdm(findings, desc=f"[controlled_transfer] {rid_train[:24]} probes", unit="finding",
                      leave=False):
            heartbeat_claim(partial_dir, f"transfer__{rid_train}")
            p = _fit_on(rid_train, f, fit_rows)
            if p is not None:
                probes[f] = p
        for rid_eval in tqdm(peers, desc=f"[controlled_transfer] {rid_train[:24]} peers", unit="cell",
                             leave=False):
            heartbeat_claim(partial_dir, f"transfer__{rid_train}")
            n_before = len(rows)
            Xe = _rel(rid_eval)
            for f, (model, scaler) in probes.items():
                col = labels[f]
                m = ev_rows & np.isfinite(col) & np.isfinite(Xe).all(axis=1)
                if m.sum() < 40 or len(np.unique(col[m])) < 2:
                    continue
                orc = _oracle(rid_eval, f)
                if orc is None:
                    continue
                yt, ys = eval_probe_scores(model, scaler, Xe[m], col[m])
                oy, oscore = eval_probe_scores(orc[0], orc[1], Xe[m], col[m])
                try:
                    a_x = float(roc_auc_score(yt, ys))
                    a_o = float(roc_auc_score(oy, oscore))
                except ValueError:
                    continue
                pa, pb = rid_train.split("__"), rid_eval.split("__")
                rows.append({
                    "enc_train": rid_train, "enc_eval": rid_eval, "group": group,
                    "finding": f, "objective_train": pa[1], "objective_eval": pb[1],
                    "same_objective": pa[1] == pb[1],
                    "same_training_data": pa[0] == pb[0],
                    "auroc_xenc_raw": round(a_x, 4), "oracle_auroc_raw": round(a_o, 4),
                    "retention_raw": (round(a_x / a_o, 4) if a_o > 0 else float("nan")),
                    "n_test": int(m.sum()), "n_anchors": n_anchors})
            new = rows[n_before:]
            if new:
                med = float(np.median([r["retention_raw"] for r in new]))
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="transfer",
        units=ready, compute_unit=compute_unit,
        build_params={"n_anchors": n_anchors, "seed": seed,
                      "max_train": max_train, "max_test": max_test},
        progress_desc="[controlled_transfer] train cells", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("the controlled transfer produced no rows.")
    write_csv_atomic(df, out_csv)

    align_csv = os.path.join(out_dir, "e5_alignment_table.csv")
    from controlled.matrix import require_e5_family_complete
    require_e5_family_complete(global_config_path,
                               align_csv, os.path.join(out_dir, "e5_cell_utility.csv"),
                               owner="controlled_transfer link")
    link_rows: List[Dict] = []
    if os.path.exists(align_csv):
        al = read_csv_defensively(align_csv)
        al["pair_lo"] = np.minimum(al["run_id_a"].astype(str), al["run_id_b"].astype(str))
        al["pair_hi"] = np.maximum(al["run_id_a"].astype(str), al["run_id_b"].astype(str))
        a = al.groupby(["pair_lo", "pair_hi"])["mknn"].mean().rename("alignment").reset_index()
        tr = df.copy()
        tr["pair_lo"] = np.minimum(tr["enc_train"].astype(str), tr["enc_eval"].astype(str))
        tr["pair_hi"] = np.maximum(tr["enc_train"].astype(str), tr["enc_eval"].astype(str))
        r = tr.groupby(["group", "pair_lo", "pair_hi"])["retention_raw"].mean().rename(
            "retention").reset_index()
        m = r.merge(a, on=["pair_lo", "pair_hi"], how="inner").dropna()
        for group, g in m.groupby("group"):
            if len(g) < 8:
                continue
            row = report_spearman(g["alignment"].values, g["retention"].values,
                                  prefix="rho_alignment_retention")
            row.update({"group": group, "level": "controlled_pairs",
                        "n_pairs": int(len(g))})
            link_rows.append(row)
    by_obj = df.groupby(["group", "objective_train", "objective_eval"]).agg(
        retention_mean_raw=("retention_raw", "mean"),
        retention_std_raw=("retention_raw", "std"),
        n_rows=("retention_raw", "size")).reset_index()
    by_obj["level"] = "objective_pairing"
    link = pd.concat([pd.DataFrame(link_rows), by_obj], ignore_index=True, sort=False)
    if "p_raw" in link.columns:
        link = add_fdr(link, family_cols=["level"])
    write_csv_atomic(link, link_csv)
    append_status(status, f"main_controlled_transfer wrote {len(df)} transfer rows over "
                          f"{df.enc_train.nunique()} train cells")
    return out_csv
