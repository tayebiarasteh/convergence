"""
alignment/ontology_analysis.py
Created on June 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
from Inference.resume_utils import append_status, status_path, write_csv_atomic
from Inference.stats_utils import bootstrap_spearman, exceedance_p, N_BOOT

import warnings
from Inference.resume_utils import MissingInput
warnings.filterwarnings("ignore")


def _finding_distances_from_consensus(
    consensus: np.ndarray,
    manifest: pd.DataFrame,
    findings: List[str],
    kept_idx: Optional[np.ndarray] = None,
    label_override: Optional[np.ndarray] = None,
) -> np.ndarray:
    K = len(findings)
    if kept_idx is not None:
        if consensus.shape[0] != len(kept_idx):
            n = min(consensus.shape[0], len(kept_idx))
            kept_idx = kept_idx[:n]
            consensus = consensus[:n]
        man = manifest.iloc[kept_idx].reset_index(drop=True)
    else:
        man = manifest.iloc[:consensus.shape[0]].reset_index(drop=True)

    centroids = np.zeros((K, consensus.shape[1]), dtype=np.float32)
    for i, f in enumerate(findings):
        if label_override is not None:
            col = np.asarray(label_override)[:, i]
        elif f in man.columns:
            col = pd.to_numeric(man[f], errors="coerce").values
        else:
            continue
        pos_idx = np.where(col == 1.0)[0]
        pos_idx = pos_idx[pos_idx < consensus.shape[0]]
        if len(pos_idx) > 0:
            centroids[i] = consensus[pos_idx].mean(axis=0)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-9)
    normed = centroids / norms
    D = 1.0 - (normed @ normed.T)
    return D.astype(np.float32, copy=False)


def _upper_triangle(M: np.ndarray) -> np.ndarray:
    K = M.shape[0]
    idx = np.triu_indices(K, k=1)
    return M[idx]


def structural_reference(labels, distance_fn, n_perm: int = 1000, seed: int = 0,
                         on_step=None):
    rng = np.random.RandomState(seed)
    lab = np.asarray(labels, dtype=float)
    draws = np.empty(int(n_perm), dtype=float)
    for b in tqdm(range(int(n_perm)), desc="[E3] structural reference", unit="perm", leave=False):
        draws[b] = distance_fn(lab[rng.permutation(lab.shape[0])])
        if on_step is not None:
            on_step()
    good = draws[np.isfinite(draws)]
    return {"reference_mean": float(np.nanmean(draws)),
            "reference_ci_lower": float(np.nanpercentile(draws, 2.5)),
            "reference_ci_upper": float(np.nanpercentile(draws, 97.5)),
            "n_perm": int(n_perm),
            "preserves_prevalence": True,
            "preserves_cooccurrence": True,
            "_draws": good}


def _chest_structure_rows(cfg: dict, consensus: np.ndarray, manifest: pd.DataFrame,
                          findings: List[str], kept_idx: Optional[np.ndarray],
                          on_step=None) -> List[Dict]:
    from Inference.report_utils import report_mantel
    D_cons = _finding_distances_from_consensus(consensus, manifest, findings, kept_idx)
    d_cons = _upper_triangle(D_cons)
    results, refs = [], {}
    icd_csv = cfg["ontology"]["icd10_distance_csv"]
    if os.path.exists(icd_csv):
        icd_mat = read_csv_defensively(icd_csv).pivot(
            index="finding_a", columns="finding_b", values="icd10_distance").reindex(
            index=findings, columns=findings).fillna(0).values.astype(float, copy=False)
        rep = report_mantel(d_cons, _upper_triangle(icd_mat), prefix="mantel",
                            n_objects=len(findings))
        rep["reference"] = "icd10_hierarchy"
        refs["icd10_hierarchy"] = _upper_triangle(icd_mat)
        results.append(rep)
    com_csv = cfg["ontology"]["comorbidity_csv"]
    if os.path.exists(com_csv):
        com_mat = read_csv_defensively(com_csv).pivot(
            index="finding_a", columns="finding_b", values="jaccard").reindex(
            index=findings, columns=findings).fillna(0).values.astype(float, copy=False)
        rep = report_mantel(d_cons, _upper_triangle(1.0 - com_mat), prefix="mantel",
                            n_objects=len(findings))
        rep["reference"] = "comorbidity_jaccard"
        refs["comorbidity_jaccard"] = _upper_triangle(1.0 - com_mat)
        results.append(rep)
    if not results:
        raise MissingInput("neither the ICD-10 distances nor the comorbidity table exists; "
                           "run main_build_ontology_geometry before E3.")
    label_mat = manifest.iloc[kept_idx][findings].apply(
        pd.to_numeric, errors="coerce").fillna(0.0).values
    for rep in results:
        d_ref = refs[rep["reference"]]

        def _perm_mantel(perm_labels, _d_ref=d_ref):
            D = _finding_distances_from_consensus(
                consensus, manifest, findings, kept_idx, label_override=perm_labels)
            return float(spearmanr(_upper_triangle(D), _d_ref).statistic)

        ref = structural_reference(label_mat, _perm_mantel,
                                   n_perm=int(cfg.get("stats", {}).get("n_perm_structural", 200)),
                                   seed=int(cfg.get("seed", 42)), on_step=on_step)
        draws = ref.pop("_draws")
        observed = float(rep.get("mantel_estimate_raw", float("nan")))
        rep.update(ref)
        rep["excess_over_structural_reference"] = observed - ref["reference_mean"]
        rep["p_structural"] = exceedance_p(observed, draws)
        rep["pool"] = "cxr_pool"
        rep["case_overlap_possible"] = True
    return results


def main_ontology_analysis(global_config_path: str, force: bool = False) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "ontology")
    aln_cfg = cfg["alignment"]
    pools = aln_cfg["e3_pools"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e3_ontology")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "manifold_vs_ontology.csv")
    trip_out = os.path.join(out_dir, "triplet_accuracy.csv")
    from Inference.regimes import regime_extra
    from alignment.consensus import consensus_state
    from Inference.resume_utils import (digest_file, fingerprint_file, heartbeat_claim,
                                        output_is_current, record_sources, run_units_resumable)
    state, reason = consensus_state(cfg, global_config_path)
    if state == "pending":
        raise MissingInput(f"[E3] {reason}; E3 reads the consensus and runs once main_build_consensus has finished.")
    consensus_ok = state == "ready"

    cons_dir = aln_cfg["consensus_dir"]
    consensus_path = os.path.join(cons_dir, "consensus.npy")
    kept_path = os.path.join(cons_dir, "consensus_kept_case_idx.npy")
    triplets_csv = cfg["reader_study"]["triplets_csv"]
    derm_csv = cfg["derm"]["pool_manifest_csv"]
    n_perm_structural = int(cfg.get("stats", {}).get("n_perm_structural", 200))
    sources = [p for p in (consensus_path, kept_path, cfg["cxr"]["pool_manifest_csv"],
                           cfg["ontology"]["icd10_distance_csv"], cfg["ontology"]["comorbidity_csv"],
                           derm_csv) if os.path.exists(p)]
    extra = {**regime_extra("e3"), "consensus_state": state, "triplets": digest_file(triplets_csv),
             "n_perm_structural": n_perm_structural}
    if (not force and output_is_current(out_csv, sources, "E3", extra={**extra, "complete": True})
            and (not os.path.exists(triplets_csv)
                 or output_is_current(trip_out, sources, "E3", extra={**extra, "complete": True}))):
        return out_dir

    manifest  = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    findings  = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]
    consensus, kept_idx = None, None
    if consensus_ok:
        consensus = np.load(consensus_path)
        kept_idx = np.load(kept_path) if os.path.exists(kept_path) else None

    units = (["chest"] if consensus_ok else []) + [p for p in pools if p != "cxr_pool"]
    if os.path.exists(triplets_csv):
        units.append("triplets")
    partial_dir = os.path.join(out_dir, "partials")

    def compute_unit(unit: str) -> List[Dict]:
        def _beat():
            heartbeat_claim(partial_dir, f"e3__{unit}")
        if unit == "chest":
            out = _chest_structure_rows(cfg, consensus, manifest, findings, kept_idx,
                                        on_step=_beat)
        elif unit == "triplets":
            out = _triplet_analysis(triplets_csv, findings, cfg, consensus, kept_idx,
                                    global_config_path, on_step=_beat)
        else:
            out = _single_label_replication(cfg, global_config_path, unit, on_step=_beat)
        for r in out:
            r["e3_unit"] = unit
        return out

    df_all = run_units_resumable(
        partial_dir=partial_dir, group="e3", units=units, compute_unit=compute_unit,
        progress_desc="[E3] units", use_claims=True, status_file=status,
        build_params={**extra, "consensus": fingerprint_file(consensus_path) if consensus_ok else None,
                      "manifest": fingerprint_file(cfg["cxr"]["pool_manifest_csv"]),
                      "derm": fingerprint_file(derm_csv), "seed": int(cfg.get("seed", 42))})
    n_skipped = int(df_all.attrs.get("n_skipped", 0))
    from Inference.report_utils import add_fdr
    from Inference.resume_utils import load_group_rows, unit_done
    complete = all(unit_done(partial_dir, "e3", u) for u in units)
    rows = [r for r in load_group_rows(partial_dir, "e3") if r.get("e3_unit") != "triplets"]
    trip_rows = [r for r in load_group_rows(partial_dir, "e3") if r.get("e3_unit") == "triplets"]
    ref_df = pd.DataFrame(rows)
    if not ref_df.empty and "p_raw" in ref_df.columns:
        ref_df = add_fdr(ref_df, family_cols=["pool"])
    complete_extra = {**extra, "complete": complete}
    if not ref_df.empty:
        write_csv_atomic(ref_df, out_csv)
        record_sources(out_csv, sources, extra=complete_extra)
    if trip_rows:
        write_csv_atomic(pd.DataFrame(trip_rows), trip_out)
        record_sources(trip_out, sources, extra=complete_extra)
    append_status(status, f"E3: {len(ref_df)} structure rows, {len(trip_rows)} triplet rows, "
                          f"{n_skipped} unit(s) not run")
    return out_dir


def _triplet_analysis(
    triplets_csv: str,
    findings: List[str],
    cfg: dict,
    consensus: Optional[np.ndarray],
    kept_idx: Optional[np.ndarray],
    global_config_path: str,
    on_step=None,
) -> List[Dict]:
    trips = read_csv_defensively(triplets_csv)
    required = {"case_a", "case_b", "case_c", "choice"}
    if not required.issubset(trips.columns):
        raise MissingInput(f"[E3/triplets] {triplets_csv} lacks {sorted(required - set(trips.columns))}; "
                           f"analyze_all in reader_study/analyze_reader_studies.py builds it from the readers' answers.")

    manifest = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    id_to_idx = {str(cid): i for i, cid in enumerate(manifest["case_id"])}

    if kept_idx is not None:
        cons_ids = manifest["case_id"].astype(str).values[kept_idx]
        cons_id_to_idx = {str(cid): i for i, cid in enumerate(cons_ids)}
    else:
        cons_id_to_idx = id_to_idx

    trip_case_ids = set()
    for c in ("case_a", "case_b", "case_c"):
        trip_case_ids |= set(trips[c].astype(str))
    tsub = manifest[manifest["case_id"].astype(str).isin(trip_case_ids)]
    pos_findings = {}
    for _, r in tsub.iterrows():
        pos_findings[str(r["case_id"])] = {
            f for f in findings
            if pd.to_numeric(pd.Series([r.get(f)]), errors="coerce").iloc[0] == 1.0}

    def _stratum(row) -> str:
        fa = pos_findings.get(str(row["case_a"]))
        fb = pos_findings.get(str(row["case_b"]))
        fc = pos_findings.get(str(row["case_c"]))
        if fa is None or fb is None or fc is None:
            return "labels_unavailable"
        sb, sc = bool(fa & fb), bool(fa & fc)
        return "one_candidate_shares_a_finding" if sb != sc else "ambiguous_by_labels"

    trips = trips.copy()
    trips["label_stratum"] = [_stratum(r) for _, r in trips.iterrows()]

    def _predict(trips_df, emb: np.ndarray, idmap: dict, present=None):
        hits, strata = [], []
        n_tie = 0
        n_absent = 0
        n_emb = emb.shape[0]
        for _, row in trips_df.iterrows():
            ia = idmap.get(str(row["case_a"]))
            ib = idmap.get(str(row["case_b"]))
            ic = idmap.get(str(row["case_c"]))
            if any(x is None for x in [ia, ib, ic]):
                n_absent += 1
                continue
            if ia >= n_emb or ib >= n_emb or ic >= n_emb:
                n_absent += 1
                continue
            if present is not None and not (present[ia] and present[ib] and present[ic]):
                n_absent += 1
                continue
            ea, eb, ec = emb[ia], emb[ib], emb[ic]
            d_ab = float(1.0 - np.dot(ea, eb))
            d_ac = float(1.0 - np.dot(ea, ec))
            if abs(d_ab - d_ac) < 1e-9:
                n_tie += 1
                continue
            pred = "b" if d_ab < d_ac else "c"
            hits.append(int(pred == str(row["choice"]).strip().lower()))
            strata.append(str(row.get("label_stratum", "labels_unavailable")))
        return np.array(hits, dtype=float), strata, n_tie, n_absent

    from Inference.report_utils import report_metric
    rows_out = []

    def _emit(name: str, hits, strata, n_tie: int, n_absent: int, kind: str = "frozen_panel"):
        groups = [("all", hits)]
        s = np.asarray(strata, dtype=object)
        for st in sorted(set(strata)):
            groups.append((st, hits[s == st]))
        for st, h in groups:
            if len(h) == 0:
                continue
            rep = report_metric(h, prefix="triplet_acc", is_percent=True)
            rep.update({"encoder": name, "encoder_kind": kind, "conditioning": st,
                        "n_scored": int(len(h)), "n_tied": int(n_tie),
                        "n_absent_case": int(n_absent), "n_triplets": int(len(trips))})
            rows_out.append(rep)

    if consensus is not None:
        hits_cons, strata_cons, tie_c, abs_c = _predict(trips, consensus, cons_id_to_idx)
        _emit("CONSENSUS", hits_cons, strata_cons, tie_c, abs_c, kind="consensus")

    def _aligned_with_presence(npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        emb_raw = d["embeddings"].astype(np.float32, copy=False)
        enc_id_map = {c: i for i, c in enumerate(d["case_ids"].astype(str))}
        aligned = np.zeros((len(manifest), emb_raw.shape[1]), dtype=np.float32)
        present = np.zeros(len(manifest), dtype=bool)
        for j, cid in enumerate(manifest["case_id"].astype(str)):
            i = enc_id_map.get(cid)
            if i is not None and np.isfinite(emb_raw[i]).all():
                aligned[j] = emb_raw[i]
                present[j] = True
        return aligned, present

    emb_dir = cfg["embeddings"]["output_dir"]
    from encoders.panel import list_encoder_names
    for enc_name in tqdm(list_encoder_names(global_config_path, roles=["core"]), desc="[E3] per-encoder", unit="enc"):
        if on_step is not None:
            on_step()
        npz_path = os.path.join(emb_dir, enc_name, "cxr_pool.npz")
        if not os.path.exists(npz_path):
            raise MissingInput(f"no chest cache for {enc_name}; run main_extract_image_embeddings for it before E3.")
        aligned, present = _aligned_with_presence(npz_path)
        hits, strata, n_tie, n_absent = _predict(trips, aligned, id_to_idx, present=present)
        _emit(enc_name, hits, strata, n_tie, n_absent)

    from controlled.matrix import enumerate_e5_runs
    ctrl_emb_dir = os.path.join(cfg["controlled"]["results_e5_dir"], "embeddings")
    for run_id, mod, obj, _bb, _init, _seed in enumerate_e5_runs(global_config_path):
        if mod != "cxr" or obj != "supervised":
            continue
        if on_step is not None:
            on_step()
        npz_path = os.path.join(ctrl_emb_dir, f"{run_id}.npz")
        if not os.path.exists(npz_path):
            continue
        aligned, present = _aligned_with_presence(npz_path)
        hits, strata, n_tie, n_absent = _predict(trips, aligned, id_to_idx, present=present)
        _emit(run_id, hits, strata, n_tie, n_absent, kind="controlled_supervised")
    return rows_out

def _taxonomy_distance(classes: List[str], mapping: Dict[str, str]) -> np.ndarray:
    n = len(classes)
    d = np.full((n, n), 2.0, dtype=float)
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            if i == j:
                d[i, j] = 0.0
            elif mapping.get(a) == mapping.get(b):
                d[i, j] = 1.0
    return d


def derm_class_columns(cfg: dict) -> Dict[str, str]:
    cmap: Dict[str, str] = {}
    for src in (cfg.get("derm", {}).get("sources", {}) or {}).values():
        for code, name in (src.get("class_map", {}) or {}).items():
            cmap[str(code).lower()] = str(name)
    return {str(c): cmap.get(str(c).lower(), str(c))
            for c in cfg["ontology"].get("derm_classes", [])}


def _single_label_replication(cfg: dict, global_config_path: str, pool: str,
                              on_step=None) -> List[Dict]:
    from encoders.cache_utils import load_embedding_cache
    from encoders.panel import list_encoder_names
    from Inference.report_utils import report_mantel

    ont = cfg["ontology"]
    col_of = derm_class_columns(cfg)
    if pool != "derm_pool" or not col_of:
        raise MissingInput(f"no single-label replication is declared for {pool}.")
    manifest_csv = cfg["derm"]["pool_manifest_csv"]
    if not os.path.exists(manifest_csv):
        raise MissingInput(f"the derm pool manifest is absent at {manifest_csv}; run main_build_derm_pool before E3.")
    man = read_csv_defensively(manifest_csv)
    present = [c for c, col in col_of.items() if col in man.columns]
    if len(present) < 4:
        raise MissingInput(f"the derm pool has only {len(present)} of the {len(col_of)} "
                           f"declared classes under the column names of derm.class_map "
                           f"({sorted(col_of.values())}), so the replication has too few points.")
    refs = {"derm_lineage": _taxonomy_distance(present, dict(ont["derm_lineage"])),
            "derm_malignancy": _taxonomy_distance(present, dict(ont["derm_malignancy"]))}

    rows: List[Dict] = []
    for enc in tqdm(list_encoder_names(global_config_path, roles=["core"]),
                    desc="[E3] single-label replication", unit="enc"):
        if on_step is not None:
            on_step()
        emb, ids = load_embedding_cache(cfg, enc, pool)
        pos = {str(c): i for i, c in enumerate(ids)}
        rowsel = man["case_id"].astype(str).isin(pos)
        sub = man[rowsel]
        idx = np.array([pos[c] for c in sub["case_id"].astype(str)])
        cents, kept = [], []
        for c in present:
            m = pd.to_numeric(sub[col_of[c]], errors="coerce").values == 1.0
            if m.sum() < 20:
                continue
            v = emb[idx[m]]
            v = v[np.isfinite(v).all(axis=1)]
            if len(v) < 20:
                continue
            cents.append(v.mean(axis=0))
            kept.append(c)
        if len(kept) < 4:
            continue
        C = np.vstack(cents)
        C = C / np.linalg.norm(C, axis=1, keepdims=True).clip(min=1e-9)
        d_enc = _upper_triangle(1.0 - C @ C.T)
        keep_i = [present.index(c) for c in kept]
        for name, full in refs.items():
            rep = report_mantel(d_enc, _upper_triangle(full[np.ix_(keep_i, keep_i)]),
                                prefix="mantel", n_objects=len(kept))
            rep.update({"reference": name, "pool": pool, "encoder": enc,
                        "n_classes": len(kept), "classes": ";".join(kept),
                        "case_overlap_possible": False,
                        "preserves_prevalence": True, "preserves_cooccurrence": True})
            rows.append(rep)
    if not rows:
        raise MissingInput(f"no encoder reached 4 classes with 20 cases each on {pool}.")
    return rows
