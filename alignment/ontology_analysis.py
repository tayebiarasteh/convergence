"""
alignment/ontology_analysis.py
Created on May 28, 2026

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
from Inference.stats_utils import bootstrap_spearman, N_BOOT

import warnings
warnings.filterwarnings("ignore")


def _finding_distances_from_consensus(
    consensus: np.ndarray,       # (N_kept, N_a) consensus configuration
    manifest: pd.DataFrame,
    findings: List[str],
    kept_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    K = len(findings)
    if kept_idx is not None:
        if consensus.shape[0] != len(kept_idx):
            # Defensive: if lengths disagree, fall back to the min to avoid OOB.
            n = min(consensus.shape[0], len(kept_idx))
            kept_idx = kept_idx[:n]
            consensus = consensus[:n]
        man = manifest.iloc[kept_idx].reset_index(drop=True)
    else:
        # No subsample index: assume row-aligned (legacy full-consensus case).
        man = manifest.iloc[:consensus.shape[0]].reset_index(drop=True)

    centroids = np.zeros((K, consensus.shape[1]), dtype=np.float32)
    for i, f in enumerate(findings):
        if f not in man.columns:
            continue
        col     = pd.to_numeric(man[f], errors="coerce").values
        pos_idx = np.where(col == 1.0)[0]          # positions within the kept subset
        pos_idx = pos_idx[pos_idx < consensus.shape[0]]
        if len(pos_idx) > 0:
            centroids[i] = consensus[pos_idx].mean(axis=0)

    # Cosine distance between centroids
    norms = np.linalg.norm(centroids, axis=1, keepdims=True).clip(min=1e-9)
    normed = centroids / norms
    D = 1.0 - (normed @ normed.T)
    return D.astype(np.float32)


def _upper_triangle(M: np.ndarray) -> np.ndarray:
    K = M.shape[0]
    idx = np.triu_indices(K, k=1)
    return M[idx]


def _mantel_permutation(d1: np.ndarray, d2: np.ndarray, n_perm: int = 10000) -> float:
    obs   = float(spearmanr(d1, d2).correlation)
    rng   = np.random.RandomState(0)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(d2)
        if abs(float(spearmanr(d1, shuffled).correlation)) >= abs(obs):
            count += 1
    return (count + 1) / (n_perm + 1)


def main_ontology_analysis(global_config_path: str) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e3_ontology")
    os.makedirs(out_dir, exist_ok=True)

    cons_dir = aln_cfg["consensus_dir"]
    consensus_path = os.path.join(cons_dir, "consensus.npy")
    if not os.path.exists(consensus_path):
        print("[E3] Consensus not found. Run build_consensus first.")
        return out_dir

    consensus = np.load(consensus_path)
    manifest  = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    findings  = [f for f in CANONICAL_CXR_FINDINGS if f in manifest.columns]

    # The consensus is on a case subsample; load the kept-case indices so finding
    # positives are located in consensus-row space, not full-manifest space.
    kept_path = os.path.join(cons_dir, "consensus_kept_case_idx.npy")
    kept_idx  = np.load(kept_path) if os.path.exists(kept_path) else None

    # Consensus inter-finding distance matrix
    D_cons = _finding_distances_from_consensus(consensus, manifest, findings, kept_idx)
    d_cons = _upper_triangle(D_cons)

    results = []

    # Reference 1: ICD-10 distances
    from Inference.report_utils import report_mantel, add_fdr
    icd_csv = cfg["ontology"]["icd10_distance_csv"]
    if os.path.exists(icd_csv):
        icd_df  = read_csv_defensively(icd_csv)
        icd_mat = icd_df.pivot(index="finding_a", columns="finding_b",
                                values="icd10_distance").reindex(
            index=findings, columns=findings
        ).fillna(0).values.astype(float)
        d_icd = _upper_triangle(icd_mat)
        rep = report_mantel(d_cons, d_icd, prefix="mantel")  # estimate+CI+p (natural)
        rep["reference"] = "icd10_hierarchy"
        results.append(rep)

    # Reference 2: comorbidity co-occurrence
    com_csv = cfg["ontology"]["comorbidity_csv"]
    if os.path.exists(com_csv):
        com_df  = read_csv_defensively(com_csv)
        com_mat = com_df.pivot(index="finding_a", columns="finding_b",
                                values="jaccard").reindex(
            index=findings, columns=findings
        ).fillna(0).values.astype(float)
        # Convert similarity (Jaccard) to distance
        d_com = _upper_triangle(1.0 - com_mat)
        rep = report_mantel(d_cons, d_com, prefix="mantel")
        rep["reference"] = "comorbidity_jaccard"
        results.append(rep)

    ref_df = pd.DataFrame(results)
    if not ref_df.empty and "p_raw" in ref_df.columns:
        ref_df = add_fdr(ref_df, family_cols=None)
    ref_df.to_csv(os.path.join(out_dir, "manifold_vs_ontology.csv"), index=False)

    # Triplet analysis (if triplets CSV exists from reader study)
    triplets_csv = cfg["reader_study"]["triplets_csv"]
    if os.path.exists(triplets_csv):
        _triplet_analysis(triplets_csv, D_cons, findings, out_dir,
                          cfg, consensus)

    return out_dir


def _triplet_analysis(
    triplets_csv: str,
    D_cons: np.ndarray,
    findings: List[str],
    out_dir: str,
    cfg: dict,
    consensus: np.ndarray,
):
    """Predict radiologist triplet judgments from the consensus and each encoder."""
    trips = read_csv_defensively(triplets_csv)
    # Expected columns: case_a, case_b, case_c, choice (b or c is closer to a)
    required = {"case_a", "case_b", "case_c", "choice"}
    if not required.issubset(trips.columns):
        print(f"[E3/triplets] Missing columns: {required - set(trips.columns)}")
        return

    # Load pool manifest case_id -> row index map for per-encoder (full) lookup.
    manifest = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    id_to_idx = {str(cid): i for i, cid in enumerate(manifest["case_id"])}

    kept_path = os.path.join(cons_dir, "consensus_kept_case_idx.npy")
    if os.path.exists(kept_path):
        kept_idx = np.load(kept_path)
        cons_ids = manifest["case_id"].astype(str).values[kept_idx]
        cons_id_to_idx = {str(cid): i for i, cid in enumerate(cons_ids)}
    else:
        cons_id_to_idx = id_to_idx

    def _predict(trips_df, emb: np.ndarray, idmap: dict) -> np.ndarray:
        hits = []
        n_emb = emb.shape[0]
        for _, row in trips_df.iterrows():
            ia = idmap.get(str(row["case_a"]))
            ib = idmap.get(str(row["case_b"]))
            ic = idmap.get(str(row["case_c"]))
            if any(x is None for x in [ia, ib, ic]):
                continue
            if ia >= n_emb or ib >= n_emb or ic >= n_emb:
                continue
            ea, eb, ec = emb[ia], emb[ib], emb[ic]
            d_ab = float(1.0 - np.dot(ea, eb))
            d_ac = float(1.0 - np.dot(ea, ec))
            pred = "b" if d_ab < d_ac else "c"
            hits.append(int(pred == str(row["choice"]).strip().lower()))
        return np.array(hits, dtype=float)

    from Inference.report_utils import report_metric
    rows_out = []
    # Consensus (uses the kept-case id map).
    hits_cons = _predict(trips, consensus, cons_id_to_idx)
    rep = report_metric(hits_cons, prefix="triplet_acc", is_percent=True)
    rep["encoder"] = "CONSENSUS"
    rows_out.append(rep)

    # Per encoder
    emb_dir = cfg["embeddings"]["output_dir"]
    from encoders.image_encoders import list_encoder_names
    for enc_name in tqdm(list_encoder_names(cfg, roles=["core"]), desc="[E3] per-encoder", unit="enc"):
        npz_path = os.path.join(emb_dir, enc_name, "cxr_pool.npz")
        if not os.path.exists(npz_path):
            continue
        d = np.load(npz_path, allow_pickle=True)
        emb_raw = d["embeddings"].astype(np.float32)
        # Re-align to manifest order
        enc_ids = d["case_ids"].astype(str)
        enc_id_map = {c: i for i, c in enumerate(enc_ids)}
        aligned = np.zeros((len(manifest), emb_raw.shape[1]), dtype=np.float32)
        for j, cid in enumerate(manifest["case_id"].astype(str)):
            if cid in enc_id_map:
                aligned[j] = emb_raw[enc_id_map[cid]]
        hits = _predict(trips, aligned, id_to_idx)
        rep = report_metric(hits, prefix="triplet_acc", is_percent=True)
        rep["encoder"] = enc_name
        rows_out.append(rep)

    pd.DataFrame(rows_out).to_csv(
        os.path.join(out_dir, "triplet_accuracy.csv"), index=False
    )
    cons_mean = rows_out[0].get("triplet_acc_mean", float("nan"))
