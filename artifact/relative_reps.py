"""
artifact/relative_reps.py
Created on June 21, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List, Optional

import numpy as np

from alignment.consensus import compute_relative_reps
from Inference.resume_utils import write_npy_atomic, MissingInput
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _load_aligned_emb(npz_path: str, manifest_ids: np.ndarray) -> Optional[np.ndarray]:
    if not os.path.exists(npz_path):
        return None
    d   = np.load(npz_path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32, copy=False)
    enc_ids = d["case_ids"].astype(str)
    enc_id_map = {c: i for i, c in enumerate(enc_ids)}
    aligned = np.full((len(manifest_ids), emb.shape[1]), float("nan"), dtype=np.float32)
    for j, cid in enumerate(manifest_ids):
        if cid in enc_id_map:
            aligned[j] = emb[enc_id_map[cid]]
    return aligned


def build_relative_reps(global_config_path: str) -> str:
    cfg        = read_config(global_config_path)["Convergence"]
    aln_cfg    = cfg["alignment"]
    emb_dir    = cfg["embeddings"]["output_dir"]
    cons_dir   = cfg["artifact"]["consensus_dir"]
    anchors_path = os.path.join(cons_dir, "anchor_case_ids.csv")
    rrep_dir     = os.path.join(cons_dir, "relative_reps")
    os.makedirs(rrep_dir, exist_ok=True)

    if not os.path.exists(anchors_path):
        raise MissingInput(
            f"[relative_reps] Anchors not found: {anchors_path}. Run build_consensus first.")
    from alignment.consensus import read_anchor_case_ids
    anchor_ids  = read_anchor_case_ids(anchors_path)
    manifest    = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    manifest_ids = manifest["case_id"].astype(str).values
    id_to_row   = {c: i for i, c in enumerate(manifest_ids)}
    absent = [c for c in anchor_ids if c not in id_to_row]
    if absent:
        raise MissingInput(f"{len(absent)} anchor case ids are not in the pool manifest.")

    primary = int(aln_cfg.get("primary_anchors", 1024))
    counts = sorted({int(n) for n in aln_cfg.get("anchor_sweep", [primary])} | {primary})
    too_big = [n for n in counts if n > len(anchor_ids)]
    if too_big:
        raise MissingInput(f"anchor_sweep asks for {too_big} anchors and the development set holds "
                           f"{len(anchor_ids)}; raise splits.anchor_n and rerun main_build_splits and main_build_consensus.")

    from encoders.panel import list_encoder_names
    encoders = list_encoder_names(global_config_path, roles=["core"])

    from tqdm import tqdm
    from Inference.resume_utils import claim_unit, output_is_current, record_sources, release_claim
    for enc_name in tqdm(encoders, desc="[relative_reps] encoders", unit="enc"):
        emb = None
        for n_a in counts:
            out_path = relative_rep_path(rrep_dir, enc_name, n_a, primary)
            if output_is_current(out_path, [anchors_path], owner="relative_reps"):
                continue
            if not claim_unit(rrep_dir, f"rr__{enc_name}__{n_a}"):
                continue
            try:
                if emb is None:
                    npz_path = os.path.join(emb_dir, enc_name, "cxr_pool.npz")
                    emb = _load_aligned_emb(npz_path, manifest_ids)
                    if emb is None:
                        raise MissingInput(f"no embedding cache for ({enc_name}, cxr_pool); "
                                           f"run main_extract_image_embeddings for that encoder before build_relative_reps.")
                idx = np.array([id_to_row[c] for c in anchor_ids[:n_a]], dtype=np.int64)
                valid = ~np.isnan(emb[:, 0])
                emb_valid = emb.copy()
                emb_valid[~valid] = 0.0
                rel = compute_relative_reps(emb_valid, idx)
                rel[~valid] = float("nan")
                write_npy_atomic(out_path, rel.astype(np.float32, copy=False))
                record_sources(out_path, [anchors_path])
            finally:
                release_claim(rrep_dir, f"rr__{enc_name}__{n_a}")
    return rrep_dir


def relative_rep_path(rrep_dir: str, enc_name: str, n_anchors: int, primary: int) -> str:
    if int(n_anchors) == int(primary):
        return os.path.join(rrep_dir, f"{enc_name}.npy")
    return os.path.join(rrep_dir, f"{enc_name}__a{int(n_anchors)}.npy")


def load_relative_rep(enc_name: str, cons_dir: str, n_anchors: Optional[int] = None,
                      primary: int = 1024) -> Optional[np.ndarray]:
    rrep_dir = os.path.join(cons_dir, "relative_reps")
    n = primary if n_anchors is None else int(n_anchors)
    path = relative_rep_path(rrep_dir, enc_name, n, primary)
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32, copy=False)
