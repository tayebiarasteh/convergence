"""
artifact/relative_reps.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List, Optional

import numpy as np

from alignment.consensus import compute_relative_reps
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def _load_aligned_emb(npz_path: str, manifest_ids: np.ndarray) -> Optional[np.ndarray]:
    """Load embeddings from .npz, reorder rows to match manifest_ids order."""
    if not os.path.exists(npz_path):
        return None
    d   = np.load(npz_path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)
    enc_ids = d["case_ids"].astype(str)
    enc_id_map = {c: i for i, c in enumerate(enc_ids)}
    aligned = np.full((len(manifest_ids), emb.shape[1]), float("nan"), dtype=np.float32)
    for j, cid in enumerate(manifest_ids):
        if cid in enc_id_map:
            aligned[j] = emb[enc_id_map[cid]]
    return aligned


def build_relative_reps(global_config_path: str) -> str:
    """Compute and cache relative representations for all core encoders."""
    cfg        = read_config(global_config_path)["Convergence"]
    aln_cfg    = cfg["alignment"]
    emb_dir    = cfg["embeddings"]["output_dir"]
    cons_dir   = aln_cfg["consensus_dir"]
    anchors_path = os.path.join(cons_dir, "anchors.npy")
    rrep_dir     = os.path.join(cons_dir, "relative_reps")
    os.makedirs(rrep_dir, exist_ok=True)

    if not os.path.exists(anchors_path):
        raise FileNotFoundError(
            f"Anchors not found: {anchors_path}. "
            f"Run build_consensus first."
        )
    anchor_idx  = np.load(anchors_path)
    manifest    = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    manifest_ids = manifest["case_id"].astype(str).values

    from encoders.image_encoders import list_encoder_names
    encoders = list_encoder_names(global_config_path, roles=["core"])

    from tqdm import tqdm
    for enc_name in tqdm(encoders, desc="[relative_reps] encoders", unit="enc"):
        out_path = os.path.join(rrep_dir, f"{enc_name}.npy")
        if os.path.exists(out_path):
            print(f"{enc_name}: exists, skip.")
            continue
        npz_path = os.path.join(emb_dir, enc_name, "cxr_pool.npz")
        emb      = _load_aligned_emb(npz_path, manifest_ids)
        if emb is None:
            print(f"{enc_name}: embedding not found.")
            continue
        # Drop rows with NaN (missing cases)
        valid = ~np.isnan(emb[:, 0])
        emb_valid = emb.copy()
        emb_valid[~valid] = 0.0   # temporary; relative rep will be NaN below
        rel = compute_relative_reps(emb_valid, anchor_idx)
        # Restore NaN for missing rows
        rel[~valid] = float("nan")
        np.save(out_path, rel.astype(np.float32))

    return rrep_dir


def load_relative_rep(enc_name: str, cons_dir: str) -> Optional[np.ndarray]:
    """Load a cached relative representation for one encoder."""
    path = os.path.join(cons_dir, "relative_reps", f"{enc_name}.npy")
    if not os.path.exists(path):
        return None
    return np.load(path).astype(np.float32)