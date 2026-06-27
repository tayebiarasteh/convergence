"""
alignment/vision_language.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from alignment.metrics import align_by_case_ids, mknn, cknna
from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")


def main_vision_language_alignment(global_config_path: str) -> str:
    cfg       = read_config(global_config_path)["Convergence"]
    aln_cfg   = cfg["alignment"]
    emb_dir   = cfg["embeddings"]["output_dir"]
    text_dir  = cfg["embeddings"]["text_output_dir"]
    out_dir   = os.path.join(aln_cfg["results_base_dir"], "results_e2_vla")
    os.makedirs(out_dir, exist_ok=True)
    out_csv   = os.path.join(out_dir, "cross_modal_alignment.csv")

    k_values   = aln_cfg.get("k_values", [5, 10, 20, 50])
    headline_k = int(aln_cfg.get("headline_k", 10))
    n_samples  = aln_cfg.get("n_samples", [1000, 5000, 20000])
    # CKNNA builds (N x N) Gram matrices; cap N (alignment_max_n, default 20000) so a
    # 100k tier in n_samples cannot OOM (100k^2 x4 ~ 40GB per matrix).
    n_cap_e2   = int(aln_cfg.get("alignment_max_n", 20000))
    n_samples  = [n for n in n_samples if n <= n_cap_e2]
    if not n_samples:
        n_samples = [n_cap_e2]
    seed       = int(cfg.get("seed", 42))

    from encoders.image_encoders import list_encoder_names
    from encoders.text_encoders  import list_text_encoder_names
    image_encoders = list_encoder_names(global_config_path, roles=["core"])
    text_encoders  = list_text_encoder_names(global_config_path)

    pool_name = "paired_reports"    # text embedding pool name
    img_pool  = "cxr_pool"         # image embedding pool name (same case_ids)

    partial_csv = os.path.join(out_dir, "cross_modal_alignment.partial.csv")
    done_pairs = set()
    prior_frames = []
    for src in (out_csv, partial_csv):
        if os.path.exists(src):
            try:
                pf = pd.read_csv(src)
                if {"image_encoder", "text_encoder"}.issubset(pf.columns):
                    done_pairs |= set(zip(pf["image_encoder"], pf["text_encoder"]))
                    prior_frames.append(pf)
            except Exception as e:
                print(f"Could not read {os.path.basename(src)} ({e}).")
    all_pairs = list(product(image_encoders, text_encoders))
    todo = [p for p in all_pairs if p not in done_pairs]
    if not todo:
        return out_dir
    # Seed the partial with prior results so the final merge includes them.
    if prior_frames and not os.path.exists(partial_csv):
        pd.concat(prior_frames, ignore_index=True).drop_duplicates(
            subset=["image_encoder", "text_encoder", "n", "k"]
        ).to_csv(partial_csv, index=False)

    for img_enc, txt_enc in tqdm(
        list(product(image_encoders, text_encoders)),
        desc="[E2] cross-modal pairs", unit="pair",
    ):
        if (img_enc, txt_enc) in done_pairs:
            continue
        img_path = os.path.join(emb_dir,  img_enc, f"{img_pool}.npz")
        txt_path = os.path.join(text_dir, txt_enc, f"{pool_name}.npz")
        if not (os.path.exists(img_path) and os.path.exists(txt_path)):
            continue

        def _load(p):
            d = np.load(p, allow_pickle=True)
            return d["embeddings"].astype(np.float32), d["case_ids"].astype(str)

        img_emb, img_ids = _load(img_path)
        txt_emb, txt_ids = _load(txt_path)

        try:
            a, b, shared = align_by_case_ids(img_emb, img_ids, txt_emb, txt_ids)
        except ValueError:
            continue

        # Drop cases where either encoder has a non-finite row (e.g. medgemma_vision
        # ~97% finite, medgemma_27b_text NaN-filled batches). mknn/cknna reject NaN.
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        if not finite.all():
            a, b = a[finite], b[finite]
            shared = [s for s, f in zip(shared, finite) if f] if len(shared) == len(finite) else shared

        pair_rows = []
        n_avail = a.shape[0]
        for n in n_samples:
            if n > n_avail:
                continue
            rng = np.random.RandomState(seed)
            idx = rng.choice(n_avail, size=n, replace=False)
            ai, bi = a[idx], b[idx]

            for k in k_values:
                from Inference.stats_utils import paired_bootstrap_statistic
                n_boot_e2 = int(aln_cfg.get("n_boot_alignment_vla", 100))
                boot_n    = min(n, int(aln_cfg.get("alignment_boot_n", 2000)))
                ai_b, bi_b = ai[:boot_n], bi[:boot_n]
                mk_point = mknn(ai, bi, k=k)
                ck_point = cknna(ai, bi, k=k)
                mk = paired_bootstrap_statistic(
                    [ai_b, bi_b], lambda x, y, kk=k: mknn(x, y, k=kk), n_boot=n_boot_e2)
                ck = paired_bootstrap_statistic(
                    [ai_b, bi_b], lambda x, y, kk=k: cknna(x, y, k=kk), n_boot=n_boot_e2)
                # Use the full-n point estimate; keep the bootstrap's CI/std.
                mk["point"], ck["point"] = mk_point, ck_point
                pair_rows.append({
                    "image_encoder": img_enc,
                    "text_encoder":  txt_enc,
                    "n": n, "k": k,
                    "mknn":           round(mk["point"], 6),
                    "mknn_std":       round(mk["std"], 6),
                    "mknn_ci_low":    round(mk["ci_lower"], 6),
                    "mknn_ci_high":   round(mk["ci_upper"], 6),
                    "cknna":          round(ck["point"], 6),
                    "cknna_std":      round(ck["std"], 6),
                    "cknna_ci_low":   round(ck["ci_lower"], 6),
                    "cknna_ci_high":  round(ck["ci_upper"], 6),
                })
        # Flush this pair's rows immediately so a kill loses at most one pair.
        if pair_rows:
            hdr = not os.path.exists(partial_csv)
            pd.DataFrame(pair_rows).to_csv(partial_csv, mode="a", header=hdr, index=False)
        done_pairs.add((img_enc, txt_enc))

    df = pd.read_csv(partial_csv) if os.path.exists(partial_csv) else pd.DataFrame()
    df.to_csv(out_csv, index=False)
    try:
        os.remove(partial_csv)
    except OSError:
        pass
    return out_dir