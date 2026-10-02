"""
alignment/vision_language.py
Created on June 21, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import product
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from alignment.metrics import align_by_case_ids, mknn, cknna_original as cknna
from Inference.resume_utils import (MissingInput, append_status, clear_partial_dir,
                                    run_units_resumable, status_path, write_csv_atomic)
from Inference.stats_utils import (assert_interval_brackets, check_intervals_bracket,
                                   subsample_statistic)
from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")


def main_vision_language_alignment(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    emb_dir = cfg["embeddings"]["output_dir"]
    text_dir = cfg["embeddings"]["text_output_dir"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e2_vla")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "cross_modal_alignment.csv")
    partial_dir = os.path.join(out_dir, "partials")
    status = status_path(cfg, "e2_vision_language")
    if force:
        clear_partial_dir(partial_dir)

    matched_n = int(aln_cfg["matched_n"])
    k_frac = float(aln_cfg["k_fraction"])
    k_values = sorted({max(2, int(round(f * matched_n)))
                       for f in (k_frac / 2, k_frac, k_frac * 2, k_frac * 5)})
    primary_k = max(2, int(round(k_frac * matched_n)))
    seed = int(cfg["seed"])
    n_boot_neighbor = int(cfg["stats"]["n_boot_neighbor"])

    from encoders.panel import list_encoder_names as _img
    from encoders.panel import list_text_encoder_names as _txt
    image_encoders = _img(global_config_path, roles=["core"])
    floor_encoders = _img(global_config_path, roles=["random_init"])
    text_encoders = _txt(global_config_path)
    units = [f"{p}|{i}|{t}" for p in aln_cfg["e2_pools"]
             for i in image_encoders + floor_encoders for t in text_encoders]
    floor_set = set(floor_encoders)

    def compute_unit(unit: str) -> List[dict]:
        pool, img_enc, txt_enc = unit.split("|")
        img_path = os.path.join(emb_dir, img_enc, f"{pool}.npz")
        txt_path = os.path.join(text_dir, txt_enc, f"{pool}.npz")
        for path, who, stage in ((img_path, img_enc, "main_extract_image_embeddings"), (txt_path, txt_enc, "main_extract_text_embeddings")):
            if not os.path.exists(path):
                raise MissingInput(f"no cache for ({who}, {pool}); run {stage} for it before E2.")

        def _load(p):
            d = np.load(p, allow_pickle=True)
            return d["embeddings"].astype(np.float32, copy=False), d["case_ids"].astype(str)

        img_emb, img_ids = _load(img_path)
        txt_emb, txt_ids = _load(txt_path)
        try:
            a, b, _shared = align_by_case_ids(img_emb, img_ids, txt_emb, txt_ids)
        except ValueError as e:
            raise MissingInput(f"({img_enc}, {txt_enc}, {pool}) share no case ids: {e}")
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[finite], b[finite]
        if a.shape[0] < matched_n:
            raise MissingInput(f"({img_enc}, {txt_enc}, {pool}) has {a.shape[0]} finite shared "
                               f"cases, fewer than the matched N of {matched_n}.")
        pool_n = min(a.shape[0], matched_n * 4)
        idx = np.random.RandomState(seed).choice(a.shape[0], size=pool_n, replace=False)
        ai, bi = a[idx], b[idx]

        rows = []
        for k in k_values:
            mk = subsample_statistic([ai, bi], lambda x, y, kk=k: mknn(x, y, kk),
                                     n_sub=matched_n, n_boot=n_boot_neighbor)
            ck = subsample_statistic([ai, bi], lambda x, y, kk=k: cknna(x, y, kk),
                                     n_sub=matched_n, n_boot=n_boot_neighbor)
            for name, r in (("mknn", mk), ("cknna", ck)):
                assert_interval_brackets(r["point"], r["ci_lower"], r["ci_upper"],
                                         f"E2 {name} {unit} k={k}")
            rows.append({
                "pool": pool, "image_encoder": img_enc, "text_encoder": txt_enc,
                "comparison_type": "floor" if img_enc in floor_set else "cross_modal",
                "n": matched_n, "k": k, "is_primary_k": bool(k == primary_k),
                "resample": mk.get("resample"),
                "mknn": mk["point"], "mknn_std": mk["std"],
                "mknn_ci_low": mk["ci_lower"], "mknn_ci_high": mk["ci_upper"],
                "cknna": ck["point"], "cknna_std": ck["std"],
                "cknna_ci_low": ck["ci_lower"], "cknna_ci_high": ck["ci_upper"],
            })
        return rows

    df = run_units_resumable(
        partial_dir=partial_dir, group="e2", units=units, compute_unit=compute_unit,
        build_params={"matched_n": matched_n, "k_fraction": k_frac,
                      "primary_k": primary_k, "resample": "subsample"},
        progress_desc="[E2] image-text pairs", use_claims=True, status_file=status)
    if df.empty:
        raise MissingInput("E2 produced no rows; extract the image and text caches first.")
    for m in ("mknn", "cknna"):
        check_intervals_bracket(df, m, f"{m}_ci_low", f"{m}_ci_high", f"E2 {m}")
    write_csv_atomic(df, out_csv)
    append_status(status, f"E2 wrote {len(df)} rows at matched N = {matched_n}, primary k = {primary_k}")
    return out_csv
