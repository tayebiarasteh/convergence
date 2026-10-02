"""
controlled/init_alignment.py
Created on September 23, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from config.serde import read_config
from Inference.resume_utils import (MissingInput, append_status, cell_is_done, check_build_params,
                                    fingerprint_file, heartbeat_claim, read_done_meta,
                                    record_sources, run_units_resumable, status_path,
                                    write_build_params, write_csv_atomic, write_npz_atomic)

import warnings
warnings.filterwarnings("ignore")

_DINOV3_BY_BACKBONE = {"vit_s": "facebook/dinov3-vits16-pretrain-lvd1689m",
                       "vit_b": "facebook/dinov3-vitb16-pretrain-lvd1689m"}


def _slug(hf_id: str) -> str:
    return str(hf_id).replace("/", "--")


def init_cells(global_config_path: str) -> Dict[str, Dict]:
    from controlled.matrix import (arm_sources, enumerate_disjoint_runs, enumerate_e5_runs,
                                   enumerate_e11_runs, enumerate_frame_runs)
    cfg = read_config(global_config_path)["Convergence"]
    ckpts = cfg["controlled"]["ckpts_dir"]
    matrix = ([(r[0], r[1]) for r in enumerate_e5_runs(global_config_path)]
              + [(r[0], "taix") for r in enumerate_e11_runs(global_config_path)]
              + [(r[0], "cxr") for r in enumerate_frame_runs(global_config_path)]
              + [(r[0], r[1]) for r in enumerate_disjoint_runs(global_config_path)])
    out = {}
    for run_id, mod in matrix:
        parts = run_id.split("__")
        if parts[3] != "dinov3":
            continue
        ckpt_dir = os.path.join(ckpts, run_id)
        if not cell_is_done(ckpt_dir):
            continue
        hf_id = str(read_done_meta(ckpt_dir).get("backbone_hf_id")
                    or _DINOV3_BY_BACKBONE[parts[2]])
        pool = arm_sources(mod)[1]
        out[run_id] = {"mod": mod, "pool": pool, "hf_id": hf_id, "backbone": parts[2],
                       "init_unit": f"{_slug(hf_id)}|{pool}"}
    return out


def _init_cache_path(emb_dir: str, init_unit: str) -> str:
    slug, pool = init_unit.split("|", 1)
    return os.path.join(emb_dir, f"init__{slug}__{pool}.npz")


def _init_expected(heldout_csv: str, hf_id: str) -> Dict:
    return {"hf_id": hf_id, "heldout": fingerprint_file(heldout_csv),
            "transform": "resize224_centercrop224_imagenet_norm", "token": "cls"}


def _embed_initialization(cfg: Dict, global_config_path: str, hf_id: str, pool: str,
                          out_path: str, partial_dir: str, unit: str) -> str:
    heldout_csv = os.path.join(cfg["controlled"]["results_e5_dir"], f"heldout_{pool}_manifest.csv")
    if not os.path.exists(heldout_csv):
        raise MissingInput(f"the held-out {pool} manifest is absent at {heldout_csv}; main_converge_eval writes it.")
    expected = _init_expected(heldout_csv, hf_id)
    if os.path.exists(out_path) and check_build_params(out_path, expected, owner="init_alignment"):
        return out_path
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from controlled.converge_eval import _eval_transform, _forward_batch
    from controlled.matrix import arm_sources
    from data_loader.modality_embedding_loaders import LOADER_REGISTRY, embedding_collate_fn
    from encoders.panel import export_hf_token, resolve_hf_token
    token = resolve_hf_token(cfg)
    export_hf_token(token)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(hf_id, token=token, trust_remote_code=True).to(device).eval()
    loader_cls = LOADER_REGISTRY[arm_sources(pool)[0]]
    dataset = loader_cls(cfg_path=global_config_path, manifest_csv=heldout_csv)
    nw = int(cfg["controlled"].get("dataloader_workers", 4))
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=nw,
                        collate_fn=embedding_collate_fn,
                        **({"persistent_workers": True, "prefetch_factor": 4} if nw > 0 else {}))
    tfm = _eval_transform()
    embs, ids = [], []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc=f"[init_alignment] {_slug(hf_id)[-22:]} {pool}",
                                       unit="batch")):
            tensors = torch.stack([tfm(im) for im in batch["images"]]).to(device)
            embs.append(_forward_batch(model, tensors).cpu().numpy())
            ids.extend(batch["case_ids"])
            if i % 50 == 0:
                heartbeat_claim(partial_dir, f"embed__{unit}")
    write_npz_atomic(out_path, embeddings=np.concatenate(embs, axis=0).astype(np.float32, copy=False),
                     case_ids=np.array(ids, dtype=object))
    write_build_params(out_path, expected)
    del model
    return out_path


def main_init_alignment(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    out_dir = ctrl["results_e5_dir"]
    emb_dir = os.path.join(out_dir, "embeddings")
    out_csv = os.path.join(out_dir, "cell_to_init_alignment.csv")
    status = status_path(cfg, "init_alignment")
    partial_dir = os.path.join(out_dir, "partials_init")
    if force:
        from Inference.resume_utils import clear_partial_dir
        clear_partial_dir(partial_dir)
    from Inference.regimes import regime_extra
    from controlled.matrix import pairing_group
    from Inference.resume_utils import output_is_current
    cells = init_cells(global_config_path)
    if not cells:
        raise MissingInput("no trained DINOv3-initialized cell has a done flag; train the controlled models first.")
    matched_n = int(cfg["alignment"]["matched_n"])
    n_boot = int(cfg["stats"]["n_boot_neighbor"])
    k_fraction = float(cfg["alignment"]["k_fraction"])
    extra = {**regime_extra("init_alignment"), "matched_n": matched_n, "n_boot_neighbor": n_boot,
             "k_fraction": k_fraction, "cells": sorted(cells)}
    if not force and output_is_current(out_csv, [], owner="main_init_alignment", extra=extra):
        from data_loader.build_utils import read_csv_defensively
        if set(read_csv_defensively(out_csv)["run_id"].astype(str)) == set(cells):
            return out_csv

    init_units = sorted({c["init_unit"] for c in cells.values()})
    hf_of = {c["init_unit"]: c["hf_id"] for c in cells.values()}

    def _embed_unit(unit: str) -> List[Dict]:
        pool = unit.split("|", 1)[1]
        path = _embed_initialization(cfg, global_config_path, hf_of[unit], pool,
                                     _init_cache_path(emb_dir, unit), partial_dir, unit)
        return [{"init_unit": unit, "path": path}]

    emb = run_units_resumable(
        partial_dir=partial_dir, group="embed", units=init_units, compute_unit=_embed_unit,
        build_params={"regime": extra["regime"]}, progress_desc="[init_alignment] initializations",
        use_claims=True, status_file=status)

    def _pair_unit(run_id: str) -> List[Dict]:
        from alignment.metrics import align_by_case_ids, cknna_original, mknn
        from Inference.stats_utils import assert_interval_brackets, subsample_statistic
        c = cells[run_id]
        heldout_csv = os.path.join(out_dir, f"heldout_{c['pool']}_manifest.csv")
        init_path = _init_cache_path(emb_dir, c["init_unit"])
        if not (os.path.exists(init_path) and check_build_params(
                init_path, _init_expected(heldout_csv, c["hf_id"]), owner="init_alignment")):
            raise MissingInput(f"the initialization {c['hf_id']} on {c['pool']} is not embedded "
                               f"yet; another job owns that unit.")
        cell_path = os.path.join(emb_dir, f"{run_id}.npz")
        if not os.path.exists(cell_path):
            raise MissingInput(f"no held-out cache for {run_id}; main_converge_eval embeds it.")
        heartbeat_claim(partial_dir, f"pairs__{run_id}")
        dc, di = np.load(cell_path, allow_pickle=True), np.load(init_path, allow_pickle=True)
        a, b, _ = align_by_case_ids(dc["embeddings"].astype(np.float32), dc["case_ids"].astype(str),
                                    di["embeddings"].astype(np.float32), di["case_ids"].astype(str))
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[finite], b[finite]
        n = min(len(a), 4 * matched_n)
        if n < 2 * matched_n:
            raise MissingInput(f"{run_id}: {len(a)} shared held-out rows allow no fresh draw of "
                               f"the matched N of {matched_n}.")
        idx = np.random.RandomState(42).choice(len(a), size=n, replace=False)
        a, b = a[idx], b[idx]
        k = max(1, int(round(k_fraction * matched_n)))
        m = subsample_statistic([a, b], lambda x, y: mknn(x, y, k=k), n_sub=matched_n, n_boot=n_boot)
        ck = subsample_statistic([a, b], lambda x, y: cknna_original(x, y, k=k), n_sub=matched_n,
                                 n_boot=n_boot)
        assert_interval_brackets(m["point"], m["ci_lower"], m["ci_upper"], label=f"mknn {run_id}")
        assert_interval_brackets(ck["point"], ck["ci_lower"], ck["ci_upper"], label=f"cknna {run_id}")
        parts = run_id.split("__")
        return [{"run_id": run_id, "arm": pairing_group(run_id), "pool": c["pool"],
                 "axis_level": parts[1], "backbone": parts[2], "seed": parts[4].replace("seed", ""),
                 "init_hf_id": c["hf_id"], "n": n, "n_sub": matched_n, "k": k,
                 "resample": "subsample_without_replacement",
                 "mknn": m["point"], "mknn_std": m["std"], "mknn_ci_lower": m["ci_lower"],
                 "mknn_ci_upper": m["ci_upper"], "cknna": ck["point"], "cknna_std": ck["std"],
                 "cknna_ci_lower": ck["ci_lower"], "cknna_ci_upper": ck["ci_upper"]}]

    df = run_units_resumable(
        partial_dir=partial_dir, group="pairs", units=sorted(cells), compute_unit=_pair_unit,
        build_params={k: v for k, v in extra.items() if k != "cells"},
        progress_desc="[init_alignment] cell against its start", use_claims=True, status_file=status)
    n_skipped = int(df.attrs.get("n_skipped", 0)) + int(emb.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(f"{n_skipped} unit(s) of main_init_alignment could not run yet, so the table would be "
                           f"short of cells; the finished units are cached and skip.")
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, [], extra=extra)
    append_status(status, f"main_init_alignment: {len(df)} cells against their initialization")
    return out_csv
