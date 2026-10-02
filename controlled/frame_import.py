"""
controlled/frame_import.py
Created on August 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
import shutil
from typing import Dict, List

from Inference.resume_utils import (MissingInput, append_status, cell_is_done, claim_cell,
                                    check_build_params, clear_cell, done_flag_path,
                                    fingerprint_file, release_claim, status_path,
                                    write_build_params, write_done_flag)
from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")


def _import_one(cfg: Dict, run_id: str, frame_id: str, force: bool) -> str:
    import torch
    from transformers import AutoModel

    ctrl = cfg["controlled"]
    src = os.path.join(cfg["frame"]["ckpts_dir"], f"{frame_id}.pt")
    dst = os.path.join(ctrl["ckpts_dir"], run_id)
    if not os.path.exists(src):
        raise MissingInput(
            f"[frame] {frame_id}.pt is absent at {src}. The sibling project's checkpoints are on "
            f"the cluster; set frame.ckpts_dir for this machine or set frame.enabled false.")
    expected = {"source": fingerprint_file(src), "frame_run_id": frame_id}
    if force:
        clear_cell(dst)
    if cell_is_done(dst, required_files=()) and not force:
        try:
            check_build_params(done_flag_path(dst), expected, owner="frame_import")
            return dst
        except Exception as e:
            clear_cell(dst)
    os.makedirs(dst, exist_ok=True)
    if not claim_cell(dst, run_id, owner=run_id):
        return dst

    try:
        return _materialize(cfg, run_id, frame_id, src, dst, expected)
    except BaseException:
        release_claim(dst, run_id)
        raise


def _materialize(cfg: Dict, run_id: str, frame_id: str, src: str, dst: str, expected: Dict) -> str:
    import torch
    from transformers import AutoModel
    blob = torch.load(src, map_location="cpu", weights_only=False)
    for key in ("state_dict", "backbone_hf_id"):
        if key not in blob:
            raise MissingInput(f"[frame] {src} carries {sorted(blob)} and not '{key}'; this is "
                               f"not a FRAME controlled checkpoint.")
    from encoders.panel import export_hf_token, resolve_hf_token
    token = resolve_hf_token(cfg)
    export_hf_token(token)
    model = AutoModel.from_pretrained(str(blob["backbone_hf_id"]),
                                      token=token, trust_remote_code=True)
    missing, unexpected = model.load_state_dict(blob["state_dict"], strict=False)
    hard = [k for k in missing if not k.startswith(("head.", "proj.", "img_proj.", "txt_proj."))]
    if hard:
        raise MissingInput(
            f"[frame] {frame_id}: {len(hard)} backbone weights are missing from the checkpoint "
            f"(first: {hard[:3]}), so the imported cell would be partly its initialization. "
            f"Check frame.ckpts_dir points at the matching release.")
    tmp = dst + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    for name in os.listdir(tmp):
        os.replace(os.path.join(tmp, name), os.path.join(dst, name))
    shutil.rmtree(tmp, ignore_errors=True)

    meta = dict(blob.get("meta", {}) or {})
    write_done_flag(dst, meta={"run_id": run_id, "provenance": "frame",
                               "frame_run_id": frame_id,
                               "backbone_hf_id": str(blob["backbone_hf_id"]),
                               "frame_meta": meta})
    write_build_params(done_flag_path(dst), expected)
    release_claim(dst, run_id)
    return dst


def main_import_frame_cells(global_config_path: str, force: bool = False) -> str:
    cfg = read_config(global_config_path)["Convergence"]
    fr = cfg.get("frame", {}) or {}
    if not fr.get("enabled", False):
        return ""
    from controlled.matrix import enumerate_frame_runs
    runs = enumerate_frame_runs(global_config_path)
    if not runs:
        raise MissingInput("frame.enabled is true and frame.runs is empty.")
    status = status_path(cfg, "frame_import")
    from tqdm import tqdm
    done = 0
    for run_id, frame_id, _mod, _obj, _bb, _seed in tqdm(runs, desc="[frame] import", unit="cell"):
        _import_one(cfg, run_id, frame_id, force)
        done += 1
    append_status(status, f"imported {done} FRAME cells into {cfg['controlled']['ckpts_dir']}")
    return cfg["controlled"]["ckpts_dir"]
