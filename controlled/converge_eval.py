"""
controlled/converge_eval.py
Created on June 1, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignment.metrics import mknn, cknna
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.modality_embedding_loaders import embedding_collate_fn

import warnings
warnings.filterwarnings("ignore")


def _extract_controlled_embeddings(
    run_id: str,
    ckpt_dir: str,
    pool_manifest: str,
    modality: str,
    cfg: dict,
    device: torch.device,
    output_dir: str,
) -> Optional[str]:
    """Extract embeddings for one controlled encoder on one pool."""
    out_path = os.path.join(output_dir, f"{run_id}.npz")
    if os.path.exists(out_path):
        return out_path

    model_pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(model_pt):
        # Try HF-style checkpoint
        from transformers import AutoModel
        try:
            model = AutoModel.from_pretrained(ckpt_dir, trust_remote_code=True)
        except Exception:
            return None
    else:
        # Infer arch from run_id
        backbone = "vit_small_patch16_224" if "vit_s" in run_id else "vit_base_patch16_224"
        import timm
        model = timm.create_model(backbone, pretrained=False, num_classes=0)
        model.load_state_dict(torch.load(model_pt, map_location="cpu"))

    model = model.to(device=device).eval()

    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    from data_loader.modality_embedding_loaders import LOADER_REGISTRY
    loader_cls = LOADER_REGISTRY.get(modality)
    cfg_path = cfg.get("global_config_path", "")
    dataset = loader_cls(cfg_path=cfg_path, manifest_csv=pool_manifest)
    loader  = DataLoader(dataset, batch_size=64, shuffle=False,
                         num_workers=4, collate_fn=embedding_collate_fn)

    all_emb = []
    all_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[converge_eval] {run_id[:30]}", unit="batch"):
            imgs = batch["images"]
            tensors = torch.stack([tfm(im) for im in imgs]).to(device)
            if hasattr(model, "forward_features"):
                feats = model.forward_features(tensors)
                if feats.dim() == 3:
                    emb = feats[:, 0, :]
                else:
                    emb = feats
            else:
                out = model(tensors)
                emb = out.last_hidden_state[:, 0, :] if hasattr(out, "last_hidden_state") else out
            emb = torch.nn.functional.normalize(emb.float(), p=2, dim=-1)
            all_emb.append(emb.cpu().numpy())
            all_ids.extend(batch["case_ids"])

    embeddings = np.concatenate(all_emb, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path,
                        embeddings=embeddings.astype(np.float32),
                        case_ids=np.array(all_ids, dtype=object))
    return out_path


def main_converge_eval(global_config_path: str) -> str:
    cfg  = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    out_dir = ctrl["results_e5_dir"]
    emb_dir = os.path.join(out_dir, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "e5_alignment_table.csv")
    if os.path.exists(out_csv):
        print(f"[E5] Output exists; skipping. ({out_csv})")
        return out_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    objectives = ctrl.get("objectives", ["ssl", "supervised", "image_text"])
    backbones  = ctrl.get("backbones",  ["vit_s", "vit_b"])
    inits      = ctrl.get("inits",      ["dinov3_s", "imagenet"])
    modalities = ctrl.get("modalities", ["cxr", "histo"])
    n_seeds    = int(ctrl.get("n_seeds", 2))

    pool_map = {
        "cxr":   cfg["cxr"]["pool_manifest_csv"],
        "histo": cfg["histo"]["pool_manifest_csv"],
    }

    # Collect all run_ids and their embeddings
    quilt_enabled = bool(cfg.get("quilt", {}).get("enabled", False))
    # Enumerate the cells to extract first, so we can show overall progress.
    cells = []
    for mod in modalities:
        for obj in objectives:
            if mod == "histo" and obj == "image_text" and not quilt_enabled:
                continue
            for bb in backbones:
                for init in inits:
                    for seed in range(n_seeds):
                        run_id   = f"{mod}__{obj}__{bb}__{init}__seed{seed}"
                        ckpt_dir = os.path.join(ctrl["ckpts_dir"], run_id)
                        if os.path.exists(os.path.join(ckpt_dir, "training_done.flag")):
                            cells.append((run_id, ckpt_dir, mod))

    emb_cache: Dict[str, str] = {}
    for ci, (run_id, ckpt_dir, mod) in enumerate(cells, 1):
        path = _extract_controlled_embeddings(
            run_id, ckpt_dir, pool_map[mod], mod, cfg, device, emb_dir
        )
        if path:
            emb_cache[run_id] = path

    # Compute pairwise alignment (same-modality pairs only).
    run_ids = list(emb_cache.keys())
    all_pairs = [(ra, rb) for ra, rb in combinations(run_ids, 2)
                 if ra.split("__")[0] == rb.split("__")[0]]

    partial_csv = os.path.join(out_dir, "e5_alignment_table.partial.csv")
    done_pairs = set()
    prior_frames = []
    if os.path.exists(partial_csv):
        try:
            pf = pd.read_csv(partial_csv)
            done_pairs = set(zip(pf["run_id_a"], pf["run_id_b"]))
            prior_frames.append(pf)
        except Exception as e:
            print(f"could not read partial ({e}); recomputing.")
    if done_pairs:
        print(f"{len(done_pairs)} alignment pairs already done; computing only missing.")

    for rid_a, rid_b in tqdm(all_pairs, desc="align pairs", unit="pair"):
        if (rid_a, rid_b) in done_pairs:
            continue
        parts_a = rid_a.split("__")
        parts_b = rid_b.split("__")

        da = np.load(emb_cache[rid_a], allow_pickle=True)
        db = np.load(emb_cache[rid_b], allow_pickle=True)
        ea = da["embeddings"].astype(np.float32)
        eb = db["embeddings"].astype(np.float32)
        ids_a = da["case_ids"].astype(str)
        ids_b = db["case_ids"].astype(str)

        from alignment.metrics import align_by_case_ids
        try:
            a, b, _ = align_by_case_ids(ea, ids_a, eb, ids_b)
        except ValueError:
            continue
        # Drop non-finite rows before the metrics.
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[finite], b[finite]
        if len(a) < 100:
            continue
        n = min(5000, len(a))
        rng = np.random.RandomState(42)
        idx = rng.choice(len(a), size=n, replace=False)
        a_s, b_s = a[idx], b[idx]

        from Inference.stats_utils import paired_bootstrap_statistic
        n_boot_e5 = int(ctrl.get("n_boot_alignment", 200))
        boot_n    = min(len(a_s), int(ctrl.get("alignment_boot_n", 3000)))
        ab, bb_   = a_s[:boot_n], b_s[:boot_n]
        mknn_bs   = paired_bootstrap_statistic(
            [ab, bb_], lambda x, y: mknn(x, y, k=10), n_boot=n_boot_e5)
        cknna_bs  = paired_bootstrap_statistic(
            [ab, bb_], lambda x, y: cknna(x, y, k=10), n_boot=n_boot_e5)
        # 'point' from paired_bootstrap_statistic is the statistic on the full passed
        # sample (ab,bb_), so point and CI are now from the same data and consistent.

        row = {
            "run_id_a":    rid_a, "run_id_b": rid_b,
            "modality":    parts_a[0],
            "objective_a": parts_a[1], "objective_b": parts_b[1],
            "backbone_a":  parts_a[2], "backbone_b": parts_b[2],
            "init_a":      parts_a[3], "init_b": parts_b[3],
            "same_objective": parts_a[1] == parts_b[1],
            "same_seed":   rid_a.split("seed")[1] == rid_b.split("seed")[1],
            "n":           n,
            "mknn":            round(mknn_bs["point"], 6),
            "mknn_std":        round(mknn_bs["std"], 6),
            "mknn_ci_lower":   round(mknn_bs["ci_lower"], 6),
            "mknn_ci_upper":   round(mknn_bs["ci_upper"], 6),
            "cknna":           round(cknna_bs["point"], 6),
            "cknna_std":       round(cknna_bs["std"], 6),
            "cknna_ci_lower":  round(cknna_bs["ci_lower"], 6),
            "cknna_ci_upper":  round(cknna_bs["ci_upper"], 6),
        }
        # Flush this pair immediately so a kill does not lose completed work.
        hdr = not os.path.exists(partial_csv)
        pd.DataFrame([row]).to_csv(partial_csv, mode="a", header=hdr, index=False)

    df = pd.read_csv(partial_csv) if os.path.exists(partial_csv) else pd.DataFrame()
    df.to_csv(out_csv, index=False)
    if os.path.exists(partial_csv):
        os.remove(partial_csv)

    # Summary
    if not df.empty:
        summary = df.groupby(["modality","objective_a","objective_b"])["mknn"].agg(
            ["mean","std","count"]
        ).reset_index()
        summary.to_csv(os.path.join(out_dir, "e5_summary.csv"), index=False)
        print(summary.to_string(index=False))

    return out_dir