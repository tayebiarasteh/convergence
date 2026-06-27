"""
controlled/train_encoder.py
Created on June 1, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import math
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")



class _CXRDataset(Dataset):
    def __init__(self, manifest_csv: str, image_root: str,
                 objective: str, label_cols: list, transform):
        from data_loader.build_utils import read_csv_defensively
        self.df         = read_csv_defensively(manifest_csv)
        self.image_root = image_root
        self.objective  = objective
        self.label_cols = label_cols
        self.transform  = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        from data_loader.cxr_harmonization import resolve_cxr_image_path
        path  = resolve_cxr_image_path("mimic", self.image_root,
                                        str(row["image_key"]))
        img   = Image.open(path).convert("RGB")
        img   = self.transform(img)
        if self.objective == "supervised":
            labels = torch.tensor(
                [1.0 if row.get(f, float("nan")) == 1.0 else 0.0
                 for f in self.label_cols], dtype=torch.float32
            )
            return img, labels
        if self.objective == "image_text":
            return img, str(row.get("report_rel_path", ""))
        return img, idx   # ssl: image + index (for pair-free MAE)


class _NCTDataset(Dataset):
    CLASS_MAP = {"ADI":0,"BACK":1,"DEB":2,"LYM":3,"MUC":4,
                 "MUS":5,"NORM":6,"STR":7,"TUM":8}

    def __init__(self, manifest_csv: str, root: str, objective: str, transform):
        from data_loader.build_utils import read_csv_defensively
        self.df        = read_csv_defensively(manifest_csv)
        self.root      = root
        self.objective = objective
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = os.path.join(self.root, str(row["image_key"]))
        img  = Image.open(path).convert("RGB")
        img  = self.transform(img)
        if self.objective == "supervised":
            cls = self.CLASS_MAP.get(str(row.get("tissue_class", "")), 0)
            return img, cls
        if self.objective == "image_text":
            return img, str(row.get("report_text", ""))
        return img, idx



class _MAEReconstructor(nn.Module):
    def __init__(self, encoder: nn.Module, embed_dim: int,
                 patch: int = 16, img_size: int = 224, in_chans: int = 3):
        super().__init__()
        self.encoder    = encoder
        self.patch      = patch
        self.n_side     = img_size // patch
        self.n_patches  = self.n_side ** 2
        self.patch_dim  = in_chans * patch * patch
        self.embed_dim  = embed_dim
        self.decoder    = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, self.patch_dim),
        )

    def _patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        B, C, H, W = imgs.shape
        p = self.patch
        x = imgs.reshape(B, C, self.n_side, p, self.n_side, p)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(B, self.n_patches, self.patch_dim)
        return x

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75) -> torch.Tensor:
        B = imgs.shape[0]
        target = self._patchify(imgs)                       # (B, N, patch_dim)

        # Encode the full image; use patch-token outputs as the per-patch latent
        feats = self.encoder.forward_features(imgs) \
            if hasattr(self.encoder, "forward_features") else self.encoder(imgs)
        if hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state
        # Drop CLS if present so patch tokens align to the patch grid
        if feats.dim() == 3 and feats.shape[1] == self.n_patches + 1:
            patch_feats = feats[:, 1:, :]
        elif feats.dim() == 3 and feats.shape[1] == self.n_patches:
            patch_feats = feats
        else:
            # Backbone does not expose patch tokens at this resolution; fall back
            # to broadcasting the global vector (still a valid, if weaker, target).
            g = feats if feats.dim() == 2 else feats[:, 0, :]
            patch_feats = g.unsqueeze(1).expand(B, self.n_patches, self.embed_dim)

        _dec_dtype = next(self.decoder.parameters()).dtype
        pred = self.decoder(patch_feats.to(_dec_dtype))     # (B, N, patch_dim)
        pred = pred.float()

        # Random mask
        noise = torch.rand(B, self.n_patches, device=imgs.device)
        ids   = torch.argsort(noise, dim=1)
        n_mask = int(self.n_patches * mask_ratio)
        mask  = torch.zeros(B, self.n_patches, device=imgs.device)
        mask.scatter_(1, ids[:, :n_mask], 1.0)              # 1 = masked

        loss = ((pred - target.float()) ** 2).mean(dim=-1)  # (B, N)
        loss = (loss * mask).sum() / mask.sum().clamp(min=1.0)
        return loss


def _clip_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor,
               temp: float = 0.07) -> torch.Tensor:
    """InfoNCE contrastive loss (CLIP-style)."""
    img_n = F.normalize(img_feats, p=2, dim=-1)
    txt_n = F.normalize(txt_feats, p=2, dim=-1)
    logits = img_n @ txt_n.T / temp
    labels = torch.arange(len(img_n), device=img_n.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0



def _build_vit(backbone: str, init: str, cfg: dict) -> nn.Module:
    import timm
    arch = "vit_small_patch16_224" if backbone == "vit_s" else "vit_base_patch16_224"
    if init == "imagenet":
        model = timm.create_model(arch, pretrained=True, num_classes=0)
    else:
        if init not in ("dinov3_s", "dinov3_b"):
            raise ValueError(f"Unknown init '{init}'. Expected: imagenet, dinov3_s, dinov3_b")
        hf_id = ("facebook/dinov3-vits16-pretrain-lvd1689m" if backbone == "vit_s"
                 else "facebook/dinov3-vitb16-pretrain-lvd1689m")
        from transformers import AutoModel
        hf_model = AutoModel.from_pretrained(hf_id, trust_remote_code=True)
        model = hf_model
    return model


def _infer_embed_dim(model: nn.Module, backbone: str) -> int:
    cfgobj = getattr(model, "config", None)
    if cfgobj is not None and getattr(cfgobj, "hidden_size", None):
        return int(cfgobj.hidden_size)
    for attr in ("embed_dim", "num_features"):
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 384 if backbone == "vit_s" else 768


def train_one_cell(
    run_id:    str,
    modality:  str,
    objective: str,
    backbone:  str,
    init:      str,
    seed:      int,
    global_config_path: str,
    force: bool = False,
) -> str:
    cfg   = read_config(global_config_path)["Convergence"]
    ctrl  = cfg["controlled"]
    ckpt_dir = os.path.join(ctrl["ckpts_dir"], run_id)
    done_flag = os.path.join(ckpt_dir, "training_done.flag")

    if os.path.exists(done_flag) and not force:
        print(f"[train_encoder] {run_id}: already done, skipping.")
        return ckpt_dir
    os.makedirs(ckpt_dir, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    from torchvision import transforms
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    # Build model
    model = _build_vit(backbone, init, cfg)
    bf16  = ctrl.get("bf16", True) and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = model.to(device=device, dtype=dtype)
    if ctrl.get("gradient_checkpointing", True) and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)

    embed_dim = _infer_embed_dim(model, backbone)
    nominal   = 384 if backbone == "vit_s" else 768
    if embed_dim != nominal:
        print(f"[train_encoder] {run_id}: model width {embed_dim} != nominal "
              f"{nominal} for {backbone}; using actual width {embed_dim}.")

    # Head for supervised objective
    head = None
    if objective == "supervised":
        n_out = ctrl.get("cxr_n_labels", 13) if modality == "cxr" \
                else ctrl.get("histo_n_classes", 9)
        head = nn.Linear(embed_dim, n_out).to(device=device, dtype=dtype)

    # MAE reconstructor for the SSL objective (real masked pixel reconstruction)
    mae = None
    if objective == "ssl":
        mae = _MAEReconstructor(model, embed_dim=embed_dim, patch=16,
                                img_size=224, in_chans=3).to(device=device, dtype=dtype)

    text_enc = None
    img_proj = None
    txt_proj = None
    clip_proj_dim = 256
    if objective == "image_text":
        from transformers import AutoTokenizer, AutoModel as HFModel
        tok = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )
        text_enc = HFModel.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        ).to(device=device, dtype=dtype).eval()
        txt_hidden = int(getattr(text_enc.config, "hidden_size", 768))
        img_proj = nn.Linear(embed_dim, clip_proj_dim).to(device=device, dtype=dtype)
        txt_proj = nn.Linear(txt_hidden, clip_proj_dim).to(device=device, dtype=dtype)

    # Dataset
    mix_dir = os.path.join(ctrl["ckpts_dir"], "mixtures")
    if modality == "cxr":
        suffix_map = {"ssl": "cxr_ssl_train.csv", "supervised": "cxr_supervised_train.csv",
                      "image_text": "cxr_image_text_train.csv"}
        from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
        label_cols = [f for f in CANONICAL_CXR_FINDINGS if f != "no_finding"]
        dataset = _CXRDataset(
            os.path.join(mix_dir, suffix_map[objective]),
            cfg["cxr"]["sites"]["mimic"]["image_root"],
            objective, label_cols, train_tfm,
        )
    else:
        suffix_map = {"ssl": "histo_ssl_train.csv", "supervised": "histo_supervised_train.csv",
                      "image_text": "histo_image_text_train.csv"}
        if objective == "image_text":
            histo_root = cfg["quilt"]["root"]
        else:
            histo_root = cfg["histo"]["nct_crc"]["root"]
        dataset = _NCTDataset(
            os.path.join(mix_dir, suffix_map[objective]),
            histo_root,
            objective, train_tfm,
        )

    bs  = int(ctrl.get("batch_size", 256))
    eff = int(ctrl.get("effective_batch_size", 512))
    accum_steps = max(1, eff // bs)
    loader = DataLoader(dataset, batch_size=bs, shuffle=True,
                        num_workers=4, pin_memory=True, drop_last=True)

    lr    = float(ctrl.get("lr", 1e-4))
    wd    = float(ctrl.get("weight_decay", 0.05))
    params = list(model.parameters())
    if head is not None:
        params += list(head.parameters())
    if mae is not None:
        params += list(mae.decoder.parameters())
    if img_proj is not None:
        params += list(img_proj.parameters()) + list(txt_proj.parameters())
    opt   = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    n_steps_target = len(loader) * int(ctrl.get("epochs", 50))
    warmup = int(ctrl.get("warmup_fraction", 0.1) * n_steps_target)
    def _lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps_target - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    step = 0
    for epoch in range(int(ctrl.get("epochs", 50))):
        model.train()
        if head:
            head.train()
        epoch_loss = 0.0
        opt.zero_grad()
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{run_id} ep{epoch}")):
            if objective == "ssl":
                imgs, _ = batch
                imgs = imgs.to(device=device, dtype=dtype)
                loss = mae(imgs, float(ctrl.get("mae_mask_ratio", 0.75)))
            elif objective == "supervised":
                imgs, labels = batch
                imgs   = imgs.to(device=device, dtype=dtype)
                labels = labels.to(device)
                feats  = model(imgs) if hasattr(model, "forward") else model.forward(imgs)
                if hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0, :]
                # Run the head in its own (bf16) dtype, then upcast logits to fp32
                # for the loss (feeding fp32 into a bf16 Linear errors).
                _hd = next(head.parameters()).dtype
                out  = head(feats.to(dtype=_hd)).float()
                if modality == "histo":
                    loss = F.cross_entropy(out, labels.long())
                else:
                    loss = F.binary_cross_entropy_with_logits(out, labels.to(dtype=torch.float32))
            else:   # image_text
                imgs, texts = batch
                imgs = imgs.to(device=device, dtype=dtype)
                feats = model(imgs) if hasattr(model, "forward") else model.forward(imgs)
                if hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0, :]
                enc = tok(list(texts), return_tensors="pt", padding=True,
                          truncation=True, max_length=256)
                enc = {k: v.to(device) for k, v in enc.items()}
                with torch.no_grad():
                    txt_out  = text_enc(**enc).last_hidden_state[:, 0, :]
                img_e = img_proj(feats.to(dtype=next(img_proj.parameters()).dtype))
                txt_e = txt_proj(txt_out.to(dtype=next(txt_proj.parameters()).dtype))
                loss = _clip_loss(img_e.float(), txt_e.float(),
                                  float(ctrl.get("clip_temp", 0.07)))

            (loss / accum_steps).backward()
            epoch_loss += loss.item()

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()
                step += 1


    # Save checkpoint
    state = model.state_dict() if not hasattr(model, "save_pretrained") \
            else None
    if state is not None:
        torch.save(state, os.path.join(ckpt_dir, "model.pt"))
    else:
        model.save_pretrained(ckpt_dir)
    if head:
        torch.save(head.state_dict(), os.path.join(ckpt_dir, "head.pt"))
    with open(done_flag, "w") as f:
        f.write(run_id)
    return ckpt_dir


def enumerate_e5_runs(global_config_path: str) -> List[tuple]:
    cfg  = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    objectives = ctrl.get("objectives", ["ssl", "supervised", "image_text"])
    backbones  = ctrl.get("backbones",  ["vit_s", "vit_b"])
    inits      = ctrl.get("inits",      ["dinov3_s", "imagenet"])
    modalities = ctrl.get("modalities", ["cxr", "histo"])
    n_seeds    = int(ctrl.get("n_seeds", 2))
    quilt_enabled = bool(cfg.get("quilt", {}).get("enabled", False))

    runs = []
    for mod in modalities:
        for obj in objectives:
            if mod == "histo" and obj == "image_text" and not quilt_enabled:
                continue
            for bb in backbones:
                for init in inits:
                    for seed in range(n_seeds):
                        run_id = f"{mod}__{obj}__{bb}__{init}__seed{seed}"
                        runs.append((run_id, mod, obj, bb, init, seed))
    return runs


def run_one_e5_cell(global_config_path: str, run_id: str, force: bool = False):
    runs = {r[0]: r for r in enumerate_e5_runs(global_config_path)}
    if run_id not in runs:
        raise ValueError(
            f" run_id '{run_id}' is not in the configured matrix. "
            f"Available ({len(runs)}): " + ", ".join(sorted(runs.keys()))
        )
    _, mod, obj, bb, init, seed = runs[run_id]
    return train_one_cell(run_id, mod, obj, bb, init, seed,
                          global_config_path, force=force)


def run_full_e5_matrix(global_config_path: str, force: bool = False,
                       run_ids: Optional[List[str]] = None):
    selected = set(run_ids) if run_ids else None
    for run_id, mod, obj, bb, init, seed in enumerate_e5_runs(global_config_path):
        if selected is not None and run_id not in selected:
            continue
        train_one_cell(run_id, mod, obj, bb, init, seed,
                       global_config_path, force=force)

