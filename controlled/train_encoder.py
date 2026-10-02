"""
controlled/train_encoder.py
Created on June 21, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import math
import os
from typing import List, Optional

import gc

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
from Inference.resume_utils import (write_csv_atomic, append_status, status_path, MissingInput, cell_is_done, claim_cell, clear_cell,
                                    done_flag_path, heartbeat_claim, read_done_meta, release_claim,
                                    write_done_flag, write_torch_atomic)
warnings.filterwarnings("ignore")


from controlled.matrix import (CXR_LIKE, GRADE_CLASSES, LABEL_FORMS, OBJECTIVES,
                              TWO_VIEW_OBJECTIVES, enumerate_disjoint_runs,
                              enumerate_e11_runs, enumerate_e5_runs, taix_label_columns)

VERIFY_PRECISIONS = ("fp32", "bf16_autocast")


BATCH_COUPLED_OBJECTIVES = ("simclr", "dino", "image_text")


def enable_grad_checkpointing(model: nn.Module) -> bool:
    if hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
        return True
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return True
    return False


def training_precision(ctrl: dict) -> str:
    return "bf16_autocast" if bool(ctrl.get("bf16", True)) else "fp32"


def require_bf16_gpu(precision: str, who: str) -> None:
    if precision != "fp32" and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        raise RuntimeError(f"[{who}] precision {precision} needs a GPU with bf16, and this process "
                           f"sees none usable; submit it to a node whose driver serves this torch build.")


def _objective_gate(ctrl: dict, precision: str):
    fp32 = precision == "fp32"
    out_csv = os.path.join(ctrl["results_e5_dir"], "objective_verification.csv" if fp32
                           else f"objective_verification_{precision}.csv")
    mix_dir = os.path.join(ctrl["ckpts_dir"], "mixtures")
    mixes = [os.path.join(mix_dir, f"{m}_common_train.csv")
             for m in ("cxr", "histo") if m in list(ctrl.get("modalities", []))]
    return out_csv, mixes, (None if fp32 else {"precision": precision})


def _gate_failures(out_csv: str) -> List[str]:
    prev = pd.read_csv(out_csv)
    ok = prev["learns_the_named_task"].astype(str).str.lower() == "true"
    return [f"{m}/{o}" for m, o in zip(prev.loc[~ok, "modality"], prev.loc[~ok, "objective"])]


def require_objective_gate(ctrl: dict) -> None:
    from Inference.resume_utils import output_is_current
    precision = training_precision(ctrl)
    out_csv, mixes, extra = _objective_gate(ctrl, precision)
    if not output_is_current(out_csv, mixes, owner=f"verify_objectives_{precision}", extra=extra):
        raise MissingInput(f"the objective verification in {precision} has not finished against the "
                           f"current mixtures ({out_csv}); a cell trains only after it passes.")
    failures = _gate_failures(out_csv)
    if failures:
        raise ValueError(f"[verify] {failures} failed the objective verification in {precision} "
                         f"({out_csv}); no cell trains until it passes.")


class _CXRDataset(Dataset):
    def __init__(self, manifest_csv: str, image_root: str,
                 objective: str, label_cols: list, transform,
                 report_root: str = ""):
        from data_loader.build_utils import read_csv_defensively
        self.df          = read_csv_defensively(manifest_csv)
        self.image_root  = image_root
        self.objective   = objective
        self.label_cols  = label_cols
        self.transform   = transform
        self.report_root = report_root or image_root

    def assert_text_present(self, n_sample: int = 300, min_frac: float = 0.1):
        if self.objective != "image_text":
            return
        idx = np.linspace(0, len(self.df) - 1, min(n_sample, len(self.df))).astype(int)
        nonempty = sum(1 for i in idx if self._report_text(self.df.iloc[int(i)]).strip())
        if nonempty / max(1, len(idx)) < min_frac:
            raise MissingInput(
                f"report text is empty for {len(idx) - nonempty} of {len(idx)} sampled rows. "
                f"The manifest must carry report_text inline or a report_rel_path that resolves "
                f"under {self.report_root!r}; a path alone is not text.")

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        from data_loader.cxr_harmonization import resolve_cxr_image_path
        path  = resolve_cxr_image_path("mimic", self.image_root,
                                        str(row["image_key"]))
        img   = Image.open(path).convert("RGB")
        img   = self.transform(img)
        if self.objective == "supervised":
            vals = []
            for f in self.label_cols:
                v = row.get(f, float("nan"))
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = float("nan")
                vals.append(v if v in (0.0, 1.0) else float("nan"))
            return img, torch.tensor(vals, dtype=torch.float32)
        if self.objective == "image_text":
            return img, self._report_text(row)
        if self.objective in ("simclr", "dino"):
            return self.transform(Image.open(path).convert("RGB")), img
        return img, idx


    def _report_text(self, row) -> str:
        inline = row.get("report_text")
        if isinstance(inline, str) and inline.strip():
            return inline
        rel = str(row.get("report_rel_path", "") or "")
        if not rel:
            return ""
        full = rel if os.path.isabs(rel) else os.path.join(self.report_root, rel)
        try:
            with open(full, "r", errors="ignore") as fh:
                return fh.read()
        except OSError:
            return ""


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
        if self.objective in TWO_VIEW_OBJECTIVES:
            return self.transform(Image.open(path).convert("RGB")), img
        return img, idx


class _TAIXDataset(Dataset):
    def __init__(self, manifest_csv: str, image_root: str, label_form: str,
                 label_cols: List[str], transform):
        from data_loader.build_utils import read_csv_defensively
        self.df = read_csv_defensively(manifest_csv)
        missing = [c for c in label_cols if c not in self.df.columns]
        if missing:
            raise MissingInput(
                f"the TAIX training mixture carries none of {missing} for label form "
                f"'{label_form}'. main_build_taix_pool writes the side-resolved and graded columns; a mixture "
                f"built before they existed has to be rebuilt.")
        self.image_root = image_root
        self.label_form = label_form
        self.label_cols = label_cols
        self.transform = transform

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.transform(Image.open(
            os.path.join(self.image_root, str(row["image_key"]))).convert("RGB"))
        if self.label_form == "severity":
            grades = [int(row[c]) if pd.notna(row[c]) else -100 for c in self.label_cols]
            return img, torch.tensor(grades, dtype=torch.long)
        vals = [1.0 if row.get(c, float("nan")) == 1.0 else 0.0 for c in self.label_cols]
        return img, torch.tensor(vals, dtype=torch.float32)


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

    def _draw_mask(self, batch: int, mask_ratio: float, device) -> torch.Tensor:
        noise = torch.rand(batch, self.n_patches, device=device)
        ids = torch.argsort(noise, dim=1)
        n_mask = int(self.n_patches * mask_ratio)
        mask = torch.zeros(batch, self.n_patches, device=device)
        mask.scatter_(1, ids[:, :n_mask], 1.0)
        return mask

    def _keep_pixels(self, mask: torch.Tensor, dtype) -> torch.Tensor:
        keep = (1.0 - mask).view(-1, self.n_side, self.n_side)
        keep = keep.repeat_interleave(self.patch, dim=1).repeat_interleave(self.patch, dim=2)
        return keep.unsqueeze(1).to(dtype)

    def _predict(self, imgs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        masked_imgs = imgs * self._keep_pixels(mask, imgs.dtype)

        feats = self.encoder.forward_features(masked_imgs) \
            if hasattr(self.encoder, "forward_features") else self.encoder(masked_imgs)
        if hasattr(feats, "last_hidden_state"):
            feats = feats.last_hidden_state
        if feats.dim() != 3 or int(feats.shape[1]) < self.n_patches:
            raise ValueError(
                f"[mae] the encoder returned {tuple(feats.shape)} for {self.n_patches} patches; "
                f"per-patch reconstruction needs one token per patch.")
        n_prefix = int(feats.shape[1]) - self.n_patches
        n_reg = int(getattr(getattr(self.encoder, "config", None), "num_register_tokens", 0) or 0)
        if n_prefix not in (0, 1, 1 + n_reg):
            raise ValueError(
                f"[mae] the encoder returned {int(feats.shape[1])} tokens for {self.n_patches} "
                f"patches with {n_reg} register tokens declared, a layout this reconstructor does "
                f"not know, so which tokens are the patches is not decidable here.")
        patch_feats = feats[:, -self.n_patches:, :]

        _dec_dtype = next(self.decoder.parameters()).dtype
        return self.decoder(patch_feats.to(_dec_dtype)).float()

    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.75,
                mask: torch.Tensor = None) -> torch.Tensor:
        target = self._patchify(imgs)
        if mask is None:
            mask = self._draw_mask(imgs.shape[0], mask_ratio, imgs.device)
        pred = self._predict(imgs, mask)

        loss = ((pred - target.float()) ** 2).mean(dim=-1)
        self.last_masked = float((loss * mask).sum() / mask.sum().clamp(min=1.0))
        self.last_visible = float((loss * (1.0 - mask)).sum() / (1.0 - mask).sum().clamp(min=1.0))
        return (loss * mask).sum() / mask.sum().clamp(min=1.0)

    @torch.no_grad()
    def assert_bottleneck(self, imgs: torch.Tensor, mask_ratio: float, tol: float = 1e-5):
        was_training = self.training
        self.eval()
        try:
            self._assert_bottleneck(imgs, mask_ratio, tol)
        finally:
            self.train(was_training)

    def _assert_bottleneck(self, imgs: torch.Tensor, mask_ratio: float, tol: float):
        mask = self._draw_mask(imgs.shape[0], mask_ratio, imgs.device)
        n_hidden, n_shown = float(mask.sum()), float((1.0 - mask).sum())
        if n_hidden == 0 or n_shown == 0:
            raise ValueError(
                f"[mae] a mask ratio of {mask_ratio} withholds {int(n_hidden)} of "
                f"{self.n_patches} patches and shows {int(n_shown)}. The objective needs both "
                f"sides, so neither count can be zero.")

        keep_px = self._keep_pixels(mask, imgs.dtype)
        hidden_perturbed = imgs * keep_px + torch.randn_like(imgs) * (1.0 - keep_px)
        shown_perturbed = imgs * (1.0 - keep_px) + torch.randn_like(imgs) * keep_px

        base = self._predict(imgs, mask)
        scale = float(base.abs().max().clamp(min=1e-12))
        spread = float(base.float().std(dim=1).mean())
        if spread <= tol * scale:
            raise ValueError(
                f"[mae] every patch position receives the same prediction (spread {spread:.3e} "
                f"against a scale of {scale:.3e}), so the decoder is not reading per-patch tokens.")
        d_hidden = float((self._predict(hidden_perturbed, mask) - base).abs().max())
        d_shown = float((self._predict(shown_perturbed, mask) - base).abs().max())

        if d_hidden > tol * scale:
            raise ValueError(
                f"[mae] no bottleneck: changing ONLY the masked pixels moved the prediction by "
                f"{d_hidden:.3e} against a prediction scale of {scale:.3e}. The withheld pixels "
                f"reach the encoder, so the task is solvable by identity and the mask is on the "
                f"loss and not on the encoder input.")
        if d_shown <= tol * scale:
            raise ValueError(
                f"[mae] the encoder ignores its input: changing the VISIBLE pixels moved the "
                f"prediction by {d_shown:.3e} against a prediction scale of {scale:.3e}. A "
                f"reconstruction that does not depend on what the encoder was shown is a constant "
                f"and the cell would measure nothing.")
        self.forward(imgs, mask_ratio, mask=mask)

def _clip_loss(img_feats: torch.Tensor, txt_feats: torch.Tensor,
               temp: float = 0.07) -> torch.Tensor:
    img_n = F.normalize(img_feats, p=2, dim=-1)
    txt_n = F.normalize(txt_feats, p=2, dim=-1)
    logits = img_n @ txt_n.T / temp
    labels = torch.arange(len(img_n), device=img_n.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2.0


def _forward_global(model: nn.Module, imgs: torch.Tensor) -> torch.Tensor:
    feats = model(imgs)
    if hasattr(feats, "last_hidden_state"):
        feats = feats.last_hidden_state[:, 0, :]
    if feats.dim() == 3:
        feats = feats[:, 0, :]
    return feats


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temp: float = 0.2) -> torch.Tensor:
    n = z1.shape[0]
    z = F.normalize(torch.cat([z1, z2], dim=0), p=2, dim=-1)
    sim = (z @ z.T) / temp
    sim.fill_diagonal_(float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


class _DINOHead(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x): return self.mlp(x)


class _DinoSelfDistiller(nn.Module):

    def __init__(self, student: nn.Module, embed_dim: int, out_dim: int = 2048,
                 momentum: float = 0.996, temp_s: float = 0.1, temp_t: float = 0.04,
                 center_momentum: float = 0.9):
        super().__init__()
        import copy
        self.student = student
        self.student_head = _DINOHead(embed_dim, out_dim)
        self.teacher = copy.deepcopy(student)
        self.teacher_head = _DINOHead(embed_dim, out_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for q in list(self.teacher.parameters()) + list(self.teacher_head.parameters()):
            q.requires_grad = False
        self.momentum = float(momentum)
        self.temp_s = float(temp_s)
        self.temp_t = float(temp_t)
        self.center_momentum = float(center_momentum)
        self.register_buffer("center", torch.zeros(1, out_dim))

    @torch.no_grad()
    def update_teacher(self):
        for pt, ps in zip(self.teacher.parameters(), self.student.parameters()):
            pt.data.mul_(self.momentum).add_(ps.data, alpha=1.0 - self.momentum)
        for pt, ps in zip(self.teacher_head.parameters(), self.student_head.parameters()):
            pt.data.mul_(self.momentum).add_(ps.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def update_center(self, t_out: torch.Tensor):
        batch_center = t_out.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(
            batch_center.to(self.center.dtype), alpha=1.0 - self.center_momentum)

    def forward(self, v1: torch.Tensor, v2: torch.Tensor) -> torch.Tensor:
        hd = next(self.student_head.parameters()).dtype
        s1 = self.student_head(_forward_global(self.student, v1).to(hd)).float()
        s2 = self.student_head(_forward_global(self.student, v2).to(hd)).float()
        with torch.no_grad():
            t1 = self.teacher_head(_forward_global(self.teacher, v1).to(hd)).float()
            t2 = self.teacher_head(_forward_global(self.teacher, v2).to(hd)).float()
        c = self.center.float()
        p_t1 = F.softmax((t1 - c) / self.temp_t, dim=-1)
        p_t2 = F.softmax((t2 - c) / self.temp_t, dim=-1)
        loss = 0.5 * (
            -(p_t2 * F.log_softmax(s1 / self.temp_s, dim=-1)).sum(dim=-1).mean()
            - (p_t1 * F.log_softmax(s2 / self.temp_s, dim=-1)).sum(dim=-1).mean()
        )
        self.update_center(torch.cat([t1, t2], dim=0))
        return loss


def _build_vit(backbone: str, init: str, cfg: dict) -> nn.Module:
    import timm
    from encoders.panel import export_hf_token, resolve_hf_token
    token = resolve_hf_token(cfg)
    export_hf_token(token)
    arch = "vit_small_patch16_224" if backbone == "vit_s" else "vit_base_patch16_224"
    if init == "random":
        return timm.create_model(arch, pretrained=False, num_classes=0)
    if init == "imagenet":
        return timm.create_model(arch, pretrained=True, num_classes=0)
    if init not in ("dinov3", "dinov3_s", "dinov3_b"):
        raise ValueError(
            f"Unknown init '{init}'. Expected one of: random, imagenet, dinov3.")
    hf_id = ("facebook/dinov3-vits16-pretrain-lvd1689m" if backbone == "vit_s"
             else "facebook/dinov3-vitb16-pretrain-lvd1689m")
    from transformers import AutoModel
    return AutoModel.from_pretrained(hf_id, token=token, trust_remote_code=True)


def _infer_embed_dim(model: nn.Module, backbone: str) -> int:
    cfgobj = getattr(model, "config", None)
    if cfgobj is not None and getattr(cfgobj, "hidden_size", None):
        return int(cfgobj.hidden_size)
    for attr in ("embed_dim", "num_features"):
        v = getattr(model, attr, None)
        if isinstance(v, int) and v > 0:
            return v
    return 384 if backbone == "vit_s" else 768


def _trainable_modules(model, head, mae, dino, simclr_proj, img_proj, txt_proj,
                       text_enc=None) -> dict:
    out = {"model": model}
    for name, mod in (("head", head), ("mae_decoder", getattr(mae, "decoder", None)),
                      ("dino_student_head", getattr(dino, "student_head", None)),
                      ("dino_teacher", getattr(dino, "teacher", None)),
                      ("dino_teacher_head", getattr(dino, "teacher_head", None)),
                      ("simclr_proj", simclr_proj), ("img_proj", img_proj),
                      ("txt_proj", txt_proj), ("text_enc", text_enc)):
        if mod is not None:
            out[name] = mod
    return out


def _save_resume(path: str, recipe: dict, modules: dict, opt, scheduler,
                 step: int, epoch: int, dino) -> None:
    state = {
        "recipe": recipe,
        "step": int(step),
        "epoch": int(epoch),
        "modules": {k: v.state_dict() for k, v in modules.items()},
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
    }
    if torch.cuda.is_available():
        state["cuda_rng"] = torch.cuda.get_rng_state_all()
    if dino is not None:
        state["dino_center"] = dino.center.detach().cpu()
    write_torch_atomic(state, path)


def _load_resume(path: str, recipe: dict, run_id: str):
    if not os.path.exists(path):
        return None
    try:
        state = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        return None
    stored = state.get("recipe", {})
    moved = {k: (stored.get(k), v) for k, v in recipe.items() if stored.get(k) != v}
    if moved:
        lines = ", ".join(f"{k}: {was!r} -> {now!r}" for k, (was, now) in moved.items())
        return None
    return state


def _worker_kwargs(n_workers: int) -> dict:
    return {"persistent_workers": True, "prefetch_factor": 4} if n_workers > 0 else {}


def _weights_present(ckpt_dir: str) -> bool:
    pt = os.path.join(ckpt_dir, "model.pt")
    if os.path.exists(pt) and os.path.getsize(pt) > 0:
        return True
    if not os.path.exists(os.path.join(ckpt_dir, "config.json")):
        return False
    for w in ("model.safetensors", "pytorch_model.bin"):
        q = os.path.join(ckpt_dir, w)
        if os.path.exists(q) and os.path.getsize(q) > 0:
            return True
    return False


def cell_regime(objective: str, ctrl: dict) -> dict:
    regime = {"precision": training_precision(ctrl), "step_budget": int(ctrl["max_steps"])}
    if objective == "mae":
        regime["mae_decoder_input"] = "patch_tokens"
    if objective == "image_text":
        trained = bool(ctrl.get("image_text_train_text_tower", True))
        regime["image_text_text_tower"] = "trained" if trained else "frozen"
    return regime


def _finished_under_this_regime(ckpt_dir: str, run_id: str, init: str, regime: dict) -> bool:
    if not (cell_is_done(ckpt_dir) and _weights_present(ckpt_dir)):
        return False
    if not regime:
        return True
    meta = read_done_meta(ckpt_dir)
    legacy = {"mae_decoder_input": "patch_tokens" if init == "random" else "cls_expanded",
              "image_text_text_tower": "frozen", "precision": "bf16_weights", "step_budget": 20000}
    for key, want in regime.items():
        have = meta.get(key, legacy[key])
        if have != want:
            clear_cell(ckpt_dir)
            return False
    return True


def train_one_cell(
    run_id:    str,
    modality:  str,
    objective: str,
    backbone:  str,
    init:      str,
    seed:      int,
    global_config_path: str,
    force: bool = False,
    label_form: str = "",
) -> str:
    if modality == "taix" and label_form not in LABEL_FORMS:
        raise ValueError(f"[E11] {run_id}: label form '{label_form}' is not one of "
                         f"{', '.join(LABEL_FORMS)}.")
    if objective not in OBJECTIVES:
        raise ValueError(
            f"[E5] {run_id}: objective '{objective}' has no branch in the training loop. "
            f"Implemented: {', '.join(OBJECTIVES)}. An objective the loop does not know would "
            f"fall through to another arm and train the wrong thing under the right name.")
    cfg   = read_config(global_config_path)["Convergence"]
    ctrl  = cfg["controlled"]
    ckpt_dir = os.path.join(ctrl["ckpts_dir"], run_id)
    done_flag = done_flag_path(ckpt_dir)
    regime = cell_regime(objective, ctrl)

    if force:
        clear_cell(ckpt_dir)
    if _finished_under_this_regime(ckpt_dir, run_id, init, regime) and not force:
        return ckpt_dir
    require_objective_gate(ctrl)
    _mix_csv = os.path.join(ctrl["ckpts_dir"], "mixtures", f"{modality}_common_train.csv")
    if not os.path.exists(_mix_csv):
        raise MissingInput(f"the training mixture is absent at {_mix_csv}; run main_build_training_mixtures first.")
    os.makedirs(ckpt_dir, exist_ok=True)
    if not claim_cell(ckpt_dir, run_id, owner=run_id):
        return ckpt_dir
    if _finished_under_this_regime(ckpt_dir, run_id, init, regime) and not force:
        release_claim(ckpt_dir, run_id)
        return ckpt_dir
    try:
        return _train_claimed_cell(run_id, modality, objective, backbone, init, seed, cfg, ctrl,
                                   ckpt_dir, regime, label_form)
    except BaseException:
        release_claim(ckpt_dir, run_id)
        raise


def _train_claimed_cell(run_id, modality, objective, backbone, init, seed, cfg, ctrl, ckpt_dir,
                        regime, label_form) -> str:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision import transforms
    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    model = _build_vit(backbone, init, cfg)
    precision = regime["precision"]
    require_bf16_gpu(precision, run_id)
    dtype = torch.float32
    model = model.to(device=device, dtype=dtype)
    if ctrl.get("gradient_checkpointing", True) and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    if objective in BATCH_COUPLED_OBJECTIVES:
        enable_grad_checkpointing(model)

    embed_dim = _infer_embed_dim(model, backbone)
    nominal   = 384 if backbone == "vit_s" else 768

    head = None
    taix_cols: List[str] = []
    if objective == "supervised":
        if modality == "taix":
            taix_cols = taix_label_columns(label_form, cfg["taix"])
            n_out = (len(taix_cols) * GRADE_CLASSES if label_form == "severity"
                     else len(taix_cols))
        else:
            n_out = ctrl.get("cxr_n_labels", 13) if modality == "cxr" \
                    else ctrl.get("histo_n_classes", 9)
        head = nn.Linear(embed_dim, n_out).to(device=device, dtype=dtype)

    mae = None
    if objective == "mae":
        mae = _MAEReconstructor(model, embed_dim=embed_dim, patch=16,
                                img_size=224, in_chans=3).to(device=device, dtype=dtype)

    simclr_proj = None
    if objective == "simclr":
        simclr_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, int(ctrl.get("simclr_proj_dim", 128))),
        ).to(device=device, dtype=dtype)

    dino = None
    if objective == "dino":
        dino = _DinoSelfDistiller(
            model, embed_dim=embed_dim,
            out_dim=int(ctrl.get("dino_out_dim", 2048)),
            momentum=float(ctrl.get("dino_teacher_momentum", 0.996)),
            temp_s=float(ctrl.get("dino_temp_student", 0.1)),
            temp_t=float(ctrl.get("dino_temp_teacher", 0.04)),
            center_momentum=float(ctrl.get("dino_center_momentum", 0.9)),
        ).to(device=device, dtype=dtype)

    text_enc = None
    img_proj = None
    txt_proj = None
    clip_proj_dim = 256
    train_text_tower = False
    if objective == "image_text":
        from transformers import AutoTokenizer, AutoModel as HFModel
        tok = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        )
        text_enc = HFModel.from_pretrained(
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        ).to(device=device, dtype=dtype)
        train_text_tower = bool(ctrl.get("image_text_train_text_tower", True))
        if train_text_tower:
            text_enc.train()
            if ctrl.get("gradient_checkpointing", True) and hasattr(text_enc, "gradient_checkpointing_enable"):
                text_enc.gradient_checkpointing_enable()
        else:
            text_enc.eval()
            for q in text_enc.parameters():
                q.requires_grad = False
        txt_hidden = int(getattr(text_enc.config, "hidden_size", 768))
        img_proj = nn.Linear(embed_dim, clip_proj_dim).to(device=device, dtype=dtype)
        txt_proj = nn.Linear(txt_hidden, clip_proj_dim).to(device=device, dtype=dtype)

    mix_dir = os.path.join(ctrl["ckpts_dir"], "mixtures")
    if modality in CXR_LIKE:
        common_csv = os.path.join(mix_dir, f"{modality}_common_train.csv")
        from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
        label_cols = [f for f in CANONICAL_CXR_FINDINGS if f != "no_finding"]
        dataset = _CXRDataset(
            common_csv,
            cfg["cxr"]["sites"]["mimic"]["image_root"],
            objective, label_cols, train_tfm,
            report_root=cfg["cxr"]["sites"]["mimic"]["image_root"],
        )
        dataset.assert_text_present()
    elif modality == "taix":
        common_csv = os.path.join(mix_dir, "taix_common_train.csv")
        dataset = _TAIXDataset(common_csv, cfg["taix"]["image_root"],
                               label_form, taix_cols, train_tfm)
    else:
        if objective == "image_text":
            raise ValueError(
                f"[E5] {run_id}: no captioned histopathology collection shares images with the "
                f"labeled one, so an image-text cell here would train on different data under the "
                f"same name. The level is excluded in controlled.modality_objective_exclusions.")
        common_csv = os.path.join(mix_dir, "histo_common_train.csv")
        dataset = _NCTDataset(common_csv, cfg["histo"]["nct_crc"]["root"],
                              objective, train_tfm)

    bs  = int(ctrl.get("batch_size", 256))
    eff = int(ctrl.get("effective_batch_size", 512))
    accum_steps = max(1, eff // bs)
    if len(dataset) < bs:
        raise MissingInput(f"the training mixture holds {len(dataset)} rows, fewer than the batch "
                           f"size of {bs}; with drop_last the cell would train on zero batches and "
                           f"still write a checkpoint and a done flag.")
    _nw = int(ctrl.get("dataloader_workers", 4))
    loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=_nw,
                        pin_memory=True, drop_last=True, **_worker_kwargs(_nw))

    lr    = float(ctrl.get("lr", 1e-4))
    wd    = float(ctrl.get("weight_decay", 0.05))
    params = list(model.parameters())
    if head is not None:
        params += list(head.parameters())
    if mae is not None:
        params += list(mae.decoder.parameters())
    if simclr_proj is not None:
        params += list(simclr_proj.parameters())
    if dino is not None:
        params += list(dino.student_head.parameters())
    if img_proj is not None:
        params += list(img_proj.parameters()) + list(txt_proj.parameters())
    if text_enc is not None and train_text_tower:
        params += list(text_enc.parameters())
    opt   = torch.optim.AdamW(params, lr=lr, weight_decay=wd,
                              fused=bool(ctrl.get("adamw_fused", False)) and device.type == "cuda")
    max_steps = int(ctrl["max_steps"])
    n_steps_target = max_steps
    warmup = int(ctrl.get("warmup_fraction", 0.1) * n_steps_target)
    def _lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, n_steps_target - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    recipe = {"run_id": run_id, "modality": modality, "objective": objective,
              "backbone": backbone, "init": init, "seed": int(seed),
              "label_form": label_form, "max_steps": int(max_steps),
              "effective_batch_size": eff, **regime}
    resume_path = os.path.join(ckpt_dir, "resume.pt")
    modules = _trainable_modules(model, head, mae, dino, simclr_proj, img_proj, txt_proj,
                                 text_enc if train_text_tower else None)

    step, start_epoch = 0, 0
    state = _load_resume(resume_path, recipe, run_id)
    if state is not None:
        for name, mod in modules.items():
            if name in state["modules"]:
                mod.load_state_dict(state["modules"][name])
        opt.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        if torch.cuda.is_available() and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        if dino is not None and "dino_center" in state:
            dino.center.copy_(state["dino_center"].to(dino.center.device))
        step, start_epoch = int(state["step"]), int(state["epoch"])
        del state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _probe = next(iter(loader))
    if objective == "image_text":
        for _t in list(_probe[1])[:3]:
            _t = str(_t).replace("\n", " ")
    elif objective == "supervised":
        _first = _probe[1][0].tolist()
        _first = _first[:8] if isinstance(_first, list) else [_first]
    elif objective in TWO_VIEW_OBJECTIVES:
        _same = bool(torch.equal(_probe[0], _probe[1]))
        if _same:
            raise ValueError(
                f"[{objective}] the 2 views are identical, so the augmentation is a no-op and the "
                f"contrastive task is solvable by identity.")
    if mae is not None:
        mae.assert_bottleneck(_probe[0].to(device=device, dtype=dtype),
                              float(ctrl.get("mae_mask_ratio", 0.75)))
    del _probe

    _bb_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _bb_total = sum(p.numel() for p in model.parameters())
    if _bb_train == 0:
        raise ValueError(f"[train_encoder] {run_id}: the backbone has no trainable parameter, so "
                         f"this cell would measure its initialization and nothing else.")
    if text_enc is not None:
        _tx_train = sum(p.numel() for p in text_enc.parameters() if p.requires_grad)

    ckpt_every = int(ctrl.get("checkpoint_every_steps", 200))
    epochs = max(1, -(-max_steps * accum_steps // max(1, len(loader))))
    def _batch_loss(batch):
        if objective == "mae":
            imgs, _ = batch
            imgs = imgs.to(device=device, dtype=dtype)
            loss = mae(imgs, float(ctrl.get("mae_mask_ratio", 0.75)))
        elif objective == "simclr":
            v1, v2 = batch
            v1 = v1.to(device=device, dtype=dtype)
            v2 = v2.to(device=device, dtype=dtype)
            _pd = next(simclr_proj.parameters()).dtype
            z1 = simclr_proj(_forward_global(model, v1).to(_pd)).float()
            z2 = simclr_proj(_forward_global(model, v2).to(_pd)).float()
            loss = _nt_xent_loss(z1, z2, float(ctrl.get("simclr_temp", 0.2)))
        elif objective == "dino":
            v1, v2 = batch
            loss = dino(v1.to(device=device, dtype=dtype),
                        v2.to(device=device, dtype=dtype))
        elif objective == "supervised":
            imgs, labels = batch
            imgs   = imgs.to(device=device, dtype=dtype)
            labels = labels.to(device)
            feats  = model(imgs) if hasattr(model, "forward") else model.forward(imgs)
            if hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0, :]
            _hd = next(head.parameters()).dtype
            out  = head(feats.to(dtype=_hd)).float()
            if modality == "histo":
                loss = F.cross_entropy(out, labels.long())
            elif modality == "taix" and label_form == "severity":
                logits = out.view(out.shape[0], len(taix_cols), GRADE_CLASSES)
                loss = F.cross_entropy(
                    logits.reshape(-1, GRADE_CLASSES), labels.reshape(-1),
                    ignore_index=-100)
            else:
                y = labels.to(dtype=torch.float32)
                m = torch.isfinite(y)
                raw = F.binary_cross_entropy_with_logits(
                    out, torch.nan_to_num(y), reduction="none")
                loss = (raw * m).sum() / m.sum().clamp(min=1)
        elif objective == "image_text":
            imgs, texts = batch
            imgs = imgs.to(device=device, dtype=dtype)
            feats = model(imgs) if hasattr(model, "forward") else model.forward(imgs)
            if hasattr(feats, "last_hidden_state"):
                feats = feats.last_hidden_state[:, 0, :]
            enc = tok(list(texts), return_tensors="pt", padding=True,
                      truncation=True, max_length=256)
            enc = {k: v.to(device) for k, v in enc.items()}
            if train_text_tower:
                txt_out = text_enc(**enc).last_hidden_state[:, 0, :]
            else:
                with torch.no_grad():
                    txt_out = text_enc(**enc).last_hidden_state[:, 0, :]
            img_e = img_proj(feats.to(dtype=next(img_proj.parameters()).dtype))
            txt_e = txt_proj(txt_out.to(dtype=next(txt_proj.parameters()).dtype))
            loss = _clip_loss(img_e.float(), txt_e.float(),
                              float(ctrl.get("clip_temp", 0.07)))
        else:
            raise ValueError(f"[E5] {run_id}: no training branch for objective "
                             f"'{objective}'.")
        return loss

    def _backward_with_oom_retry(batch, weight: float, n_full: int = 0):
        n = len(batch[0])
        if n_full == 0:
            n_full = n
        oom = False
        phase = "forward"
        try:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=precision == "bf16_autocast"):
                loss = _batch_loss(batch)
            phase = "backward"
            (loss * weight * (n / n_full)).backward()
            return float(loss) * (n / n_full)
        except torch.cuda.OutOfMemoryError:
            oom = True
        finally:
            loss = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if not oom:
            raise RuntimeError("unreachable")
        free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
        card = f"The card reports {free / 1024 ** 3:.1f} GiB free of {total / 1024 ** 3:.1f} GiB; move this cell to a larger GPU."
        if phase == "backward":
            raise torch.cuda.OutOfMemoryError(
                f"[train_encoder] {run_id}: out of memory in the backward pass at batch {n}, where a retry would count part of the batch twice. {card}")
        if objective in BATCH_COUPLED_OBJECTIVES:
            raise torch.cuda.OutOfMemoryError(
                f"[train_encoder] {run_id}: out of memory at batch {n}. The {objective} loss takes its negatives or its center from the whole batch, so a split batch would train a different objective. {card}")
        if n == 1:
            free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
            n_train = sum(p.numel() for p in params if p.requires_grad)
            raise torch.cuda.OutOfMemoryError(
                f"[train_encoder] {run_id}: out of memory at a batch of ONE, so no batch setting "
                f"can fix it. {n_train / 1e6:.0f}M trainable parameters need about "
                f"{n_train * 16 / 1024 ** 3:.1f} GiB for the weights, the gradients and AdamW's "
                f"2 moments before a single activation, and the card reports "
                f"{free / 1024 ** 3:.1f} GiB free of {total / 1024 ** 3:.1f} GiB. "
                f"Move this cell to a larger GPU.")
        half = n // 2
        a = _backward_with_oom_retry(tuple(x[:half] for x in batch), weight, n_full)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        b = _backward_with_oom_retry(tuple(x[half:] for x in batch), weight, n_full)
        return a + b

    for epoch in range(start_epoch, epochs):
        model.train()
        if head:
            head.train()
        if text_enc is not None and train_text_tower:
            text_enc.train()
        epoch_loss = 0.0
        opt.zero_grad()
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"{run_id} ep{epoch}")):
            loss = _backward_with_oom_retry(batch, 1.0 / accum_steps)
            epoch_loss += float(loss)

            if (batch_idx + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                scheduler.step()
                opt.zero_grad()
                if dino is not None:
                    dino.update_teacher()
                step += 1
                if step % ckpt_every == 0:
                    _save_resume(resume_path, recipe, modules, opt, scheduler,
                                 step, epoch, dino)
                    heartbeat_claim(ckpt_dir, run_id)
                if step >= max_steps:
                    break
        if step >= max_steps:
            break

        _save_resume(resume_path, recipe, modules, opt, scheduler, step, epoch + 1, dino)
        heartbeat_claim(ckpt_dir, run_id)
    state = model.state_dict() if not hasattr(model, "save_pretrained") \
            else None
    if state is not None:
        write_torch_atomic(state, os.path.join(ckpt_dir, "model.pt"))
    else:
        model.save_pretrained(ckpt_dir)
    if head:
        write_torch_atomic(head.state_dict(), os.path.join(ckpt_dir, "head.pt"))
    if os.path.exists(resume_path):
        os.remove(resume_path)
    write_done_flag(ckpt_dir, meta={"run_id": run_id, "objective": objective,
                                    "backbone": backbone, "init": init, "seed": seed,
                                    "label_form": label_form, "steps": int(step), **regime})
    release_claim(ckpt_dir, run_id)
    append_status(status_path(cfg, "controlled_training"),
                  f"{run_id}: finished at step {step}")
    return ckpt_dir


def _verify_label_columns(modality: str) -> Optional[List[str]]:
    if modality == "histo":
        return None
    from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
    return [f for f in CANONICAL_CXR_FINDINGS if f != "no_finding"]


def _verify_attachments(objective: str, modality: str, model: nn.Module, embed_dim: int,
                        cfg: dict, device) -> dict:
    ctrl = cfg["controlled"]
    parts = {"model": model, "mae": None, "simclr_proj": None, "dino": None, "head": None,
             "text_enc": None, "tok": None, "img_proj": None, "txt_proj": None,
             "train_text_tower": False, "params": []}
    if objective == "mae":
        parts["mae"] = _MAEReconstructor(model, embed_dim=embed_dim).to(device)
        parts["params"] += list(parts["mae"].decoder.parameters())
    elif objective == "simclr":
        parts["simclr_proj"] = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(),
            nn.Linear(embed_dim, int(ctrl.get("simclr_proj_dim", 128)))).to(device)
        parts["params"] += list(parts["simclr_proj"].parameters())
    elif objective == "dino":
        parts["dino"] = _DinoSelfDistiller(
            model, embed_dim=embed_dim,
            out_dim=int(ctrl.get("dino_out_dim", 2048)),
            momentum=float(ctrl.get("dino_teacher_momentum", 0.996)),
            temp_s=float(ctrl.get("dino_temp_student", 0.1)),
            temp_t=float(ctrl.get("dino_temp_teacher", 0.04)),
            center_momentum=float(ctrl.get("dino_center_momentum", 0.9))).to(device)
        parts["params"] += list(parts["dino"].student_head.parameters())
    elif objective == "supervised":
        cols = _verify_label_columns(modality)
        n_out = int(ctrl.get("histo_n_classes", 9)) if cols is None else len(cols)
        parts["head"] = nn.Linear(embed_dim, n_out).to(device)
        parts["params"] += list(parts["head"].parameters())
    elif objective == "image_text":
        from transformers import AutoTokenizer, AutoModel as HFModel
        hf = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
        parts["tok"] = AutoTokenizer.from_pretrained(hf)
        text_enc = HFModel.from_pretrained(hf).to(device)
        parts["text_enc"] = text_enc
        parts["train_text_tower"] = bool(ctrl.get("image_text_train_text_tower", True))
        if not parts["train_text_tower"]:
            for q in text_enc.parameters():
                q.requires_grad = False
        parts["img_proj"] = nn.Linear(embed_dim, 256).to(device)
        parts["txt_proj"] = nn.Linear(int(text_enc.config.hidden_size), 256).to(device)
        parts["params"] += (list(parts["img_proj"].parameters())
                            + list(parts["txt_proj"].parameters()))
        if parts["train_text_tower"]:
            parts["params"] += list(text_enc.parameters())
    else:
        raise ValueError(f"[verify] objective '{objective}' has no verification branch, so the "
                         f"gate would pass it without computing anything.")
    return parts


def _verify_batch_loss(objective: str, modality: str, parts: dict, batch, ctrl: dict,
                       device, scramble_rng=None, mae_mask=None):
    model = parts["model"]
    dt = next(model.parameters()).dtype

    def _perm(n):
        return torch.from_numpy(scramble_rng.permutation(n)).to(device)

    if objective == "mae":
        imgs = batch[0].to(device=device, dtype=dt)
        mae = parts["mae"]
        mask = (mae._draw_mask(imgs.shape[0], float(ctrl.get("mae_mask_ratio", 0.75)), imgs.device)
                if mae_mask is None else mae_mask.to(imgs.device))
        target = imgs if scramble_rng is None else imgs[_perm(len(imgs))]
        per_patch = ((mae._predict(imgs, mask) - mae._patchify(target).float()) ** 2).mean(dim=-1)
        return (per_patch * mask).sum() / mask.sum().clamp(min=1.0)

    if objective in TWO_VIEW_OBJECTIVES:
        v1, v2 = batch[0].to(device=device, dtype=dt), batch[1].to(device=device, dtype=dt)
        if scramble_rng is not None:
            v2 = v2[_perm(len(v2))]
        if objective == "simclr":
            z1 = parts["simclr_proj"](_forward_global(model, v1))
            z2 = parts["simclr_proj"](_forward_global(model, v2))
            return _nt_xent_loss(z1.float(), z2.float(), float(ctrl.get("simclr_temp", 0.2)))
        return parts["dino"](v1, v2)

    if objective == "supervised":
        imgs, labels = batch[0].to(device=device, dtype=dt), batch[1].to(device)
        if scramble_rng is not None:
            labels = labels[_perm(len(labels))]
        out = parts["head"](_forward_global(model, imgs)).float()
        if modality == "histo":
            return F.cross_entropy(out, labels.long())
        y = labels.float()
        m = torch.isfinite(y)
        raw = F.binary_cross_entropy_with_logits(out, torch.nan_to_num(y), reduction="none")
        return (raw * m).sum() / m.sum().clamp(min=1)

    imgs, texts = batch[0].to(device=device, dtype=dt), [str(t) for t in batch[1]]
    if scramble_rng is not None:
        texts = [texts[i] for i in scramble_rng.permutation(len(texts))]
    enc = parts["tok"](texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    if parts["train_text_tower"]:
        t = parts["text_enc"](**enc).last_hidden_state[:, 0, :]
    else:
        with torch.no_grad():
            t = parts["text_enc"](**enc).last_hidden_state[:, 0, :]
    return _clip_loss(parts["img_proj"](_forward_global(model, imgs)).float(),
                      parts["txt_proj"](t).float(), float(ctrl.get("clip_temp", 0.07)))


def _verify_modules(parts: dict) -> List[nn.Module]:
    return [m for m in (parts["model"], parts["mae"], parts["simclr_proj"], parts["dino"],
                        parts["head"], parts["text_enc"], parts["img_proj"], parts["txt_proj"])
            if isinstance(m, nn.Module)]


def _verify_evaluate(objective: str, modality: str, parts: dict, loader, ctrl: dict,
                     device, masks: Optional[List] = None, eval_seed: int = 20260914) -> List[float]:
    mods = _verify_modules(parts)
    was_training = [m.training for m in mods]
    for m in mods:
        m.eval()
    center = None
    if parts["dino"] is not None:
        center = parts["dino"].center.detach().clone()
    torch.manual_seed(eval_seed)
    out = []
    try:
        with torch.no_grad():
            for i, batch in enumerate(loader):
                out.append(float(_verify_batch_loss(
                    objective, modality, parts, batch, ctrl, device,
                    scramble_rng=None, mae_mask=None if masks is None else masks[i])))
    finally:
        if center is not None:
            parts["dino"].center.copy_(center)
        for m, t in zip(mods, was_training):
            m.train(t)
    return out


def _verify_feature_geometry(model: nn.Module, loader, device,
                             eval_seed: int = 20260914) -> dict:
    was_training = model.training
    model.eval()
    torch.manual_seed(eval_seed)
    feats = []
    dt = next(model.parameters()).dtype
    try:
        with torch.no_grad():
            for batch in loader:
                feats.append(_forward_global(model, batch[0].to(device=device, dtype=dt)).float().cpu())
    finally:
        model.train(was_training)
    X = torch.cat(feats).double()
    n = int(X.shape[0])
    Xn = F.normalize(X, p=2, dim=-1)
    cos = Xn @ Xn.T
    mean_cos = float((cos.sum() - cos.diagonal().sum()) / max(n * (n - 1), 1))
    var = torch.linalg.svdvals(X - X.mean(dim=0, keepdim=True)) ** 2
    return {"feat_mean_cosine": mean_cos,
            "feat_top1_var_share": float(var[0] / var.sum().clamp(min=1e-30))}


def _verify_one_objective(global_config_path: str, objective: str, modality: str,
                          n_steps: int, batch_size: int, n_eval_batches: int,
                          precision: str = "fp32") -> dict:
    if precision not in VERIFY_PRECISIONS:
        raise ValueError(f"[verify] precision {precision!r} is not one of {VERIFY_PRECISIONS}.")
    cfg  = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    mix_dir = os.path.join(ctrl["ckpts_dir"], "mixtures")
    common_csv = os.path.join(mix_dir, f"{modality}_common_train.csv")
    if not os.path.exists(common_csv):
        raise MissingInput(f"the training mixture is absent at {common_csv}; run main_build_training_mixtures first.")
    if modality == "histo":
        dataset = _NCTDataset(common_csv, cfg["histo"]["nct_crc"]["root"], objective, tfm)
    else:
        root = cfg["cxr"]["sites"]["mimic"]["image_root"]
        dataset = _CXRDataset(common_csv, root, objective, _verify_label_columns(modality),
                              tfm, report_root=root)
        dataset.assert_text_present()

    n_rows = len(dataset)
    bs = int(min(batch_size, max(2, n_rows // 8)))
    n_eval = int(max(2, min(n_eval_batches, n_rows // (4 * bs))))
    if n_rows < (n_eval + 4) * bs:
        raise MissingInput(
            f"[verify] the {modality} mixture holds {n_rows} rows, too few for {n_eval} held-out "
            f"batches of {bs} plus any training; with drop_last this would verify nothing.")
    order = np.random.RandomState(20260914).permutation(n_rows)
    eval_set  = torch.utils.data.Subset(dataset, order[:n_eval * bs].tolist())
    train_set = torch.utils.data.Subset(dataset, order[n_eval * bs:].tolist())
    eval_loader = DataLoader(eval_set, batch_size=bs, shuffle=False,
                             num_workers=0, drop_last=True)
    if len(eval_loader) != n_eval:
        raise MissingInput(f"[verify] the held-out set of {len(eval_set)} rows yields "
                           f"{len(eval_loader)} batches of {bs}, not the {n_eval} the paired "
                           f"comparison and the fixed masks are built for.")

    masks = None
    if objective == "mae":
        g = torch.Generator().manual_seed(20260914)
        ratio = float(ctrl.get("mae_mask_ratio", 0.75))
        n_patch = (224 // 16) ** 2
        n_hide = int(n_patch * ratio)
        masks = []
        for _ in range(n_eval):
            ids = torch.argsort(torch.rand(bs, n_patch, generator=g), dim=1)
            m = torch.zeros(bs, n_patch)
            m.scatter_(1, ids[:, :n_hide], 1.0)
            masks.append(m)

    lr = float(ctrl.get("lr", 1e-4))
    result = {"n_rows": n_rows, "batch_size": bs, "n_eval_batches": n_eval,
              "n_steps": n_steps, "held_out_rows": n_eval * bs}

    def _arm(scramble: bool) -> dict:
        torch.manual_seed(0)
        np.random.seed(0)
        model = _build_vit("vit_s", "dinov3", cfg).to(device)
        embed_dim = _infer_embed_dim(model, "vit_s")
        parts = _verify_attachments(objective, modality, model, embed_dim, cfg, device)
        opt = torch.optim.AdamW(list(model.parameters()) + parts["params"], lr=lr)
        baseline = _verify_evaluate(objective, modality, parts, eval_loader, ctrl, device, masks)
        loader = DataLoader(train_set, batch_size=bs, shuffle=True,
                            num_workers=0, drop_last=True)
        if len(loader) == 0:
            raise MissingInput(f"[verify] {modality}/{objective}: {len(train_set)} training rows "
                               f"yield no batch at size {bs}.")
        rng = np.random.RandomState(0)
        curve, step = [], 0
        bar = tqdm(total=n_steps, desc=f"verify {modality}/{objective}"
                                       f"{' scrambled' if scramble else ' real'}",
                   unit="step", leave=False)
        while step < n_steps:
            for batch in loader:
                if step >= n_steps:
                    break
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=precision == "bf16_autocast"):
                    loss = _verify_batch_loss(objective, modality, parts, batch, ctrl, device,
                                              scramble_rng=rng if scramble else None)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if parts["dino"] is not None:
                    parts["dino"].update_teacher()
                curve.append(float(loss))
                step += 1
                bar.update(1)
        bar.close()
        if not curve:
            raise MissingInput(f"[verify] {modality}/{objective}: the mixture yielded no batch "
                               f"at size {bs}.")
        trained = _verify_evaluate(objective, modality, parts, eval_loader, ctrl, device, masks)
        geom = _verify_feature_geometry(model, eval_loader, device)
        del parts, opt, model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {"baseline": baseline, "held_out": trained, "curve": curve, "geometry": geom}

    real = _arm(scramble=False)
    fake = _arm(scramble=True)
    drift = float(np.abs(np.asarray(real["baseline"]) - np.asarray(fake["baseline"])).max())
    if drift > 1e-4:
        raise ValueError(
            f"[verify] {modality}/{objective}: the 2 arms scored the untrained model differently "
            f"(largest difference {drift:.3e}), so they did not start from one initialization and "
            f"the comparison between them measures the seed as well as the target.")
    result["baseline"] = real["baseline"]
    result["held_out_real"] = real["held_out"]
    result["held_out_scrambled"] = fake["held_out"]
    result["curve_real"] = real["curve"]
    result["curve_scrambled"] = fake["curve"]
    result["geometry"] = real["geometry"]
    return result


def verify_objectives_on_scrambled_targets(global_config_path: str,
                                           n_steps: Optional[int] = None,
                                           precision: Optional[str] = None) -> str:
    from scipy import stats as _stats
    cfg  = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    precision = training_precision(ctrl) if precision is None else precision
    if precision not in VERIFY_PRECISIONS:
        raise ValueError(f"[verify] precision {precision!r} is not one of {VERIFY_PRECISIONS}.")
    require_bf16_gpu(precision, "verify")
    out_dir = ctrl["results_e5_dir"]
    os.makedirs(out_dir, exist_ok=True)
    fp32 = precision == "fp32"
    out_csv, _mixes, extra = _objective_gate(ctrl, precision)
    status = status_path(cfg, "verify_objectives" if fp32 else f"verify_objectives_{precision}")
    tag = "verify" if fp32 else f"verify_{precision}"
    from Inference.resume_utils import claim_unit, output_is_current, release_claim
    if output_is_current(out_csv, _mixes, owner=f"verify_objectives_{precision}", extra=extra):
        failures = _gate_failures(out_csv)
        if failures:
            raise ValueError(f"[verify] {failures} failed the verification in {precision}; the "
                             f"per-cell numbers are in {out_csv}. Do not run the matrix until this passes.")
        return out_csv
    if not claim_unit(out_dir, tag):
        return out_csv

    n_steps = int(ctrl.get("verify_steps", 300) if n_steps is None else n_steps)
    bs      = int(ctrl.get("verify_batch_size", 32))
    n_eval  = int(ctrl.get("verify_eval_batches", 20))
    min_rel = float(ctrl.get("verify_min_relative_gain", 0.01))

    from controlled.matrix import enumerate_e5_runs
    cells = sorted({(r[1], r[2]) for r in enumerate_e5_runs(global_config_path)})
    rows, failures = [], []
    for modality, objective in cells:
        heartbeat_claim(out_dir, tag)
        r = _verify_one_objective(global_config_path, objective, modality,
                                  n_steps, bs, n_eval, precision=precision)
        base = np.asarray(r["baseline"], dtype=float)
        real = np.asarray(r["held_out_real"], dtype=float)
        fake = np.asarray(r["held_out_scrambled"], dtype=float)
        d = fake - real
        mean_d = float(d.mean())
        t_stat, p_val = _stats.ttest_rel(fake, real)
        floor = min_rel * float(base.mean())
        learns = bool(mean_d > floor and float(p_val) < 0.05)
        geom = r["geometry"]
        rows.append({
            "modality": modality, "objective": objective,
            "n_steps": r["n_steps"], "batch_size": r["batch_size"],
            "held_out_rows": r["held_out_rows"],
            "loss_first_real": r["curve_real"][0], "loss_last_real": r["curve_real"][-1],
            "loss_first_scrambled": r["curve_scrambled"][0],
            "loss_last_scrambled": r["curve_scrambled"][-1],
            "held_out_baseline": float(base.mean()),
            "held_out_real": float(real.mean()),
            "held_out_scrambled": float(fake.mean()),
            "held_out_advantage": mean_d,
            "held_out_advantage_sd": float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
            "required_advantage": floor,
            "t_statistic": float(t_stat), "p_value": float(p_val),
            "feat_mean_cosine": geom["feat_mean_cosine"],
            "feat_top1_var_share": geom["feat_top1_var_share"],
            "learns_the_named_task": learns, "precision": precision})
        if not learns:
            failures.append(f"{modality}/{objective}")
    write_csv_atomic(pd.DataFrame(rows), out_csv)
    from Inference.resume_utils import record_sources
    record_sources(out_csv, _mixes, extra=extra)
    release_claim(out_dir, tag)
    append_status(status, f"objective verification ({precision}): {len(failures)} failure(s)")
    if failures:
        raise ValueError(
            f"[verify] {failures} do no better on held-out data from the real target than from a "
            f"scrambled one, so whatever they optimize is not the task they are named for. The "
            f"per-cell numbers are in {out_csv}. Do not run the matrix until this passes.")
    return out_csv


def run_one_e11_cell(global_config_path: str, run_id: str, force: bool = False):
    runs = {r[0]: r for r in enumerate_e11_runs(global_config_path)}
    if run_id not in runs:
        raise ValueError(
            f"[E11] run_id '{run_id}' is not in the configured granularity matrix. "
            f"Available ({len(runs)}): " + ", ".join(sorted(runs.keys())))
    _, form, bb, seed = runs[run_id]
    init = str((read_config(global_config_path)["Convergence"]
                .get("granularity", {}) or {}).get("init", "dinov3"))
    return train_one_cell(run_id, "taix", "supervised", bb, init, seed,
                          global_config_path, force=force, label_form=form)


def run_one_disjoint_cell(global_config_path: str, run_id: str, force: bool = False):
    runs = {r[0]: r for r in enumerate_disjoint_runs(global_config_path)}
    if run_id not in runs:
        raise ValueError(
            f"[disjoint] run_id '{run_id}' is not in the configured disjoint-pair control. "
            f"Available ({len(runs)}): " + ", ".join(sorted(runs.keys())))
    _, mod, obj, bb, init, seed = runs[run_id]
    return train_one_cell(run_id, mod, obj, bb, init, seed, global_config_path, force=force)


def run_one_e5_cell(global_config_path: str, run_id: str, force: bool = False):
    runs = {r[0]: r for r in enumerate_e5_runs(global_config_path)}
    if run_id not in runs: raise ValueError(
            f"[E5] run_id '{run_id}' is not in the configured matrix. "
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
