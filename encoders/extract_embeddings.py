"""
encoders/extract_embeddings.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.modality_embedding_loaders import (
    LOADER_REGISTRY, embedding_collate_fn,
)
from encoders.image_encoders import extract_image_embeddings, list_encoder_names
from encoders.text_encoders import extract_text_embeddings, list_text_encoder_names

import warnings
warnings.filterwarnings("ignore")


def _npz_path(output_dir: str, encoder_name: str, pool_name: str) -> str:
    return os.path.join(output_dir, encoder_name, f"{pool_name}.npz")


def _is_done(npz_path: str, expected_n: int) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        d = np.load(npz_path, allow_pickle=True)
        return int(d["embeddings"].shape[0]) == expected_n
    except Exception:
        return False


def _save_npz(npz_path: str, embeddings: np.ndarray, case_ids: List[str]) -> None:
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        embeddings=embeddings.astype(np.float32),
        case_ids=np.array(case_ids, dtype=object),
    )



def extract_pool(
    encoder_name: str,
    pool_name: str,
    pool_cfg: dict,
    output_dir: str,
    cfg_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
) -> str:
    manifest_csv = pool_cfg["manifest"]
    modality     = pool_cfg["modality"]
    out_path     = _npz_path(output_dir, encoder_name, pool_name)

    manifest = read_csv_defensively(manifest_csv)
    n_total  = len(manifest)
    if _is_done(out_path, n_total):
        print(f"[extract] {encoder_name}/{pool_name}: already done ({n_total} rows). Skip.")
        return out_path


    loader_cls = LOADER_REGISTRY.get(modality)
    if loader_cls is None:
        raise KeyError(f"[extract] No loader for modality '{modality}'")

    dataset = loader_cls(cfg_path=cfg_path, manifest_csv=manifest_csv, resolution=224)
    loader  = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=embedding_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    try:
        from encoders.image_encoders import _load_encoder
        _load_encoder(encoder_name, cfg_path, device)
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            f"[extract] '{encoder_name}': required package missing ({e}). "
            f"Install it (see requirements.txt) and re-run. NOT writing any output."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"[extract] '{encoder_name}': failed to load the encoder ({e}). "
            f"Check the hf_id / gated-model access / local path in config. "
            f"NOT writing any output."
        ) from e

    spec = read_config(cfg_path)["Convergence"]["encoder_panel"]["image"].get(
        encoder_name, {})
    emb_dim = int(spec.get("dim", 0)) or None

    embeddings: Optional[np.ndarray] = None
    all_case_ids: List[str] = []
    n_filled = 0
    n_failed_batches = 0

    for batch in tqdm(loader, desc=f"{encoder_name[:20]}|{pool_name}", unit="batch"):
        images   = batch["images"]
        case_ids = batch["case_ids"]

        try:
            emb = extract_image_embeddings(encoder_name, images, cfg_path, device)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"[extract] '{encoder_name}': required package missing ({e}). "
                f"Install it and re-run. NOT writing partial output."
            ) from e
        except Exception as e:
            import traceback as _tb
            print(f"[extract] per-batch error ({encoder_name}/{pool_name}): {e}. "
                  f"Filling NaN for this batch.")
            if n_failed_batches == 0:   # print the full traceback once
                _tb.print_exc()
            n_failed_batches += 1
            d   = emb_dim or (embeddings.shape[1] if embeddings is not None else 1)
            emb = np.full((len(images), d), np.nan, dtype=np.float32)

        emb = np.ascontiguousarray(emb, dtype=np.float32)
        if embeddings is None:
            # Now we know the true embedding dim; allocate the full array once.
            d = emb.shape[1]
            embeddings = np.empty((n_total, d), dtype=np.float32)
        b = emb.shape[0]
        embeddings[n_filled:n_filled + b] = emb
        n_filled += b
        all_case_ids.extend(case_ids)

    if embeddings is None or n_filled == 0:
        raise RuntimeError(
            f"[extract] '{encoder_name}/{pool_name}': no batches produced output."
        )
    # Trim in case the manifest had fewer usable rows than n_total
    if n_filled != n_total:
        embeddings = embeddings[:n_filled]

    # Refuse to mark a run "done" if it is entirely NaN (silent total failure).
    nan_frac = float(np.isnan(embeddings).all(axis=1).mean())
    if nan_frac == 1.0:
        raise RuntimeError(
            f"[extract] '{encoder_name}/{pool_name}': ALL embeddings are NaN. "
            f"Not writing output. Fix the encoder setup and re-run."
        )
    if n_failed_batches > 0:
        print(f"[extract] WARNING: {n_failed_batches} batch(es) failed and were "
              f"NaN-filled ({nan_frac:.1%} of cases all-NaN).")

    _save_npz(out_path, embeddings, all_case_ids)
    print(f"[extract] Saved {embeddings.shape} -> {out_path}")
    return out_path


def main_extract_image_embeddings(
    global_config_path: str,
    encoder_names: Optional[List[str]] = None,
    pool_names: Optional[List[str]] = None,
    device: str = "cuda",
) -> None:
    """Extract and cache image embeddings for all (encoder, pool) pairs.

    Args:
        encoder_names: subset of image encoder keys to run. None = all.
        pool_names:    subset of pool keys to run. None = all.
        device:        torch device string.
    """
    cfg   = read_config(global_config_path)["Convergence"]
    emb_cfg = cfg["embeddings"]
    output_dir  = emb_cfg["output_dir"]
    batch_size  = int(emb_cfg.get("batch_size", 64))
    num_workers = int(emb_cfg.get("num_workers", 4))
    pools = emb_cfg["pools"]

    all_encoders = encoder_names or list_encoder_names(global_config_path)
    all_pools    = pool_names    or list(pools.keys())

    image_panel = cfg["encoder_panel"]["image"]

    for encoder_name in all_encoders:
        # Honor an explicit skip flag (e.g. models with incompatible custom code).
        if image_panel.get(encoder_name, {}).get("skip", False):
            print(f"[extract] '{encoder_name}' has skip:true in config; skipping.")
            continue
        for pool_name in all_pools:
            if pool_name not in pools:
                print(f"[extract] Pool '{pool_name}' not in config; skipping.")
                continue
            enc_bs = int(image_panel.get(encoder_name, {}).get("batch_size", batch_size))
            if enc_bs != batch_size:
                print(f"[extract] {encoder_name}: using per-encoder batch_size={enc_bs}")
            try:
                extract_pool(
                    encoder_name, pool_name, pools[pool_name],
                    output_dir, global_config_path,
                    enc_bs, num_workers, device,
                )
            except Exception as e:
                print(f"[extract] FAILED {encoder_name}/{pool_name}: {e}")
            # Clear GPU cache between encoders
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        # This encoder has finished all its pools; free its disk weights cache so
        # the next model has room (controlled by embeddings.purge_hf_cache_after_use).
        try:
            from encoders.cache_utils import purge_encoders_after_stage
            purge_encoders_after_stage([encoder_name], image_panel, global_config_path)
        except Exception as e:
            print(f"[extract] cache purge skipped for {encoder_name}: {e}")


def main_extract_text_embeddings(
    global_config_path: str,
    encoder_names: Optional[List[str]] = None,
    device: str = "cuda",
) -> None:
    """Extract and cache text embeddings for the paired-report manifest (E2)."""
    cfg       = read_config(global_config_path)["Convergence"]
    emb_cfg   = cfg["embeddings"]
    output_dir = emb_cfg["text_output_dir"]
    batch_size = int(emb_cfg.get("batch_size", 64))
    paired_csv = cfg["cxr"]["paired_reports_csv"]
    mimic_root = cfg["cxr"]["sites"]["mimic"]["image_root"]

    if not os.path.exists(paired_csv):
        print(f"[extract_text] Paired reports manifest not found: {paired_csv}. "
              f"Run build_cxr_paired_reports first.")
        return

    df = read_csv_defensively(paired_csv)

    # Build text list: read files for MIMIC, use inline for CheXpert
    texts: List[str] = []
    for _, row in df.iterrows():
        rpath = row.get("report_path")
        rtext = row.get("report_text")
        if str(rpath) not in ("nan", "", "None") and rpath and os.path.exists(str(rpath)):
            try:
                with open(str(rpath), "r", encoding="utf-8", errors="replace") as f:
                    texts.append(f.read().strip())
            except Exception:
                texts.append("")
        elif str(rtext) not in ("nan", "", "None") and rtext:
            texts.append(str(rtext).strip())
        else:
            texts.append("")

    case_ids = df["case_id"].astype(str).tolist()
    pool_name = "paired_reports"

    encoders = encoder_names or list_text_encoder_names(global_config_path)
    for enc_name in encoders:
        out_path = _npz_path(output_dir, enc_name, pool_name)
        if _is_done(out_path, len(texts)):
            print(f"[extract_text] {enc_name}/{pool_name}: already done. Skip.")
            continue
        # Resolve the spec once (dim for NaN-fill); never re-read config per batch.
        text_panel = read_config(global_config_path)["Convergence"]["encoder_panel"]["text"]
        spec = text_panel.get(enc_name, {})
        enc_dim = int(spec.get("dim", 768))
        enc_bs = int(spec.get("batch_size", batch_size))
        if enc_bs != batch_size:
            print(f"[extract_text] {enc_name}: using per-encoder batch_size={enc_bs}")

        try:
            _ = extract_text_embeddings(enc_name, texts[:1], global_config_path, device)
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"[extract_text] '{enc_name}': required package missing ({e}). "
                f"Install it and re-run. NOT writing output."
            ) from e
        except Exception as e:
            raise RuntimeError(
                f"[extract_text] '{enc_name}': failed to load/run the encoder "
                f"({e}). Check the hf_id / gated access / loader. NOT writing output."
            ) from e

        all_emb: List[np.ndarray] = []
        n_failed = 0
        seen_dim = None   # actual embedding width from the first successful batch
        for i in tqdm(range(0, len(texts), enc_bs), unit="batch"):
            batch = texts[i:i + enc_bs]
            try:
                emb = extract_text_embeddings(enc_name, batch, global_config_path, device)
                if seen_dim is None and emb is not None and emb.ndim == 2:
                    seen_dim = emb.shape[1]   # ground truth width for NaN-fill
            except Exception as e:
                print(f"[extract_text] per-batch error ({enc_name}): {e}. "
                      f"Filling NaN for this batch.")
                if n_failed == 0:
                    import traceback as _tb; _tb.print_exc()
                n_failed += 1
                fill_dim = seen_dim if seen_dim is not None else enc_dim
                emb = np.full((len(batch), fill_dim), np.nan, dtype=np.float32)
            all_emb.append(emb)
        # If the first batch(es) were NaN-filled at enc_dim but later batches have a
        # different true width, repair the early fills to match before concatenating.
        if seen_dim is not None:
            for j, e in enumerate(all_emb):
                if e.ndim == 2 and e.shape[1] != seen_dim:
                    all_emb[j] = np.full((e.shape[0], seen_dim), np.nan, dtype=np.float32)
        embeddings = np.concatenate(all_emb, axis=0)
        if bool(np.isnan(embeddings).all()):
            raise RuntimeError(
                f"[extract_text] '{enc_name}': ALL embeddings are NaN. "
                f"Not writing output. Fix the encoder setup and re-run."
            )
        if n_failed:
            print(f"[extract_text] WARNING: {n_failed} batch(es) NaN-filled.")
        _save_npz(out_path, embeddings, case_ids)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            from encoders.cache_utils import purge_encoders_after_stage
            purge_encoders_after_stage([enc_name], text_panel, global_config_path)
        except Exception as e:
            print(f"[extract_text] cache purge skipped for {enc_name}: {e}")
