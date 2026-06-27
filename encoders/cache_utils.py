"""
encoders/cache_utils.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import gc
import os
import shutil
from typing import Optional

from config.serde import read_config


def _repo_folder_name(repo_id: str) -> str:
    """Hub cache folder name for a repo id. Prefer the official helper; fall back
    to the documented convention models--<org>--<name>."""
    try:
        from huggingface_hub import repo_folder_name
        return repo_folder_name(repo_id=repo_id, repo_type="model")
    except Exception:
        return "models--" + repo_id.replace("/", "--")


def _release_memory():
    """Drop Python refs and free the CUDA allocator so memory-mapped weights are
    released before the folder is deleted."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def purge_model_cache(hf_id: str, cfg_path: str, enabled: Optional[bool] = None) -> None:
    cfg = read_config(cfg_path)["Convergence"]
    emb = cfg.get("embeddings", {})
    if enabled is None:
        enabled = bool(emb.get("purge_hf_cache_after_use", False))
    if not enabled:
        return

    # Never delete a model that was loaded from a local path rather than the hub.
    if hf_id is None or os.path.isdir(hf_id):
        print(f"[purge] '{hf_id}' looks like a local path or is empty; not purging.")
        return

    cache_dir = emb.get("hf_cache_dir", "")
    if not cache_dir:
        # Fall back to the standard env/default location.
        cache_dir = os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        )

    folder = os.path.join(cache_dir, _repo_folder_name(hf_id))

    _release_memory()

    if not os.path.isdir(folder):
        print(f"[purge] ENABLED but found no cache folder for '{hf_id}' at {folder} "
              f"(check embeddings.hf_cache_dir if this is unexpected).")
        return

    try:
        shutil.rmtree(folder)
        print(f"[purge] removed HF weights cache for '{hf_id}' at {folder}")
    except Exception as e:
        print(f"[purge] could NOT fully remove '{hf_id}' cache at {folder}: "
              f"{type(e).__name__}: {e} (files may still be open).")


def purge_encoders_after_stage(encoder_names, panel: dict, cfg_path: str) -> None:
    """Purge the hub cache for each encoder in encoder_names, looking up its hf_id
    in the given panel dict (image or text). Encoders without an hf_id (e.g. timm
    or local checkpoints) are skipped."""
    for name in encoder_names:
        spec = panel.get(name, {})
        hf_id = spec.get("hf_id")
        if not hf_id:
            continue
        purge_model_cache(hf_id, cfg_path)
