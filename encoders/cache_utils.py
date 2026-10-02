"""
encoders/cache_utils.py
Created on June 19, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import gc
import os
import shutil
from typing import List, Optional, Tuple

import numpy as np

from config.serde import read_config
from Inference.resume_utils import MissingInput


def embedding_cache_path(cfg: dict, encoder: str, pool: str) -> str:
    return os.path.join(cfg["embeddings"]["output_dir"], encoder, f"{pool}.npz")


HARMONIZED_PIPELINE = "v2_native_grid"


def harmonized_cache_name(pool_name: str, cfg: dict) -> str:
    return f"{pool_name}{cfg['embeddings']['harmonized_preprocessing']['suffix']}"


def npz_member_shape(npz_path: str, member: str) -> Optional[tuple]:
    import zipfile
    from numpy.lib import format as npformat
    try:
        with zipfile.ZipFile(npz_path) as zf:
            name = member if member in zf.namelist() else member + ".npy"
            with zf.open(name) as f:
                version = npformat.read_magic(f)
                if version == (1, 0):
                    return npformat.read_array_header_1_0(f)[0]
                if version == (2, 0):
                    return npformat.read_array_header_2_0(f)[0]
    except Exception:
        return None
    return None


def cache_is_current(npz_path: str, expected_n: int, expected_dim: int = 0,
                     label: str = "", expected_ids: Optional[List[str]] = None) -> bool:
    if not os.path.exists(npz_path):
        return False
    try:
        shape = npz_member_shape(npz_path, "embeddings")
        if shape is None:
            shape = np.load(npz_path, allow_pickle=True)["embeddings"].shape
        if len(shape) != 2 or int(shape[0]) != expected_n:
            return False
        if expected_dim and int(shape[1]) != expected_dim:
            return False
        if expected_ids is not None:
            cached = np.load(npz_path, allow_pickle=True)["case_ids"].astype(str)
            want = np.asarray([str(c) for c in expected_ids], dtype=str)
            if cached.shape[0] != want.shape[0] or not bool((cached == want).all()):
                n_diff = int((cached != want).sum()) if cached.shape == want.shape else -1
                return False
        return True
    except Exception:
        return False


def assert_declared_width(encoder_name: str, declared: int, got: int) -> None:
    if declared and int(got) != int(declared):
        raise RuntimeError(
            f"[extract] '{encoder_name}': the encoder returned {int(got)}-dim vectors while the panel "
            f"declares {int(declared)}. Read the width from the checkpoint's own config.json, correct "
            f"config.yaml, and check the tower neither loaded from random init nor lost its projection.")


def load_embedding_cache(cfg: dict, encoder: str, pool: str) -> Tuple[np.ndarray, np.ndarray]:
    path = embedding_cache_path(cfg, encoder, pool)
    if not os.path.exists(path):
        raise MissingInput(f"no embedding cache for ({encoder}, {pool}) at {path}; "
                           f"run main_extract_image_embeddings for that encoder before this stage.")
    d = np.load(path, allow_pickle=True)
    if "embeddings" not in d or "case_ids" not in d:
        raise MissingInput(f"the cache at {path} carries {sorted(d.files)} and not "
                           f"embeddings plus case_ids; re-extract that encoder.")
    return (d["embeddings"].astype(np.float32, copy=False), d["case_ids"].astype(str))


def finite_ids_cache_dir(cfg: dict) -> str:
    return os.path.join(cfg["alignment"]["results_base_dir"], "results_e14_shared", "encoder_cache")


def finite_case_ids(cfg: dict, enc: str, pool: str) -> np.ndarray:
    from Inference.resume_utils import (check_build_params, ensure_dir, fingerprint_file,
                                        write_build_params, write_npz_atomic)
    cache_dir = finite_ids_cache_dir(cfg)
    path = os.path.join(cache_dir, f"finite__{enc}__{pool}.npz")
    expected = {"cache": fingerprint_file(embedding_cache_path(cfg, enc, pool))}
    if os.path.exists(path) and check_build_params(path, expected, owner="finite_ids"):
        return np.load(path, allow_pickle=True)["case_ids"].astype(str)
    emb, ids = load_embedding_cache(cfg, enc, pool)
    ok = np.asarray(ids).astype(str)[np.isfinite(emb).all(axis=1)]
    ensure_dir(cache_dir)
    write_npz_atomic(path, case_ids=np.array(ok, dtype=object))
    write_build_params(path, expected)
    return ok


def _repo_folder_name(repo_id: str) -> str:
    try:
        from huggingface_hub import repo_folder_name
        return repo_folder_name(repo_id=repo_id, repo_type="model")
    except Exception:
        return "models--" + repo_id.replace("/", "--")


def _release_memory():
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

    if hf_id is None or os.path.isdir(hf_id):
        return

    cache_dir = emb.get("hf_cache_dir", "")
    if not cache_dir:
        cache_dir = os.environ.get(
            "HF_HUB_CACHE",
            os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub"),
        )

    folder = os.path.join(cache_dir, _repo_folder_name(hf_id))

    _release_memory()

    if not os.path.isdir(folder):
        return

    try:
        shutil.rmtree(folder)
    except Exception as e:
        pass


def purge_encoders_after_stage(encoder_names, panel: dict, cfg_path: str) -> None:
    for name in encoder_names:
        spec = panel.get(name, {})
        hf_id = spec.get("hf_id")
        if not hf_id:
            continue
        purge_model_cache(hf_id, cfg_path)
