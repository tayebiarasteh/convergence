"""
encoders/extract_embeddings.py
Created on June 21, 2026

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
from encoders.cache_utils import (cache_is_current, harmonized_cache_name,
                                  assert_declared_width)
from encoders.image_encoders import extract_image_embeddings, list_encoder_names
from encoders.panel import resolve_hf_token
from encoders.text_encoders import extract_text_embeddings, list_text_encoder_names

import warnings
from Inference.resume_utils import (append_status, status_path, MissingInput,
                                    print_projected_peak, write_build_params)
warnings.filterwarnings("ignore")


def _npz_path(output_dir: str, encoder_name: str, pool_name: str) -> str:
    return os.path.join(output_dir, encoder_name, f"{pool_name}.npz")


def _report_parameter_count(encoder_name: str, model, spec: dict) -> Optional[float]:
    try:
        got = sum(p.numel() for p in model.parameters()) / 1e6
    except Exception:
        return None
    declared = float(spec.get("params_m", 0) or 0)
    line = f"[extract] {encoder_name}: {got:.0f}M parameters loaded"
    return round(float(got), 1)


def _save_npz(npz_path: str, embeddings: np.ndarray, case_ids: List[str]) -> None:
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    tmp = npz_path + ".tmp.npz"
    np.savez(
        tmp,
        embeddings=embeddings.astype(np.float32, copy=False),
        case_ids=np.array(case_ids, dtype=object),
    )
    os.replace(tmp, npz_path)


def _extract_with_oom_retry(encoder_name, images, cfg_path, device, depth: int = 0,
                            harmonized: bool = False):
    import gc
    oom = False
    try:
        return extract_image_embeddings(encoder_name, images, cfg_path, device,
                                        harmonized=harmonized)
    except torch.cuda.OutOfMemoryError:
        oom = True
    finally:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not oom:
        raise RuntimeError("unreachable")
    if len(images) == 1:
        free, total = (torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0))
        raise torch.cuda.OutOfMemoryError(
            f"[extract] {encoder_name}: out of memory at a batch of ONE image, so no batch "
            f"setting can fix it. The card reports {free / 1024 ** 3:.1f} GiB free of "
            f"{total / 1024 ** 3:.1f} GiB. This is a statement about the model's persistent "
            f"footprint: quantize it, shard it across GPUs, or give it a larger card.")
    half = len(images) // 2
    first = _extract_with_oom_retry(encoder_name, images[:half], cfg_path, device,
                                    depth + 1, harmonized=harmonized)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    second = _extract_with_oom_retry(encoder_name, images[half:], cfg_path, device,
                                     depth + 1, harmonized=harmonized)
    return np.concatenate([first, second], axis=0)


def resolved_revision(hf_id: Optional[str], token: Optional[str]) -> Optional[str]:
    if not hf_id or os.path.isdir(str(hf_id)):
        return None
    try:
        from huggingface_hub import model_info
        return str(model_info(str(hf_id), token=token).sha)
    except Exception:
        return None


def _load_failure_message(e: Exception) -> str:
    msg = str(e)
    low = msg.lower()
    if "cuda" in low or "nvidia driver" in low or "no kernel image" in low:
        built, count = "unknown", 0
        try:
            import torch
            built = torch.version.cuda or "unknown"
            count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            pass
        return (f"the GPU is unusable on this node, so nothing was loaded ({msg}). This torch is "
                f"built for CUDA {built} and sees {count} usable device(s). Run on a node whose "
                f"driver is at least as new, naming this one in the sbatch script's exclude list, "
                f"or install a torch built for the node's CUDA.")
    return (f"failed to load the encoder ({msg}). "
            f"Check the hf_id / gated-model access / local path in config.")


def extract_pool(
    encoder_name: str,
    pool_name: str,
    pool_cfg: dict,
    output_dir: str,
    cfg_path: str,
    batch_size: int,
    num_workers: int,
    device: str,
    harmonized: bool = False,
    max_cases: int = 0,
) -> str:
    manifest_csv = pool_cfg["manifest"]
    modality     = pool_cfg["modality"]
    cfg_all      = read_config(cfg_path)["Convergence"]
    cache_name   = (harmonized_cache_name(pool_name, cfg_all) if harmonized else pool_name)
    out_path     = _npz_path(output_dir, encoder_name, cache_name)

    if not os.path.exists(manifest_csv):
        raise MissingInput(
            f"[extract] the {pool_name} manifest is absent at {manifest_csv}; build the pool "
            f"(main_build_cxr_pool, main_build_histo_pool, main_build_taix_pool, or main_build_rexgradient_pool) and then run main_build_splits "
            f"before extracting embeddings on it.")
    manifest = read_csv_defensively(manifest_csv)
    if max_cases and len(manifest) > max_cases:
        manifest = manifest.sort_values("case_id").head(int(max_cases)).reset_index(drop=True)
        capped_csv = os.path.join(os.path.dirname(out_path), f"_{cache_name}_manifest.csv")
        os.makedirs(os.path.dirname(capped_csv), exist_ok=True)
        manifest.to_csv(capped_csv, index=False)
        manifest_csv = capped_csv
    n_total  = len(manifest)
    if n_total == 0:
        raise MissingInput(f"[extract] the {pool_name} manifest at {manifest_csv} holds no rows.")
    declared_dim = int(cfg_all["encoder_panel"]["image"].get(encoder_name, {}).get("dim", 0) or 0)
    manifest_ids = manifest["case_id"].astype(str).tolist()
    if cache_is_current(out_path, n_total, declared_dim, f"{encoder_name}/{cache_name}", manifest_ids):
        if not harmonized:
            return out_path
        from encoders.image_encoders import HARMONIZED_PIPELINE
        from Inference.resume_utils import read_build_params
        rec = read_build_params(out_path) or {}
        if rec.get("pipeline") == HARMONIZED_PIPELINE:
            return out_path
    from Inference.resume_utils import claim_unit, release_claim
    if not claim_unit(os.path.join(output_dir, encoder_name), f"x__{cache_name}"):
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

    measured_params_m: Optional[float] = None
    try:
        from encoders.image_encoders import _load_encoder
        _model, _, _ = _load_encoder(encoder_name, cfg_path, device)
        measured_params_m = _report_parameter_count(
            encoder_name, _model, cfg_all["encoder_panel"]["image"].get(encoder_name, {}))
        del _model
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            f"[extract] '{encoder_name}': required package missing ({e}). "
            f"Install it and re-run. NOT writing any output."
        ) from e
    except Exception as e:
        raise RuntimeError(
            f"[extract] '{encoder_name}': {_load_failure_message(e)} "
            f"NOT writing any output."
        ) from e

    spec = read_config(cfg_path)["Convergence"]["encoder_panel"]["image"].get(
        encoder_name, {})
    emb_dim = int(spec.get("dim", 0)) or None

    if emb_dim:
        print_projected_peak("extract",
                             {"embedding block": n_total * emb_dim * 4 / 1024 ** 3},
                             note=f"{n_total} cases x {emb_dim} dims, float32.")
    embeddings: Optional[np.ndarray] = None
    all_case_ids: List[str] = []
    n_filled = 0
    n_failed_batches = 0
    width_checked = False

    for batch in tqdm(loader, desc=f"{encoder_name[:20]}|{cache_name}", unit="batch"):
        images   = batch["images"]
        case_ids = batch["case_ids"]
        from_model = False

        try:
            emb = _extract_with_oom_retry(encoder_name, images, cfg_path, device,
                                          harmonized=harmonized)
            from_model = True
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"[extract] '{encoder_name}': required package missing ({e}). "
                f"Install it and re-run. NOT writing partial output."
            ) from e
        except Exception as e:
            import traceback as _tb
            if n_filled == 0:
                raise RuntimeError(
                    f"[extract] '{encoder_name}/{cache_name}': the FIRST batch failed "
                    f"({type(e).__name__}: {e}). A failure on the first batch is the code path "
                    f"and not an image, so nothing is written.") from e
            if n_failed_batches == 0:
                _tb.print_exc()
            n_failed_batches += 1
            d   = emb_dim or (embeddings.shape[1] if embeddings is not None else 1)
            emb = np.full((len(images), d), np.nan, dtype=np.float32)

        if from_model and not width_checked:
            assert_declared_width(encoder_name, emb_dim or 0, emb.shape[1])
            width_checked = True

        emb = np.ascontiguousarray(emb, dtype=np.float32)
        if embeddings is None:
            d = emb.shape[1]
            embeddings = np.empty((n_total, d), dtype=np.float32)
        b = emb.shape[0]
        embeddings[n_filled:n_filled + b] = emb
        n_filled += b
        all_case_ids.extend(case_ids)

    if embeddings is None or n_filled == 0:
        raise RuntimeError(
            f"[extract] '{encoder_name}/{cache_name}': no batches produced output."
        )
    if n_filled != n_total:
        embeddings = embeddings[:n_filled]

    nan_frac = float(np.isnan(embeddings).all(axis=1).mean())
    if nan_frac == 1.0:
        raise RuntimeError(
            f"[extract] '{encoder_name}/{cache_name}': ALL embeddings are NaN. "
            f"Not writing output. Fix the encoder setup and re-run."
        )
    if n_failed_batches > 0:
        if nan_frac > 0.01:
            raise RuntimeError(
                f"[extract] '{encoder_name}/{cache_name}': {nan_frac:.1%} of cases are all-NaN "
                f"after {n_failed_batches} failed batches, above the 1% a corrupt image or two "
                f"could explain. Not writing output.")

    _save_npz(out_path, embeddings, all_case_ids)
    release_claim(os.path.join(output_dir, encoder_name), f"x__{cache_name}")
    write_build_params(out_path, {
        "encoder": encoder_name, "pool": pool_name, "preprocessing":
            "harmonized" if harmonized else "native",
        "pipeline": (__import__("encoders.image_encoders", fromlist=["HARMONIZED_PIPELINE"]).HARMONIZED_PIPELINE
                     if harmonized else "native"),
        "n_rows": int(embeddings.shape[0]), "dim": int(embeddings.shape[1]),
        "hf_id": spec.get("hf_id"),
        "params_m_measured": measured_params_m,
        "hf_revision": resolved_revision(spec.get("hf_id"),
                                         resolve_hf_token(read_config(cfg_path))),
    })
    return out_path


def main_extract_image_embeddings(
    global_config_path: str,
    encoder_names: Optional[List[str]] = None,
    pool_names: Optional[List[str]] = None,
    device: str = "cuda",
) -> None:
    cfg   = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "image_embeddings")
    target_resolution = int(cfg["target_resolution"])
    emb_cfg = cfg["embeddings"]
    output_dir  = emb_cfg["output_dir"]
    batch_size  = int(emb_cfg.get("batch_size", 64))
    num_workers = int(emb_cfg.get("num_workers", 4))
    pools = emb_cfg["pools"]

    all_encoders = encoder_names or list_encoder_names(
        global_config_path, roles=sorted({r for spec in cfg["encoder_panel"]["image"].values() for r in spec.get("roles", [])}),
        require_token=True)
    all_pools    = pool_names    or list(pools.keys())

    image_panel = cfg["encoder_panel"]["image"]
    from encoders.panel import requires_access_token
    token = str(cfg.get("hf_token", "")).strip()
    gated = [n for n in all_encoders
             if requires_access_token(image_panel.get(n, {})) and not token.startswith("hf_")]
    if gated:
        raise RuntimeError(f"these encoders are gated and Convergence.hf_token is not set: "
                           f"{sorted(gated)}. Accept their terms and paste the token into "
                           f"config.yaml before running any extraction stage.")
    harm = emb_cfg["harmonized_preprocessing"]
    harm_pools = set(cfg["alignment"].get("e10_harmonized_pools", []))

    for encoder_name in all_encoders:
        if image_panel.get(encoder_name, {}).get("skip", False):
            continue
        for pool_name in all_pools:
            if pool_name not in pools:
                continue
            enc_bs = int(image_panel.get(encoder_name, {}).get("batch_size", batch_size))
            extract_pool(encoder_name, pool_name, pools[pool_name], output_dir,
                         global_config_path, enc_bs, num_workers, device)
            if harm.get("enabled") and pool_name in harm_pools:
                extract_pool(encoder_name, pool_name, pools[pool_name], output_dir,
                             global_config_path, enc_bs, num_workers, device,
                             harmonized=True, max_cases=int(harm.get("max_cases", 0)))
            if torch.cuda.is_available():
                from encoders.image_encoders import _load_encoder
                try:
                    _load_encoder.cache_clear()
                except AttributeError:
                    pass
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        try:
            from encoders.cache_utils import purge_encoders_after_stage
            purge_encoders_after_stage([encoder_name], image_panel, global_config_path)
        except Exception as e:
            pass
        append_status(status, f"image embeddings finished for {encoder_name}")


def _pool_texts(cfg: dict, pool_name: str):
    if pool_name == "cxr_pool":
        paired_csv = cfg["cxr"]["paired_reports_csv"]
        if not os.path.exists(paired_csv):
            raise MissingInput(f"the paired-reports manifest is absent at {paired_csv}; "
                               f"run main_build_cxr_paired_reports before main_extract_text_embeddings.")
        df = read_csv_defensively(paired_csv)
    else:
        blk = cfg.get(pool_name.replace("_pool", ""), {}) or {}
        man = blk.get("pool_manifest_csv", "")
        if not man or not os.path.exists(man):
            raise MissingInput(f"the {pool_name} manifest is absent; build its pool before main_extract_text_embeddings.")
        df = read_csv_defensively(man)
        if "report_text" not in df.columns:
            raise MissingInput(f"the {pool_name} manifest carries no report_text column, so it "
                               f"contributes nothing to the image-to-text analysis.")
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
    keep = [i for i, t in enumerate(texts) if t]
    n_empty = len(texts) - len(keep)
    case_ids = df["case_id"].astype(str).tolist()
    return [case_ids[i] for i in keep], [texts[i] for i in keep]


def main_extract_text_embeddings(
    global_config_path: str,
    encoder_names: Optional[List[str]] = None,
    device: str = "cuda",
) -> None:
    cfg       = read_config(global_config_path)["Convergence"]
    emb_cfg   = cfg["embeddings"]
    output_dir = emb_cfg["text_output_dir"]
    batch_size = int(emb_cfg.get("batch_size", 64))
    e2_pools = list(cfg["alignment"].get("e2_pools", ["cxr_pool"]))

    encoders = encoder_names or list_text_encoder_names(global_config_path)
    units = [(enc, pool) for enc in encoders for pool in e2_pools]
    text_panel = cfg["encoder_panel"]["text"]
    pool_cache: Dict[str, tuple] = {}
    for enc_name, pool_name in units:
        if pool_name not in pool_cache:
            pool_cache[pool_name] = _pool_texts(cfg, pool_name)
        case_ids, texts = pool_cache[pool_name]
        spec = text_panel.get(enc_name, {})
        enc_dim = int(spec.get("dim", 768))
        out_path = _npz_path(output_dir, enc_name, pool_name)
        if cache_is_current(out_path, len(texts), enc_dim, f"{enc_name}/{pool_name}", case_ids):
            continue
        from Inference.resume_utils import claim_unit, heartbeat_claim, release_claim
        if not claim_unit(os.path.join(output_dir, enc_name), f"tx__{pool_name}"):
            continue
        enc_bs = int(spec.get("batch_size", batch_size))

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

        embeddings = None
        pending: List[tuple] = []
        n_failed = 0
        seen_dim = None
        row = 0
        for i in tqdm(range(0, len(texts), enc_bs), unit="batch"):
            if (i // enc_bs) % 200 == 0:
                heartbeat_claim(os.path.join(output_dir, enc_name), f"tx__{pool_name}")
            batch = texts[i:i + enc_bs]
            emb = None
            try:
                emb = extract_text_embeddings(enc_name, batch, global_config_path, device)
                if seen_dim is None and emb is not None and emb.ndim == 2:
                    seen_dim = emb.shape[1]
            except Exception as e:
                if n_failed == 0:
                    import traceback as _tb; _tb.print_exc()
                n_failed += 1
            if embeddings is None and seen_dim is not None:
                assert_declared_width(enc_name, enc_dim, seen_dim)
                embeddings = np.empty((len(texts), seen_dim), dtype=np.float32)
                for off, n_rows in pending:
                    embeddings[off:off + n_rows] = np.nan
                pending = []
            if embeddings is None:
                pending.append((row, len(batch)))
            elif emb is None:
                embeddings[row:row + len(batch)] = np.nan
            else:
                embeddings[row:row + len(batch)] = emb.astype(np.float32, copy=False)
            row += len(batch)
        if embeddings is None:
            raise RuntimeError(
                f"[extract_text] '{enc_name}': no batch succeeded, so the embedding width is "
                f"unknown. Not writing output.")
        if bool(np.isnan(embeddings).all()):
            raise RuntimeError(
                f"[extract_text] '{enc_name}': ALL embeddings are NaN. "
                f"Not writing output. Fix the encoder setup and re-run."
            )
        _save_npz(out_path, embeddings, case_ids)
        release_claim(os.path.join(output_dir, enc_name), f"tx__{pool_name}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if pool_name == e2_pools[-1]:
            try:
                from encoders.cache_utils import purge_encoders_after_stage
                purge_encoders_after_stage([enc_name], text_panel, global_config_path)
            except Exception as e:
                pass
