"""
encoders/image_encoders.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from functools import lru_cache
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")



def _get_encoder_spec(model_name: str, cfg: dict) -> dict:
    panel = cfg["Convergence"]["encoder_panel"]["image"]
    if model_name not in panel:
        raise KeyError(
            f"[image_encoders] '{model_name}' not in encoder_panel.image. "
            f"Available: {sorted(panel.keys())}"
        )
    return panel[model_name]


def _load_vlm_full(hf_id: str, token, device: str, dtype):
    import transformers as tf
    from transformers import AutoConfig

    device_map = None
    import inspect
    _sig = inspect.signature(tf.AutoModel.from_pretrained)
    dtype_kw = "dtype" if "dtype" in _sig.parameters else "torch_dtype"
    common = {"token": token, "trust_remote_code": True, "device_map": device_map,
              dtype_kw: dtype, "low_cpu_mem_usage": True}

    import os as _os
    is_local = _os.path.isabs(hf_id) or hf_id.startswith("./") or hf_id.startswith("../")
    if is_local and not _os.path.isdir(hf_id):
        raise RuntimeError(
            f"Local model path '{hf_id}' does not exist or is not a directory. "
            f"Fix the hf_id in config (point it at a valid local HF-format dir or "
            f"a HuggingFace repo id)."
        )
    try:
        cfg = AutoConfig.from_pretrained(
            hf_id, token=token, trust_remote_code=True,
            local_files_only=is_local,
        )
        arch_names = list(getattr(cfg, "architectures", []) or [])
    except Exception as e:
        # Config could not be read; fall through to the Auto-classes which may
        # still resolve it, but record why the declared-arch path was skipped.
        arch_names = []

    last_err = None

    for arch in arch_names:
        cls = getattr(tf, arch, None)
        if cls is not None:
            try:
                model = cls.from_pretrained(hf_id, **common)
                if _has_vision_tower(model):
                    print(f"[vlm_load] loaded via declared class '{arch}'")
                    return model
            except Exception as e:
                last_err = e

    declared_custom = arch_names and all(getattr(tf, a, None) is None for a in arch_names)
    if declared_custom:
        for cls_name in ("AutoModelForCausalLM", "AutoModelForImageTextToText",
                         "AutoModelForVision2Seq", "AutoModel"):
            cls = getattr(tf, cls_name, None)
            if cls is None:
                continue
            try:
                model = cls.from_pretrained(hf_id, **common)
            except Exception as e:
                last_err = e
                continue
            # For custom remote code, the resolved class name should match the
            # declared custom architecture; accept it if so or if a vision tower
            # is present (custom classes are trusted via the repo's own auto_map).
            resolved = type(model).__name__
            if resolved in arch_names or _has_vision_tower(model):
                print(f"[vlm_load] loaded custom remote-code class '{resolved}'")
                return model
            last_err = RuntimeError(
                f"{cls_name} resolved {hf_id} to '{resolved}', no vision tower."
            )

    for cls_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM",
                     "AutoModelForVision2Seq", "AutoModel"):
        cls = getattr(tf, cls_name, None)
        if cls is None:
            continue
        try:
            model = cls.from_pretrained(hf_id, **common)
        except Exception as e:
            last_err = e
            continue
        resolved = type(model).__name__
        if arch_names and resolved not in arch_names:
            last_err = RuntimeError(
                f"{cls_name} resolved {hf_id} to '{resolved}' but the checkpoint "
                f"declares {arch_names}. Refusing silent wrong-class load."
            )
            continue
        if _has_vision_tower(model):
            return model

    try:
        from transformers import AutoModel as _AutoModel
        vcfg = getattr(cfg, "vision_config", None)
        if vcfg is not None:
            vtower = _AutoModel.from_config(vcfg, trust_remote_code=True)
            from huggingface_hub import snapshot_download
            import os as _os, glob as _glob, torch as _torch
            from safetensors import safe_open as _safe_open
            local = snapshot_download(hf_id, token=token,
                                      allow_patterns=["*.safetensors", "*.bin",
                                                      "*.json"])
            prefix = "vision_model."
            vsd = {}
            sft_files = sorted(_glob.glob(_os.path.join(local, "*.safetensors")))
            if sft_files:
                for f in sft_files:
                    with _safe_open(f, framework="pt", device="cpu") as st:
                        for k in st.keys():
                            if k.startswith(prefix):
                                vsd[k[len(prefix):]] = st.get_tensor(k)
            else:
                for f in sorted(_glob.glob(_os.path.join(local, "*.bin"))):
                    shard = _torch.load(f, map_location="cpu")
                    for k, v in shard.items():
                        if k.startswith(prefix):
                            vsd[k[len(prefix):]] = v
                    del shard
            if vsd:
                missing, unexpected = vtower.load_state_dict(vsd, strict=False)
                print(f"[vlm_load] loaded VISION TOWER ONLY for '{hf_id}' "
                      f"(bypassed broken chat wrapper; {len(vsd)} vision tensors, "
                      f"{len(missing)} missing, {len(unexpected)} unexpected).")
                class _VisionOnly(_torch.nn.Module):
                    def __init__(self, vm): super().__init__(); self.vision_model = vm
                return _VisionOnly(vtower)
    except Exception as e:
        last_err = RuntimeError(f"vision-only fallback failed: {e}")

    raise RuntimeError(
        f"Could not load VLM '{hf_id}' with its declared architecture "
        f"{arch_names}. Last error: {last_err}. "
        f"Your transformers version may be too new/old for this checkpoint; "
        f"check the model card for the required transformers version. "
        f"If only the chat wrapper is incompatible, the vision-tower-only "
        f"fallback above may need the right weight prefix for this model."
    )


def _has_vision_tower(model) -> bool:
    """True if the model exposes any known vision-tower attribute path."""
    paths = ["visual", "model.visual", "vision_tower", "model.vision_tower",
             "vision_model", "model.vision_model", "vision_encoder",
             "model.vision_encoder"]
    for path in paths:
        node = model
        ok = True
        for part in path.split("."):
            if hasattr(node, part):
                node = getattr(node, part)
            else:
                ok = False
                break
        if ok and node is not None:
            return True
    return False


@lru_cache(maxsize=4)
def _load_encoder(model_name: str, cfg_path: str, device: str):
    """Load and cache one image encoder. Returns (model, processor, spec)."""
    cfg  = read_config(cfg_path)
    spec = _get_encoder_spec(model_name, cfg)
    enc_type = spec["type"]
    hf_id    = spec.get("hf_id")

    token = cfg["Convergence"].get("hf_token", None)
    dtype = torch.float16 if cfg["Convergence"]["embeddings"].get("fp16", True) else torch.float32

    if enc_type == "dino_cls":
        from transformers import AutoModel, AutoProcessor
        model = AutoModel.from_pretrained(hf_id, token=token, trust_remote_code=True)
        proc  = AutoProcessor.from_pretrained(hf_id, token=token, trust_remote_code=True)
        model = model.to(device=device, dtype=dtype).eval()
        return model, proc, spec

    if enc_type == "clip_pooled":
        if spec.get("loader") == "open_clip" or "BiomedCLIP" in hf_id:
            import open_clip
            oc_model, _, preprocess = open_clip.create_model_and_transforms(
                f"hf-hub:{hf_id}"
            )
            oc_model = oc_model.to(device=device, dtype=dtype).eval()
            # Stash the open_clip preprocess transform on the model for extraction
            oc_model._oc_preprocess = preprocess
            return oc_model, "open_clip", spec
        from transformers import AutoModel, AutoProcessor
        import transformers as _tf
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(hf_id, token=token, trust_remote_code=True)
        _archs = list(getattr(_cfg, "architectures", []) or [])
        # Candidate order: declared arch(es), then the standard multimodal heads.
        _candidates = _archs + ["CLIPModel", "SiglipModel", "AltCLIPModel"]
        model = None
        _errors = []
        for _name in _candidates:
            _cls = getattr(_tf, _name, None)
            if _cls is None:
                _errors.append(f"{_name}: not in transformers")
                continue
            try:
                _m = _cls.from_pretrained(hf_id, token=token, trust_remote_code=True)
            except Exception as _e:
                _errors.append(f"{_name}: {type(_e).__name__}: {_e}")
                continue
            if hasattr(_m, "get_image_features"):
                model = _m
                break
            _errors.append(f"{_name}: loaded but no get_image_features")
        if model is None:
            model = AutoModel.from_pretrained(hf_id, token=token, trust_remote_code=True)
        proc  = AutoProcessor.from_pretrained(hf_id, token=token, trust_remote_code=True)
        model = model.to(device=device, dtype=dtype).eval()
        return model, proc, spec

    if enc_type == "timm_cls":
        import timm

        if model_name in ("conch_image",):
            try:
                from conch.open_clip_custom import create_model_from_pretrained
            except (ImportError, ModuleNotFoundError) as e:
                raise RuntimeError(
                    f"CONCH requires the 'conch' package (pip install git+https://github.com/"
                    f"mahmoodlab/CONCH.git) and accepted HF terms. Import error: {e}"
                ) from e
            conch_model, _ = create_model_from_pretrained(
                "conch_ViT-B-16", checkpoint_path=f"hf_hub:{hf_id}", hf_auth_token=token,
            )
            model = conch_model.visual.to(device=device, dtype=dtype).eval()
            return model, "conch", spec

        # timm hf-hub models. Per-model kwargs needed because some checkpoints
        # carry custom init args that timm must be told about explicitly.
        timm_kwargs = {"pretrained": True, "num_classes": 0}
        extra = {
            # UNI is a ViT-L/16 WITH LayerScale (init_values) and dynamic img size.
            # Without init_values the model has no ls1/ls2 modules and the
            # checkpoint's ls1.gamma/ls2.gamma keys fail to load.
            "uni":           {"img_size": 224, "patch_size": 16, "init_values": 1e-5,
                              "num_classes": 0, "dynamic_img_size": True},
            "uni2":          {"img_size": 224, "patch_size": 14, "depth": 24,
                              "num_heads": 24, "init_values": 1e-5, "embed_dim": 1536,
                              "mlp_ratio": 2.66667 * 2, "num_classes": 0,
                              "no_embed_class": True, "reg_tokens": 8,
                              "dynamic_img_size": True},
            "virchow":       {},
            "virchow2":      {},
            "prov_gigapath": {},
        }.get(model_name, {})

        # UNI2, Virchow, Virchow2 use SwiGLU MLP + SiLU; import lazily and inject.
        if model_name in ("uni2", "virchow", "virchow2"):
            from timm.layers import SwiGLUPacked
            import torch.nn as nn
            extra["mlp_layer"] = SwiGLUPacked
            extra["act_layer"] = nn.SiLU

        timm_kwargs.update(extra)
        # Avoid the 'multiple values for num_classes' clash for GigaPath: the HF
        # timm wrapper sets num_classes itself, so create from hub via timm only.
        model = timm.create_model(f"hf-hub:{hf_id}", **timm_kwargs)
        model = model.to(device=device, dtype=dtype).eval()
        return model, "timm_raw", spec

    if enc_type == "vlm_vision":
        from transformers import AutoProcessor
        full_model = _load_vlm_full(hf_id, token, device, dtype)

        vision_attr = spec.get("vision_attr", "")
        candidates = [vision_attr] if vision_attr else []
        candidates += [
            "visual",                       # Qwen2-VL / Qwen3-VL
            "model.visual",                 # some Qwen wrappers
            "vision_tower",                 # LLaVA family
            "model.vision_tower",
            "vision_model",                 # common HF convention
            "model.vision_model",
            "vision_encoder",
            "model.vision_encoder",
        ]
        submodel = None
        for path in candidates:
            if not path:
                continue
            try:
                node = full_model
                for part in path.split("."):
                    node = getattr(node, part)
                submodel = node
                break
            except AttributeError:
                continue
        if submodel is None:
            avail = [n for n, _ in full_model.named_children()]
            raise RuntimeError(
                f"Could not locate vision tower for '{model_name}'. "
                f"Tried {candidates}. Top-level submodules: {avail}. "
                f"Set the correct 'vision_attr' in config."
            )
        proc = AutoProcessor.from_pretrained(hf_id, token=token, trust_remote_code=True)

        import gc as _gc
        vision = submodel
        for attr_name, child in list(full_model.named_children()):
            if child is vision:
                continue
            try:
                setattr(full_model, attr_name, None)
            except Exception:
                pass
        inner = getattr(full_model, "model", None)
        if inner is not None and inner is not vision:
            for attr_name, child in list(inner.named_children()):
                if child is vision:
                    continue
                try:
                    setattr(inner, attr_name, None)
                except Exception:
                    pass
        del submodel, full_model
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        vision = vision.to(device=device, dtype=dtype).eval()
        return vision, proc, spec

    if enc_type == "txrv":
        import torchxrayvision as xrv
        weights = spec.get("weights", "densenet121-res224-all")
        _orig_load = torch.load
        def _trusted_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return _orig_load(*args, **kwargs)
        torch.load = _trusted_load
        try:
            model = xrv.models.DenseNet(weights=weights)
        finally:
            torch.load = _orig_load
        model = model.to(device=device).eval()
        return model, None, spec

    if enc_type == "random_init":
        import timm
        arch = spec.get("arch", "vit_large_patch16_224")
        model = timm.create_model(arch, pretrained=False, num_classes=0)
        model = model.to(device=device, dtype=dtype).eval()
        return model, None, spec

    raise ValueError(f"[image_encoders] Unknown encoder type '{enc_type}'")



def _prep_images_hf(images: List[Image.Image], processor, device: str,
                    dtype: torch.dtype) -> dict:
    errs = []
    inputs = None

    img_proc = getattr(processor, "image_processor", None)
    if img_proc is not None:
        try:
            inputs = img_proc(images=images, return_tensors="pt")
        except Exception as e:
            errs.append(f"image_processor(images=...): {type(e).__name__}: {e}")
            inputs = None

    if inputs is None:
        try:
            inputs = processor(images=images, return_tensors="pt")
        except Exception as e:
            errs.append(f"processor(images=...): {type(e).__name__}: {e}")
            inputs = None

    if inputs is None:
        try:
            inputs = processor(images=images, text=[""] * len(images),
                               return_tensors="pt")
        except Exception as e:
            errs.append(f"processor(images=..., text=['']*N): {type(e).__name__}: {e}")
            inputs = None

    if inputs is None:
        raise RuntimeError(
            "Image preprocessing failed for this VLM processor. Tried:\n  "
            + "\n  ".join(errs)
        )

    return {k: (v.to(device=device, dtype=dtype if v.is_floating_point() else v.dtype)
                if hasattr(v, "to") else v)
            for k, v in inputs.items()}


def _l2_norm(x: torch.Tensor) -> np.ndarray:
    x = x.float()
    x = F.normalize(x, p=2, dim=-1)
    return x.cpu().numpy()


def _extract_dino_cls(model, processor, images: List[Image.Image],
                      device: str, dtype: torch.dtype) -> np.ndarray:
    inputs = _prep_images_hf(images, processor, device, dtype)
    with torch.no_grad():
        out = model(**{k: v for k, v in inputs.items()
                       if k in ("pixel_values", "pixel_values_videos")})
    if hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
        hs = out.last_hidden_state
    elif hasattr(out, "pooler_output") and out.pooler_output is not None:
        hs = out.pooler_output
    elif isinstance(out, (tuple, list)):
        hs = out[0]
    else:
        hs = out
    emb = hs[:, 0, :] if hs.dim() == 3 else hs
    if not torch.isfinite(emb).all():
        model.float()
        inputs32 = {k: (v.float() if v.is_floating_point() else v)
                    for k, v in inputs.items()}
        with torch.no_grad():
            out32 = model(**{k: v for k, v in inputs32.items()
                             if k in ("pixel_values", "pixel_values_videos")})
        hs32 = getattr(out32, "last_hidden_state", None)
        if hs32 is None:
            hs32 = out32[0] if isinstance(out32, (tuple, list)) else out32
        emb = hs32[:, 0, :] if hs32.dim() == 3 else hs32
    return _l2_norm(emb)


def _extract_clip_pooled(model, processor, images: List[Image.Image],
                         device: str, dtype: torch.dtype) -> np.ndarray:
    if processor == "open_clip":
        preprocess = getattr(model, "_oc_preprocess", None)
        if preprocess is None:
            raise RuntimeError("open_clip model missing its preprocess transform.")
        batch = torch.stack([preprocess(img.convert("RGB")) for img in images]
                            ).to(device=device, dtype=dtype)
        with torch.no_grad():
            emb = model.encode_image(batch)
        return _l2_norm(emb)

    inputs = _prep_images_hf(images, processor, device, dtype)
    pixel_key = "pixel_values" if "pixel_values" in inputs else list(inputs.keys())[0]
    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            emb = model.get_image_features(pixel_values=inputs[pixel_key])
        else:
            vision = getattr(model, "vision_model", model)
            out = vision(pixel_values=inputs[pixel_key])
            emb = _resolve_vision_output(out)
    # Final guard: if emb is still not a tensor, resolve once more, then fail.
    if not isinstance(emb, torch.Tensor):
        emb = _resolve_vision_output(emb)
    if not isinstance(emb, torch.Tensor):
        raise RuntimeError(
            f"CLIP extraction did not yield a tensor (got {type(emb).__name__}). "
            f"Check the model's image-feature API."
        )
    return _l2_norm(emb)


def _resolve_vision_output(out):
    """Resolve any HF vision output (wrapper object, tuple, dict, or tensor) to a
    pooled 2-D tensor. Never returns a wrapper object (which lacks .float())."""
    import torch as _torch
    if isinstance(out, _torch.Tensor):
        return out[:, 0, :] if out.dim() == 3 else out
    # Attribute-style outputs (BaseModelOutputWithPooling, CLIPVisionModelOutput)
    for attr in ("image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[:, 0, :] if lhs.dim() == 3 else lhs
    # Dict-style
    if isinstance(out, dict):
        for k in ("image_embeds", "pooler_output", "last_hidden_state"):
            v = out.get(k, None)
            if v is not None:
                return v[:, 0, :] if hasattr(v, "dim") and v.dim() == 3 else v
    # Tuple/list-style
    if isinstance(out, (tuple, list)) and len(out) > 0:
        first = out[0]
        if isinstance(first, _torch.Tensor):
            return first[:, 0, :] if first.dim() == 3 else first
    return out


def _extract_timm_cls(model, processor, images: List[Image.Image],
                      device: str, dtype: torch.dtype) -> np.ndarray:
    from torchvision import transforms

    if processor == "timm_raw":
        # Build the exact transform timm recorded for this checkpoint
        try:
            import timm
            data_cfg = timm.data.resolve_model_data_config(model)
            tfm = timm.data.create_transform(**data_cfg, is_training=False)
        except Exception:
            tfm = transforms.Compose([
                transforms.Resize(224), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        pixel_values = torch.stack([tfm(img.convert("RGB")) for img in images]
                                   ).to(device=device, dtype=dtype)
    elif processor == "conch":
        # CONCH uses CLIP-style 0.48/0.27 normalization at 224
        tfm = transforms.Compose([
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                 std=[0.26862954, 0.26130258, 0.27577711]),
        ])
        pixel_values = torch.stack([tfm(img.convert("RGB")) for img in images]
                                   ).to(device=device, dtype=dtype)
    elif processor is not None and not isinstance(processor, str):
        inputs = _prep_images_hf(images, processor, device, dtype)
        pixel_values = inputs.get("pixel_values", list(inputs.values())[0])
    else:
        tfm = transforms.Compose([
            transforms.Resize(224), transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        pixel_values = torch.stack([tfm(img.convert("RGB")) for img in images]
                                   ).to(device=device, dtype=dtype)

    with torch.no_grad():
        if processor == "conch":
            # CONCH visual tower returns a pooled embedding directly
            emb = model(pixel_values)
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            if emb.dim() == 3:
                emb = emb[:, 0, :]
        elif hasattr(model, "forward_features"):
            feats = model.forward_features(pixel_values)
            if feats.dim() == 3:
                emb = feats[:, 0, :]          # CLS token
            else:
                emb = feats                    # already pooled (B, D)
        else:
            out = model(pixel_values)
            if isinstance(out, torch.Tensor):
                emb = out[:, 0, :] if out.dim() == 3 else out
            else:
                emb = out.last_hidden_state[:, 0, :]
    return _l2_norm(emb)


def _extract_vlm_vision(model, processor, images: List[Image.Image],
                        device: str, dtype: torch.dtype) -> np.ndarray:
    inputs = _prep_images_hf(images, processor, device, dtype)
    # The grid tensor (Qwen) tells us how many patches belong to each image.
    grid = None
    for gk in ("image_grid_thw", "grid_thw"):
        if gk in inputs:
            grid = inputs[gk]
            break

    vision_kwargs = {k: v for k, v in inputs.items()
                     if any(t in k for t in ("pixel", "grid", "image_sizes",
                                             "spatial", "patch"))}
    if not vision_kwargs:
        vision_kwargs = inputs
    pixel_key = next((k for k in vision_kwargs if "pixel" in k),
                     list(vision_kwargs.keys())[0])

    tile_counts = None
    pv = vision_kwargs.get(pixel_key, None)
    if pv is not None and hasattr(pv, "dim") and pv.dim() == 5:
        b, t = pv.shape[0], pv.shape[1]
        tile_counts = [t] * b
        vision_kwargs = dict(vision_kwargs)
        vision_kwargs[pixel_key] = pv.reshape(b * t, *pv.shape[2:])

    with torch.no_grad():
        out, errors = None, []
        attempts = []
        if grid is not None:
            pv_arg = vision_kwargs[pixel_key]
            attempts += [
                # Qwen3-VL's vision tower forward signature is
                # forward(hidden_states, grid_thw): the first arg is named
                # 'hidden_states', NOT 'pixel_values'.
                ("qwen3_hidden_kw", dict(call="kw",  kwargs={"hidden_states": pv_arg, "grid_thw": grid})),
                ("qwen_pixel_kw",   dict(call="kw",  kwargs={pixel_key: pv_arg, "grid_thw": grid})),
                ("qwen_positional", dict(call="pos", args=(pv_arg, grid))),
            ]
        attempts += [
            ("full_kwargs",      dict(call="kw",  kwargs=vision_kwargs)),
            ("pixel_only_kw",    dict(call="kw",  kwargs={pixel_key: vision_kwargs[pixel_key]})),
            ("pixel_positional", dict(call="pos", args=(vision_kwargs[pixel_key],))),
        ]
        for name, spec in attempts:
            try:
                if spec["call"] == "kw":
                    out = model(**spec["kwargs"])
                else:
                    out = model(*spec["args"])
                if out is not None:
                    break
            except Exception as e:
                errors.append(f"{name}: {type(e).__name__}: {e}")
                out = None
                continue
        if out is None:
            raise RuntimeError(
                "VLM vision tower could not be called with any known signature. "
                "Tried:\n  " + "\n  ".join(errors)
            )
        n_eff = sum(tile_counts) if tile_counts is not None else len(images)
        emb = _resolve_vlm_output(out, grid=grid, n_images=n_eff)
        if tile_counts is not None and isinstance(emb, torch.Tensor) \
                and emb.dim() == 2 and emb.size(0) == sum(tile_counts):
            chunks = torch.split(emb, tile_counts, dim=0)
            emb = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
    return _l2_norm(emb)


def _resolve_vlm_output(out, grid=None, n_images: int = 1):
    val = None
    for attr in ("last_hidden_state", "pooler_output", "image_embeds"):
        v = getattr(out, attr, None)
        if v is not None:
            val = v
            break
    if val is None and isinstance(out, torch.Tensor):
        val = out
    if val is None and isinstance(out, (tuple, list)) and len(out) > 0 \
            and isinstance(out[0], torch.Tensor):
        val = out[0]
    if val is None:
        raise RuntimeError(
            f"Could not resolve VLM vision output of type {type(out).__name__}."
        )

    if val.dim() == 2 and grid is not None and n_images >= 1:
        try:
            import numpy as _np
            g_np = grid.detach().cpu().numpy() if hasattr(grid, "detach") \
                else _np.asarray([[int(x) for x in row] for row in grid])
            g_np = g_np.reshape(-1, 3)
            raw_counts = [int(r[0] * r[1] * r[2]) for r in g_np]
            total_raw  = sum(raw_counts)
            rows = int(val.size(0))
            if len(raw_counts) == n_images and total_raw % max(rows, 1) == 0 \
                    and rows > 0 and total_raw >= rows:
                divisor = total_raw // rows         # 1, 4, 9, ... (merge^2)
                if all(c % divisor == 0 for c in raw_counts):
                    counts = [c // divisor for c in raw_counts]
                    if sum(counts) == rows:
                        chunks = torch.split(val, counts, dim=0)
                        return torch.stack([c.mean(dim=0) for c in chunks], dim=0)
        except Exception:
            pass  # fall through to generic handling

    if val.dim() == 3:               # (B, tokens, D) -> mean-pool tokens
        return val.mean(dim=1)
    if val.dim() == 2:               # (B, D) already pooled, or single-image (tokens, D)
        if n_images == 1 and val.size(0) != 1:
            return val.mean(dim=0, keepdim=True)   # (tokens, D) -> (1, D)
        return val
    return val.view(val.size(0), -1)


def _extract_txrv(model, processor, images: List[Image.Image],
                  device: str, dtype: torch.dtype) -> np.ndarray:
    import torchxrayvision as xrv
    from torchvision import transforms
    tfm = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    imgs_np = []
    for img in images:
        arr = np.array(img.convert("L")).astype(np.float32)
        # TorchXRayVision expects images normalized to the [-1024, 1024] range
        # via its own helper (NOT [-1, 1]); using the wrong range yields garbage.
        arr = xrv.datasets.normalize(arr, 255)
        arr = tfm(arr[None])                 # (1, 224, 224) after transforms
        imgs_np.append(arr)
    tensor = torch.tensor(np.stack(imgs_np)).float().to(device)
    with torch.no_grad():
        # features2() returns the post-global-average-pool feature vector (B, 1024)
        emb = model.features2(tensor)
        if emb.dim() > 2:
            emb = emb.view(emb.size(0), -1)
    return _l2_norm(emb)


def _extract_random_init(model, processor, images: List[Image.Image],
                         device: str, dtype: torch.dtype) -> np.ndarray:
    return _extract_timm_cls(model, processor, images, device, dtype)


_EXTRACT_FN = {
    "dino_cls":    _extract_dino_cls,
    "clip_pooled": _extract_clip_pooled,
    "timm_cls":    _extract_timm_cls,
    "vlm_vision":  _extract_vlm_vision,
    "txrv":        _extract_txrv,
    "random_init": _extract_random_init,
}


def extract_image_embeddings(
    model_name: str,
    images: List[Image.Image],
    cfg_path: str,
    device: str = "cuda",
) -> np.ndarray:
    model, processor, spec = _load_encoder(model_name, cfg_path, device)
    enc_type = spec["type"]
    cfg  = read_config(cfg_path)
    fp16 = cfg["Convergence"]["embeddings"].get("fp16", True)
    dtype = torch.float16 if fp16 else torch.float32

    fn = _EXTRACT_FN.get(enc_type)
    if fn is None:
        raise ValueError(f"[image_encoders] No extraction function for type '{enc_type}'")

    return fn(model, processor, images, device, dtype)


def list_encoder_names(cfg_path: str, roles: Optional[List[str]] = None) -> List[str]:
    """Return image encoder names from the panel, optionally filtered by role."""
    cfg   = read_config(cfg_path)
    panel = cfg["Convergence"]["encoder_panel"]["image"]
    if roles is None:
        return sorted(panel.keys())
    role_set = set(roles)
    return sorted(
        name for name, spec in panel.items()
        if role_set.intersection(spec.get("roles", []))
    )