"""
encoders/image_encoders.py
Created on June 20, 2026

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
    cfg = None
    try:
        cfg = AutoConfig.from_pretrained(
            hf_id, token=token, trust_remote_code=True,
            local_files_only=is_local,
        )
        arch_names = list(getattr(cfg, "architectures", []) or [])
    except Exception as e:
        arch_names = []

    last_err = None

    for arch in arch_names:
        cls = getattr(tf, arch, None)
        if cls is not None:
            try:
                model = cls.from_pretrained(hf_id, **common)
                if _has_vision_tower(model):
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
            resolved = type(model).__name__
            if resolved in arch_names or _has_vision_tower(model):
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
    cfg  = read_config(cfg_path)
    spec = _get_encoder_spec(model_name, cfg)
    enc_type = spec["type"]
    hf_id    = spec.get("hf_id")

    token = resolve_hf_token(cfg)
    export_hf_token(token)
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
            oc_model._oc_preprocess = preprocess
            return oc_model, "open_clip", spec
        from transformers import AutoModel, AutoProcessor
        import transformers as _tf
        from transformers import AutoConfig
        _cfg = AutoConfig.from_pretrained(hf_id, token=token, trust_remote_code=True)
        _archs = list(getattr(_cfg, "architectures", []) or [])
        _by_type = {"siglip": ["SiglipModel", "Siglip2Model"], "siglip2": ["Siglip2Model", "SiglipModel"],
                    "clip": ["CLIPModel"], "altclip": ["AltCLIPModel"]}
        _candidates = []
        for _n in _archs + _by_type.get(str(getattr(_cfg, "model_type", "")), []) + ["CLIPModel", "SiglipModel", "AltCLIPModel"]:
            if _n not in _candidates:
                _candidates.append(_n)
        model = None
        _errors = []
        for _name in _candidates:
            _cls = getattr(_tf, _name, None)
            if _cls is None:
                _errors.append(f"{_name}: not in transformers")
                continue
            try:
                _m, _info = _cls.from_pretrained(hf_id, token=token, trust_remote_code=True,
                                                 output_loading_info=True)
            except Exception as _e:
                _errors.append(f"{_name}: {type(_e).__name__}: {str(_e)[:160]}")
                continue
            _missing = list(_info.get("missing_keys", []) or [])
            _mismatched = list(_info.get("mismatched_keys", []) or [])
            if _missing or _mismatched:
                _errors.append(f"{_name}: loaded with {len(_missing)} missing and "
                               f"{len(_mismatched)} re-initialized weights, refused "
                               f"(first: {(_missing + [str(x) for x in _mismatched])[:2]})")
                del _m
                continue
            if not hasattr(_m, "get_image_features"):
                _errors.append(f"{_name}: loaded but no get_image_features")
                del _m
                continue
            model = _m
            break
        if model is None:
            raise RuntimeError(
                f"[load/clip_pooled] {model_name}: no class loads {hf_id} cleanly with "
                f"get_image_features. A model with re-initialized weights would embed noise under "
                f"the right name, so nothing is extracted. Tried:\n  " + "\n  ".join(_errors))
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

        timm_kwargs = {"pretrained": True, "num_classes": 0}
        extra = {
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

        if model_name in ("uni2", "virchow", "virchow2"):
            from timm.layers import SwiGLUPacked
            import torch.nn as nn
            extra["mlp_layer"] = SwiGLUPacked
            extra["act_layer"] = nn.SiLU

        timm_kwargs.update(extra)
        model = timm.create_model(f"hf-hub:{hf_id}", **timm_kwargs)
        model = model.to(device=device, dtype=dtype).eval()
        return model, "timm_raw", spec

    if enc_type == "vlm_vision":
        from transformers import AutoProcessor
        full_model = _load_vlm_full(hf_id, token, device, dtype)

        vision_attr = spec.get("vision_attr", "")
        candidates = [vision_attr] if vision_attr else []
        candidates += [
            "visual",
            "model.visual",
            "vision_tower",
            "model.vision_tower",
            "vision_model",
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


def harmonize_images(images: List[Image.Image], cfg: dict) -> List[Image.Image]:
    from torchvision import transforms
    h = cfg["embeddings"]["harmonized_preprocessing"]
    geom = transforms.Compose([transforms.Resize(int(h["resize"])),
                               transforms.CenterCrop(int(h["crop"]))])
    return [geom(im.convert("RGB")) for im in images]


def harmonized_pixels(images: List[Image.Image], cfg: dict,
                      device: str, dtype: torch.dtype) -> torch.Tensor:
    from torchvision import transforms
    h = cfg["embeddings"]["harmonized_preprocessing"]
    to_t = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize(mean=list(h["mean"]), std=list(h["std"]))])
    return torch.stack([to_t(im) for im in harmonize_images(images, cfg)]
                       ).to(device=device, dtype=dtype)


HARMONIZABLE_TYPES = ("dino_cls", "clip_pooled", "timm_cls", "vlm_vision", "random_init")

from encoders.cache_utils import HARMONIZED_PIPELINE


def _native_size(model, processor) -> Optional[int]:
    cached = getattr(model, "_native_input_size", None)
    if isinstance(cached, int):
        return cached
    probe = Image.new("RGB", (320, 288), color=(90, 90, 90))
    if processor == "conch":
        return 224
    if processor == "timm_raw":
        try:
            import timm
            size = int(timm.data.resolve_model_data_config(model)["input_size"][-1])
            model._native_input_size = size
            return size
        except Exception:
            pass
    try:
        if processor is not None and not isinstance(processor, str):
            out = processor(images=[probe], return_tensors="pt")
            pv = out.get("pixel_values") if hasattr(out, "get") else None
            if pv is None and hasattr(out, "keys"):
                pv = next((out[k] for k in out.keys() if "pixel" in k), None)
            if pv is not None and hasattr(pv, "shape") and len(pv.shape) >= 3:
                size = int(pv.shape[-1])
                try:
                    model._native_input_size = size
                except Exception:
                    pass
                return size
        pre = getattr(model, "_oc_preprocess", None)
        if pre is not None:
            size = int(pre(probe).shape[-1])
            try:
                model._native_input_size = size
            except Exception:
                pass
            return size
    except Exception:
        pass
    def _from_size_dict(d):
        if isinstance(d, dict):
            for key in ("height", "shortest_edge", "width"):
                v = d.get(key)
                if isinstance(v, (int, float)) and v > 0:
                    return int(v)
        if isinstance(d, (int, float)) and d > 0:
            return int(d)
        return None
    for holder in (processor, getattr(processor, "image_processor", None)):
        if holder is None or isinstance(holder, str):
            continue
        for attr in ("crop_size", "size"):
            v = _from_size_dict(getattr(holder, attr, None))
            if v:
                return v
    pre = getattr(model, "_oc_preprocess", None)
    if pre is not None:
        for t in reversed(getattr(pre, "transforms", [])):
            sz = getattr(t, "size", None)
            if isinstance(sz, (list, tuple)) and sz:
                return int(sz[-1])
            if isinstance(sz, int):
                return int(sz)
    vis = getattr(model, "visual", model)
    v = getattr(vis, "image_size", None)
    if isinstance(v, (list, tuple)) and v:
        return int(v[-1])
    if isinstance(v, int) and v > 0:
        return int(v)
    pc = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", None) or {}
    if isinstance(pc, dict) and pc.get("input_size"):
        return int(pc["input_size"][-1])
    cfgobj = getattr(model, "config", None)
    for c in (getattr(cfgobj, "vision_config", None), cfgobj):
        v = getattr(c, "image_size", None)
        if isinstance(v, (list, tuple)) and v:
            return int(v[-1])
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


def _fit_to_native(pixels: torch.Tensor, size: Optional[int]) -> torch.Tensor:
    if size is None or pixels.dim() != 4 or tuple(pixels.shape[-2:]) == (size, size):
        return pixels
    out = F.interpolate(pixels.float(), size=(size, size), mode="bicubic",
                        align_corners=False, antialias=True)
    return out.to(pixels.dtype)


def _l2_norm(x: torch.Tensor) -> np.ndarray:
    x = x.float()
    x = F.normalize(x, p=2, dim=-1)
    return x.cpu().numpy()


def _cast_inputs(inputs: dict, target: torch.dtype) -> dict:
    return {k: (v.to(target) if hasattr(v, "is_floating_point") and v.is_floating_point() else v)
            for k, v in inputs.items()}


def _promote_until_finite(model, inputs: dict, forward, name: str):
    emb = forward(model, inputs)
    if torch.isfinite(emb).all():
        return emb
    weight_dtype = next((p.dtype for p in model.parameters()), None)
    if weight_dtype not in (torch.float16, torch.bfloat16):
        return emb
    for target in (torch.bfloat16, torch.float32):
        if target == weight_dtype:
            continue
        model.to(dtype=target)
        model._promoted_dtype = target
        emb = forward(model, _cast_inputs(inputs, target))
        if torch.isfinite(emb).all():
            return emb
    return emb


def _dino_forward(model, inputs: dict) -> torch.Tensor:
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
    return hs[:, 0, :] if hs.dim() == 3 else hs


def _extract_dino_cls(model, processor, images: List[Image.Image],
                      device: str, dtype: torch.dtype, pixels=None) -> np.ndarray:
    if pixels is not None:
        pixels = _fit_to_native(pixels, _native_size(model, processor))
    inputs = ({"pixel_values": pixels} if pixels is not None
              else _prep_images_hf(images, processor, device, dtype))
    promoted = getattr(model, "_promoted_dtype", None)
    if promoted is not None:
        inputs = _cast_inputs(inputs, promoted)
    name = str(getattr(model, "name_or_path", "") or type(model).__name__)
    return _l2_norm(_promote_until_finite(model, inputs, _dino_forward, name))


def _clip_image_features(model, pixel_values: torch.Tensor) -> torch.Tensor:
    out = model.get_image_features(pixel_values=pixel_values)
    if isinstance(out, torch.Tensor):
        return out
    if not getattr(model, "_clip_pooled_projects_here", False):
        model._clip_pooled_projects_here = True
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        raise RuntimeError(
            f"[clip_pooled] get_image_features returned {type(out).__name__} with no pooler_output, "
            f"so the projected image embedding cannot be recovered. Fix this checkpoint's loader; "
            f"pooling a hidden state instead would land in a different space.")
    proj = getattr(model, "visual_projection", None)
    if proj is None:
        return pooled
    width = int(pooled.shape[-1])
    in_features = getattr(proj, "in_features", None)
    out_features = getattr(proj, "out_features", None)
    if out_features is not None and width == int(out_features) and width != int(in_features or -1):
        return pooled
    if in_features is not None and width == int(in_features):
        return proj(pooled)
    raise RuntimeError(
        f"[clip_pooled] the pooled vision output is {width} wide while visual_projection maps "
        f"{in_features} to {out_features}, so which space this vector sits in is not decidable here.")


def _extract_clip_pooled(model, processor, images: List[Image.Image],
                         device: str, dtype: torch.dtype, pixels=None) -> np.ndarray:
    if processor == "open_clip":
        if pixels is None:
            preprocess = getattr(model, "_oc_preprocess", None)
            if preprocess is None:
                raise RuntimeError("open_clip model missing its preprocess transform.")
            pixels = torch.stack([preprocess(img.convert("RGB")) for img in images]
                                 ).to(device=device, dtype=dtype)
        else:
            pixels = _fit_to_native(pixels, _native_size(model, processor))
        with torch.no_grad():
            return _l2_norm(model.encode_image(pixels))

    if pixels is None:
        inputs = _prep_images_hf(images, processor, device, dtype)
        pixel_key = "pixel_values" if "pixel_values" in inputs else list(inputs.keys())[0]
        pixels = inputs[pixel_key]
    else:
        pixels = _fit_to_native(pixels, _native_size(model, processor))

    with torch.no_grad():
        if hasattr(model, "get_image_features"):
            emb = _clip_image_features(model, pixels)
        else:
            vision = getattr(model, "vision_model", model)
            emb = _resolve_vision_output(vision(pixel_values=pixels))
    if not isinstance(emb, torch.Tensor):
        raise RuntimeError(
            f"[clip_pooled] extraction did not yield a tensor (got {type(emb).__name__}). "
            f"Check this checkpoint's image-feature API."
        )
    return _l2_norm(emb)


def _resolve_vision_output(out):
    import torch as _torch
    if isinstance(out, _torch.Tensor):
        return out[:, 0, :] if out.dim() == 3 else out
    for attr in ("image_embeds", "pooler_output"):
        v = getattr(out, attr, None)
        if v is not None:
            return v
    lhs = getattr(out, "last_hidden_state", None)
    if lhs is not None:
        return lhs[:, 0, :] if lhs.dim() == 3 else lhs
    if isinstance(out, dict):
        for k in ("image_embeds", "pooler_output", "last_hidden_state"):
            v = out.get(k, None)
            if v is not None:
                return v[:, 0, :] if hasattr(v, "dim") and v.dim() == 3 else v
    if isinstance(out, (tuple, list)) and len(out) > 0:
        first = out[0]
        if isinstance(first, _torch.Tensor):
            return first[:, 0, :] if first.dim() == 3 else first
    return out


def _extract_timm_cls(model, processor, images: List[Image.Image],
                      device: str, dtype: torch.dtype, pixels=None) -> np.ndarray:
    from torchvision import transforms

    if pixels is not None:
        pixel_values = _fit_to_native(pixels, _native_size(model, processor))
    elif processor == "timm_raw":
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
            emb = model(pixel_values)
            if isinstance(emb, (tuple, list)):
                emb = emb[0]
            if emb.dim() == 3:
                emb = emb[:, 0, :]
        elif hasattr(model, "forward_features"):
            feats = model.forward_features(pixel_values)
            if feats.dim() == 3:
                emb = feats[:, 0, :]
            else:
                emb = feats
        else:
            out = model(pixel_values)
            if isinstance(out, torch.Tensor):
                emb = out[:, 0, :] if out.dim() == 3 else out
            else:
                emb = out.last_hidden_state[:, 0, :]
    return _l2_norm(emb)


def _extract_vlm_vision(model, processor, images: List[Image.Image],
                        device: str, dtype: torch.dtype, pixels=None) -> np.ndarray:
    if pixels is not None:
        pv = _fit_to_native(pixels, _native_size(model, processor))
        with torch.no_grad():
            out = None
            errors = []
            for name, call in (("pixel_kw", lambda: model(pixel_values=pv)),
                               ("positional", lambda: model(pv))):
                try:
                    out = call()
                    if out is not None:
                        break
                except Exception as e:
                    errors.append(f"{name}: {type(e).__name__}: {e}")
                    out = None
            if out is None:
                raise RuntimeError("VLM vision tower refused the harmonized tensor. Tried:\n  "
                                   + "\n  ".join(errors))
            emb = _resolve_vlm_output(out, grid=None, n_images=int(pv.shape[0]))
        return _l2_norm(emb)
    inputs = _prep_images_hf(images, processor, device, dtype)
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
                divisor = total_raw // rows
                if all(c % divisor == 0 for c in raw_counts):
                    counts = [c // divisor for c in raw_counts]
                    if sum(counts) == rows:
                        chunks = torch.split(val, counts, dim=0)
                        return torch.stack([c.mean(dim=0) for c in chunks], dim=0)
        except Exception:
            pass

    if val.dim() == 3:
        return val.mean(dim=1)
    if val.dim() == 2:
        if n_images == 1 and val.size(0) != 1:
            return val.mean(dim=0, keepdim=True)
        return val
    return val.view(val.size(0), -1)


def _extract_txrv(model, processor, images: List[Image.Image],
                  device: str, dtype: torch.dtype, pixels=None) -> np.ndarray:
    import torchxrayvision as xrv
    from torchvision import transforms
    tfm = transforms.Compose([
        xrv.datasets.XRayCenterCrop(),
        xrv.datasets.XRayResizer(224),
    ])
    imgs_np = []
    for img in images:
        arr = np.array(img.convert("L")).astype(np.float32, copy=False)
        arr = xrv.datasets.normalize(arr, 255)
        arr = tfm(arr[None])
        imgs_np.append(arr)
    tensor = torch.tensor(np.stack(imgs_np)).float().to(device)
    with torch.no_grad():
        emb = model.features2(tensor)
        if emb.dim() > 2:
            emb = emb.view(emb.size(0), -1)
    return _l2_norm(emb)


def _extract_random_init(model, processor, images: List[Image.Image],
                         device: str, dtype: torch.dtype, pixels=None) -> np.ndarray:
    return _extract_timm_cls(model, processor, images, device, dtype, pixels=pixels)


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
    harmonized: bool = False,
) -> np.ndarray:
    model, processor, spec = _load_encoder(model_name, cfg_path, device)
    enc_type = spec["type"]
    cfg  = read_config(cfg_path)
    fp16 = cfg["Convergence"]["embeddings"].get("fp16", True)
    dtype = torch.float16 if fp16 else torch.float32

    fn = _EXTRACT_FN.get(enc_type)
    if fn is None:
        raise ValueError(f"[image_encoders] No extraction function for type '{enc_type}'")

    if not harmonized:
        return fn(model, processor, images, device, dtype)
    conv = cfg["Convergence"]
    if enc_type in HARMONIZABLE_TYPES:
        return fn(model, processor, images, device, dtype,
                  pixels=harmonized_pixels(images, conv, device, dtype))
    return fn(model, processor, harmonize_images(images, conv), device, dtype)


from encoders.panel import (encoder_objective, export_hf_token,
                            list_encoder_names, requires_access_token,
                            resolve_hf_token)
