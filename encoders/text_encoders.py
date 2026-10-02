"""
encoders/text_encoders.py
Created on June 21, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from functools import lru_cache
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from config.serde import read_config
from encoders.panel import export_hf_token, resolve_hf_token

import warnings
warnings.filterwarnings("ignore")


def _get_text_spec(model_name: str, cfg: dict) -> dict:
    panel = cfg["Convergence"]["encoder_panel"]["text"]
    if model_name not in panel:
        raise KeyError(
            f"[text_encoders] '{model_name}' not in encoder_panel.text. "
            f"Available: {sorted(panel.keys())}"
        )
    return panel[model_name]


@lru_cache(maxsize=4)
def _load_text_encoder(model_name: str, cfg_path: str, device: str):
    cfg  = read_config(cfg_path)
    spec = _get_text_spec(model_name, cfg)
    hf_id    = spec["hf_id"]
    enc_type = spec["type"]
    token = resolve_hf_token(cfg)
    export_hf_token(token)
    fp16  = cfg["Convergence"]["embeddings"].get("fp16", True)
    dtype = torch.float16 if fp16 else torch.float32

    if enc_type in ("bert_cls", "bert_mean"):
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(hf_id, token=token)
        tok   = AutoTokenizer.from_pretrained(hf_id, token=token)
        model = model.to(device=device, dtype=dtype).eval()
        return model, tok, spec

    if enc_type == "clip_text":
        if spec.get("loader") == "open_clip" or "BiomedCLIP" in hf_id:
            import open_clip
            oc_model, _, _ = open_clip.create_model_and_transforms(f"hf-hub:{hf_id}")
            oc_tokenizer   = open_clip.get_tokenizer(f"hf-hub:{hf_id}")
            oc_model = oc_model.to(device=device, dtype=dtype).eval()
            return oc_model, ("open_clip", oc_tokenizer), spec
        from transformers import AutoModel, AutoTokenizer
        model = AutoModel.from_pretrained(hf_id, token=token, trust_remote_code=True)
        tok   = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)
        model = model.to(device=device, dtype=dtype).eval()
        return model, tok, spec

    if enc_type == "conch_text":
        try:
            from conch.open_clip_custom import create_model_from_pretrained
            from conch.open_clip_custom import get_tokenizer as conch_get_tokenizer
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"CONCH text requires the 'conch' package (pip install git+https://"
                f"github.com/mahmoodlab/CONCH.git) and accepted HF terms. "
                f"Import error: {e}"
            ) from e
        conch_model, _ = create_model_from_pretrained(
            "conch_ViT-B-16", checkpoint_path=f"hf_hub:{hf_id}", hf_auth_token=token,
        )
        conch_model = conch_model.to(device=device, dtype=dtype).eval()
        base_tok = conch_get_tokenizer()
        if not hasattr(base_tok, "batch_encode_plus"):
            def _batch_encode_plus(self, batch, **kw):
                return self(batch, **kw)
            import types as _types
            try:
                base_tok.batch_encode_plus = _types.MethodType(_batch_encode_plus, base_tok)
            except Exception:
                pass
        conch_tok = ("conch_native", base_tok)
        return conch_model, ("conch", conch_tok), spec

    if enc_type == "causal_lm_last":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch as _torch

        load_kwargs = dict(token=token, trust_remote_code=True)

        load_8bit = bool(spec.get("load_in_8bit", False))
        if load_8bit:
            raise ValueError(
                f"[text] 8-bit quantization is not used in this project: it has produced all-NaN "
                f"outputs for checkpoints of this family. Set load_in_4bit instead.")
        load_4bit = bool(spec.get("load_in_4bit", False))

        if device == "cuda" and _torch.cuda.is_available():
            n_gpus = _torch.cuda.device_count()
            headroom = float(cfg["Convergence"].get("embeddings", {}).get("gpu_headroom_gib", 6.0))
            max_mem = {}
            for gi in range(n_gpus):
                total_gib = _torch.cuda.get_device_properties(gi).total_memory / (1024**3)
                cap = cfg["Convergence"].get("embeddings", {}).get("max_memory_per_gpu_gib")
                budget = float(cap) if cap else max(total_gib - headroom, 1.0)
                max_mem[gi] = f"{budget:.0f}GiB"
            max_mem["cpu"] = f"{int(cfg['Convergence'].get('embeddings', {}).get('cpu_offload_gib', 96))}GiB"
            load_kwargs["device_map"] = "auto"
            load_kwargs["max_memory"] = max_mem
        else:
            load_kwargs["device_map"] = None

        if load_8bit or load_4bit:
            from transformers import BitsAndBytesConfig
            if load_4bit:
                bnb = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=dtype,
                    bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
            else:
                bnb = BitsAndBytesConfig(load_in_8bit=True)
            load_kwargs["quantization_config"] = bnb
        else:
            import inspect
            _sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
            load_kwargs["dtype" if "dtype" in _sig.parameters else "torch_dtype"] = dtype

        ModelClass = AutoModelForCausalLM
        is_gemma3_vlm = False
        try:
            from transformers import AutoConfig
            _cfg = AutoConfig.from_pretrained(hf_id, token=token, trust_remote_code=True)
            arch = (getattr(_cfg, "architectures", None) or [""])[0]
            if "Gemma3ForConditionalGeneration" in arch or (
                    "Gemma3" in arch and hasattr(_cfg, "vision_config")):
                from transformers import Gemma3ForConditionalGeneration
                ModelClass = Gemma3ForConditionalGeneration
                is_gemma3_vlm = True
        except Exception as e:
            pass

        model = ModelClass.from_pretrained(hf_id, **load_kwargs)
        if is_gemma3_vlm:
            lm = getattr(model, "language_model", None)
            if lm is not None:
                model = lm
        tok = AutoTokenizer.from_pretrained(hf_id, token=token, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model.eval()
        return model, tok, spec

    raise ValueError(f"[text_encoders] Unknown type '{enc_type}'")


def _l2_norm(x: torch.Tensor) -> np.ndarray:
    return F.normalize(x.float(), p=2, dim=-1).cpu().numpy()


def _project_text(model, out) -> torch.Tensor:
    pooled = getattr(out, "pooler_output", None)
    if pooled is None:
        raise RuntimeError(
            f"[text_encoders] the text tower returned {type(out).__name__} with no pooler_output, "
            f"so the projected embedding cannot be recovered.")
    proj = getattr(model, "text_projection", None)
    if proj is None or not hasattr(proj, "in_features"):
        return pooled
    if int(pooled.shape[-1]) != int(proj.in_features):
        raise RuntimeError(
            f"[text_encoders] the pooled text output is {int(pooled.shape[-1])} wide while "
            f"text_projection takes {int(proj.in_features)}.")
    return proj(pooled)


def _last_real_token(attention_mask: torch.Tensor) -> torch.Tensor:
    flipped = attention_mask.flip(dims=[1])
    trailing = flipped.to(torch.int64).argmax(dim=1)
    return (attention_mask.size(1) - 1 - trailing).clamp(min=0)


def extract_text_embeddings(
    model_name: str,
    texts: List[str],
    cfg_path: str,
    device: str = "cuda",
) -> np.ndarray:
    model, tok, spec = _load_text_encoder(model_name, cfg_path, device)
    enc_type = spec["type"]

    if enc_type in ("bert_cls", "bert_mean"):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc)
        hs = out.last_hidden_state
        if enc_type == "bert_cls":
            emb = hs[:, 0, :]
        else:
            mask = enc["attention_mask"].unsqueeze(-1).float()
            emb  = (hs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return _l2_norm(emb)

    if enc_type == "clip_text":
        if isinstance(tok, tuple) and tok[0] == "open_clip":
            oc_tokenizer = tok[1]
            tokens = oc_tokenizer(texts).to(device)
            with torch.no_grad():
                emb = model.encode_text(tokens)
            return _l2_norm(emb)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=77)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            emb = None
            if hasattr(model, "get_text_features"):
                emb = model.get_text_features(**enc)
                if not isinstance(emb, torch.Tensor):
                    emb = _project_text(model, emb)
            if emb is None:
                out = model.text_model(**enc)
                emb = _project_text(model, out)
        return _l2_norm(emb)

    if enc_type == "conch_text":
        if isinstance(tok, tuple) and tok[0] == "conch":
            _kind, conch_tok = tok[1]
            from conch.open_clip_custom import tokenize as conch_tokenize
            token_ids = conch_tokenize(texts=texts, tokenizer=conch_tok).to(device)
            with torch.no_grad():
                emb = model.encode_text(token_ids)
            return _l2_norm(emb)
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            out = model.get_text_features(**enc) if hasattr(model, "get_text_features") \
                  else model(**enc).last_hidden_state[:, 0, :]
        return _l2_norm(out)

    if enc_type == "causal_lm_last":
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=512)
        try:
            in_dev = model.get_input_embeddings().weight.device
        except Exception:
            in_dev = next(model.parameters()).device
        enc = {k: v.to(in_dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        hs  = out.hidden_states[-1]
        idx = _last_real_token(enc["attention_mask"]).to(hs.device)
        emb  = hs[torch.arange(len(texts), device=hs.device), idx, :]
        return _l2_norm(emb)

    raise ValueError(f"[text_encoders] No extraction path for type '{enc_type}'")


def list_text_encoder_names(cfg_path: str) -> List[str]:
    cfg = read_config(cfg_path)
    return sorted(cfg["Convergence"]["encoder_panel"]["text"].keys())
