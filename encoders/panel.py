"""
encoders/panel.py
Created on August 30, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List, Optional

from config.serde import read_config

import warnings
warnings.filterwarnings("ignore")


def resolve_hf_token(cfg: dict) -> Optional[str]:
    block = cfg.get("Convergence", cfg)
    token = str(block.get("hf_token") or "").strip()
    return token if token.startswith("hf_") else None


def export_hf_token(token: Optional[str]) -> None:
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def requires_access_token(spec: dict) -> bool:
    return bool(spec.get("gated", False))


def list_encoder_names(cfg_path: str, roles: Optional[List[str]] = None,
                       require_token: bool = False) -> List[str]:
    cfg = read_config(cfg_path)
    panel = cfg["Convergence"]["encoder_panel"]["image"]
    if require_token:
        token = str(cfg["Convergence"].get("hf_token", "")).strip()
        ungated = [n for n, spec in panel.items()
                   if requires_access_token(spec) and not token.startswith("hf_")]
        if ungated:
            raise RuntimeError(f"these encoders are gated and Convergence.hf_token is not set: "
                               f"{sorted(ungated)}. Accept their terms and paste the token into "
                               f"config.yaml before running any extraction stage.")
    if roles is None:
        raise ValueError(
            "list_encoder_names: name the roles you want, for example roles=['core']. "
            "Pass roles=list(panel) explicitly if you really mean every encoder.")
    role_set = set(roles)
    return sorted(name for name, spec in panel.items()
                  if role_set.intersection(spec.get("roles", [])))


def list_text_encoder_names(cfg_path: str) -> List[str]:
    return sorted(read_config(cfg_path)["Convergence"]["encoder_panel"].get("text", {}))


def encoder_objective(cfg: dict, name: str) -> str:
    return str(cfg["encoder_panel"]["image"].get(name, {}).get("objective", "unknown"))
