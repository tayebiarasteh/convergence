"""
data_loader/preprocess_fundus.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

from config.serde import read_config
from data_loader.preprocess_utils import resize_tree


def main_preprocess_fundus(global_config_path: str):
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    fcfg   = cfg.get("fundus", {})
    if not fcfg.get("enabled", False):
        return

    image_root = fcfg["image_root"]
    for name, scfg in fcfg.get("sources", {}).items():
        raw_dir = scfg.get("raw_dir")           # absent -> skip (no-op)
        tag = scfg.get("source_tag", name)
        resize_tree(
            raw_root=raw_dir,
            out_root_224=os.path.join(image_root, "preprocessed224", tag),
            out_root_512=os.path.join(image_root, "preprocessed", tag),
            tag=f"_fundus/{name}",
        )
