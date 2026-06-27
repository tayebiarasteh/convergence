"""
data_loader/preprocess_cxr.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

from config.serde import read_config
from data_loader.preprocess_utils import resize_tree


# Per-site preprocessed output sub-paths relative to the site image_root.
_OUT_SUBPATH = {
    "mimic":      ("preprocessed224", "preprocessed"),
    "chexpert":   ("CheXpert-v1.0/preprocessed224", "CheXpert-v1.0/preprocessed"),
    "nih_cxr14":  ("CXR14/preprocessed224", "CXR14/preprocessed"),
    "padchest":   ("preprocessed224", "preprocessed"),
    "vindr_cxr":  ("preprocessed224", "preprocessed"),
    "vindr_pcxr": ("preprocessed224", "preprocessed"),
}


def main_preprocess_cxr(global_config_path: str):
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    sites  = cfg["cxr"]["sites"]

    for site, scfg in sites.items():
        if not scfg.get("enabled", True):
            continue
        raw_root = scfg.get("raw_image_root")   # absent -> skip (no-op)
        image_root = scfg["image_root"]
        sub224, sub512 = _OUT_SUBPATH[site]
        resize_tree(
            raw_root=raw_root,
            out_root_224=os.path.join(image_root, sub224),
            out_root_512=os.path.join(image_root, sub512),
            tag=f"_cxr/{site}",
        )
