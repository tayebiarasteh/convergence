"""
data_loader/preprocess_cxr.py
Created on June 13, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

from config.serde import read_config
from data_loader.preprocess_utils import resize_tree
from Inference.resume_utils import (MissingInput, append_status, check_build_params,
                                    status_path, write_build_params, write_csv_atomic)


_OUT_SUBPATH = {
    "mimic":      ("preprocessed224", "preprocessed"),
    "chexpert":   ("CheXpert-v1.0/preprocessed224", "CheXpert-v1.0/preprocessed"),
    "nih_cxr14":  ("CXR14/preprocessed224", "CXR14/preprocessed"),
    "padchest":   ("preprocessed224", "preprocessed"),
    "vindr_cxr":  ("preprocessed224", "preprocessed"),
    "vindr_pcxr": ("preprocessed224", "preprocessed"),
}


def main_preprocess_cxr(global_config_path: str):
    cfg = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "preprocess_cxr")
    n_done = 0
    for site, scfg in cfg["cxr"]["sites"].items():
        if not scfg.get("enabled", True):
            continue
        sub224, sub512 = _OUT_SUBPATH[site]
        try:
            resize_tree(
                raw_root=scfg.get("raw_image_root"),
                out_root_224=os.path.join(scfg["image_root"], sub224),
                out_root_512=os.path.join(scfg["image_root"], sub512),
                tag=f"_cxr/{site}",
            )
            n_done += 1
        except MissingInput as e:
            pass
    append_status(status, f"chest preprocessing ran for {n_done} site(s)")
