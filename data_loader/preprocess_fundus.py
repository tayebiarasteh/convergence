"""
data_loader/preprocess_fundus.py
Created on June 13, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

from config.serde import read_config
from data_loader.preprocess_utils import resize_tree
from Inference.resume_utils import (MissingInput, append_status, check_build_params,
                                    status_path, write_build_params, write_csv_atomic)


def main_preprocess_fundus(global_config_path: str):
    cfg = read_config(global_config_path)["Convergence"]
    status = status_path(cfg, "preprocess_fundus")
    fcfg = cfg.get("fundus", {})
    if not fcfg.get("enabled", False):
        return
    image_root = fcfg["image_root"]
    n_done = 0
    for name, scfg in fcfg.get("sources", {}).items():
        tag = scfg.get("source_tag", name)
        try:
            resize_tree(
                raw_root=scfg.get("raw_dir"),
                out_root_224=os.path.join(image_root, "preprocessed224", tag),
                out_root_512=os.path.join(image_root, "preprocessed", tag),
                tag=f"_fundus/{name}",
            )
            n_done += 1
        except MissingInput as e:
            pass
    append_status(status, f"fundus preprocessing ran for {n_done} source(s)")
