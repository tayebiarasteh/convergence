"""
data_loader/preprocess_fundus.py

Reproduction record for fundus preprocessing.

Fundus images (APTOS-2019, Messidor-2) are already preprocessed to 224 and 512 px
under source-tagged trees (preprocessed224/<source_tag>/, preprocessed/<source_tag>/),
so this script does not run as part of the pipeline. It documents and, if pointed
at the raw images, reproduces the resize convention used by the loaders.

To reproduce a source, add a `raw_dir` key to that source's config block under
Convergence.fundus.sources.<name>; without it the source is skipped.

Run:
    python -m data_loader.preprocess_fundus

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
        print("[preprocess_fundus] fundus disabled in config; nothing to do.")
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


if __name__ == "__main__":
    main_preprocess_fundus(
        "/home/homesOnMaster/sarasteh/Documents/Repositories/convergence/config/config.yaml"
    )
