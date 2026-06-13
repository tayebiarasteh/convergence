"""
data_loader/preprocess_cxr.py

Reproduction record for chest-radiograph preprocessing.

The six CXR datasets are already preprocessed to 224 and 512 px on the system,
so this script does not run as part of the pipeline. It documents and, if
pointed at the raw images, reproduces the resize convention: every raw image is
resized with LANCZOS to a square 224 and 512 target and written to the
preprocessed trees the loaders read (preprocessed224/ and preprocessed/).

To reproduce for a site, add a `raw_image_root` key to that site's config block
pointing at the raw image tree; without it the site is skipped. Output roots
follow each site's on-disk convention:
    MIMIC      <image_root>/preprocessed224 , <image_root>/preprocessed
    CheXpert   <image_root>/CheXpert-v1.0/preprocessed224 , .../preprocessed
    NIH        <image_root>/CXR14/preprocessed224 , .../preprocessed
    PadChest   <image_root>/preprocessed224 , <image_root>/preprocessed
    VinDr-CXR  <image_root>/preprocessed224 , <image_root>/preprocessed
    VinDr-PCXR <image_root>/preprocessed224 , <image_root>/preprocessed

Run:
    python -m data_loader.preprocess_cxr

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


if __name__ == "__main__":
    main_preprocess_cxr(
        "/home/homesOnMaster/sarasteh/Documents/Repositories/convergence/config/config.yaml"
    )
