"""
data_loader/build_mammo_pool.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import ast
import os
from typing import List

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import (
    assert_unique_case_ids, cap_per_group, finalize_manifest,
    read_csv_defensively,
)


def _parse_categories(raw) -> List[str]:
    if pd.isna(raw):
        return []
    try:
        val = ast.literal_eval(str(raw))
        return [str(v) for v in val] if isinstance(val, (list, tuple)) else [str(val)]
    except (ValueError, SyntaxError):
        return [str(raw)]


def _resolve_mammo_path(image_root: str, study_id: str,
                         image_id: str, resolution: int = 224) -> str:
    res = "preprocessed224" if resolution == 224 else "preprocessed"
    return os.path.join(image_root, res, str(study_id), f"{image_id}.png")


def main_build_mammo_pool(global_config_path: str) -> str:
    params = read_config(global_config_path)
    cfg    = params["Convergence"]
    mcfg   = cfg["mammo"]

    if not mcfg.get("enabled", True):
        print("[build_mammo_pool] Mammo disabled in config; nothing to do.")
        return ""

    image_root    = mcfg["image_root"]
    findings_csv  = mcfg["findings_csv"]
    out_csv       = mcfg["pool_manifest_csv"]
    cap           = int(mcfg.get("cases_per_finding", 500))
    seed          = int(cfg.get("seed", 42))
    category_map  = mcfg.get("category_map", {})
    usable        = mcfg.get("usable_findings",
                             list(category_map.values()) if category_map else [])

    if not os.path.exists(findings_csv):
        raise FileNotFoundError(
            f"[build_mammo_pool] findings_csv not found: {findings_csv}"
        )

    ann = read_csv_defensively(findings_csv)
    ann["_cats"] = ann["finding_categories"].apply(_parse_categories)

    images = ann.drop_duplicates("image_id")[["image_id", "study_id"]].copy()

    for cat_str, finding in category_map.items():
        if finding not in usable:
            continue
        pos_ids = ann[ann["_cats"].apply(lambda cs: cat_str in cs)]["image_id"]
        images[finding] = images["image_id"].isin(pos_ids).astype(float)

    no_finding_ids = ann[ann["_cats"].apply(
        lambda cs: cs == ["No Finding"]
    )]["image_id"]
    images["no_finding"] = images["image_id"].isin(no_finding_ids).astype(float)

    # Verify images exist at 224px
    def _exists(row):
        return os.path.exists(
            _resolve_mammo_path(image_root, row["study_id"], row["image_id"])
        )

    exists_mask = images.apply(_exists, axis=1)
    n_miss = (~exists_mask).sum()
    if n_miss:
        print(f"[build_mammo_pool] {n_miss} images not found at 224px; dropping.")
    images = images[exists_mask].copy()

    # For the discriminant control we cap total images (not per-finding balance)
    total_cap = cap * len(usable)
    if len(images) > total_cap:
        images = images.sample(n=total_cap, random_state=seed)

    finding_cols = [f for f in list(category_map.values()) + ["no_finding"]
                    if f in images.columns]

    pool = pd.DataFrame({
        "case_id":      [f"mammo__vindr__{row['image_id']}"
                         for _, row in images.iterrows()],
        "dataset":      "mammo_vindr",
        "modality":     "mammo",
        "split":        "train",
        "image_key":    images["image_id"].apply(lambda x: f"{x}.png").values,
        "image_subdir": images["study_id"].values,
        **{f: images[f].values for f in finding_cols},
    })

    pool = finalize_manifest(pool, label_cols=finding_cols)
    assert_unique_case_ids(pool)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pool.to_csv(out_csv, index=False)
    for f in finding_cols:
        n_pos = int((pool[f] == 1).sum())
        print(f"  {f}: {n_pos} positive")
    return out_csv
