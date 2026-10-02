"""
data_loader/cxr_embedding_loader.py
Created on June 15, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Any, Dict, List, Optional

from data_loader.base_embedding_loader import BaseEmbeddingDataset
from data_loader.cxr_harmonization import resolve_cxr_image_path


class CXREmbeddingDataset(BaseEmbeddingDataset):

    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
        site_roots: Optional[Dict[str, str]] = None,
    ):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        conv = self.params["Convergence"]
        sites_cfg = conv["cxr"]["sites"]
        roots = {site: scfg["image_root"] for site, scfg in sites_cfg.items()}
        for extra in ("taix", "rexgradient"):
            blk = conv.get(extra, {}) or {}
            if blk.get("image_root"):
                roots[extra] = blk["image_root"]
        self._roots: Dict[str, str] = site_roots or roots

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        dataset = str(row.get("dataset", ""))
        root    = self._roots.get(dataset)
        if not root:
            raise KeyError(
                f"[CXREmbeddingDataset] No image_root for dataset '{dataset}'. "
                f"Known datasets: {list(self._roots.keys())}"
            )
        raw_subdir = row.get("image_subdir", None)
        subdir_str = str(raw_subdir) if raw_subdir is not None else ""
        subdir = None if subdir_str.lower() in ("", "nan", "none") else subdir_str
        return resolve_cxr_image_path(
            dataset=dataset,
            image_root=root,
            image_key=str(row.get("image_key", "")),
            resolution=self.resolution,
            split=str(row.get("split", "")) if row.get("split") else None,
            image_subdir=subdir,
        )
