"""
data_loader/cxr_embedding_loader.py

CXR embedding loader for all six sites in the convergence pool.

_resolve_path delegates entirely to cxr_harmonization.resolve_cxr_image_path
so path logic lives in exactly one place and is guaranteed consistent with the
pool builder.

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Any, Dict, List, Optional

from data_loader.base_embedding_loader import BaseEmbeddingDataset
from data_loader.cxr_harmonization import resolve_cxr_image_path


class CXREmbeddingDataset(BaseEmbeddingDataset):
    """Loads CXR images from the pool manifest for embedding extraction.

    Args (additional to BaseEmbeddingDataset):
        site_roots : dict mapping dataset key -> image_root. If None, roots are
                     read from Convergence.cxr.sites.<dataset>.image_root in
                     config. Passing site_roots explicitly allows the caller to
                     override roots without touching config, which is useful for
                     cross-site transfer experiments in E7.
    """

    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
        site_roots: Optional[Dict[str, str]] = None,
    ):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        sites_cfg = self.params["Convergence"]["cxr"]["sites"]
        self._roots: Dict[str, str] = site_roots or {
            site: scfg["image_root"] for site, scfg in sites_cfg.items()
        }

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        dataset = str(row.get("dataset", ""))
        root    = self._roots.get(dataset)
        if not root:
            raise KeyError(
                f"[CXREmbeddingDataset] No image_root for dataset '{dataset}'. "
                f"Known datasets: {list(self._roots.keys())}"
            )
        return resolve_cxr_image_path(
            dataset=dataset,
            image_root=root,
            image_key=str(row.get("image_key", "")),
            resolution=self.resolution,
            split=str(row.get("split", "")) if row.get("split") else None,
            image_subdir=str(row["image_subdir"])
            if row.get("image_subdir") and str(row.get("image_subdir")) != "nan"
            else None,
        )
