"""
data_loader/modality_embedding_loaders.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Any, Dict, List, Optional

from data_loader.base_embedding_loader import BaseEmbeddingDataset, embedding_collate_fn
from data_loader.cxr_embedding_loader import CXREmbeddingDataset


class HistoEmbeddingDataset(BaseEmbeddingDataset):

    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
    ):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        hcfg = self.params["Convergence"]["histo"]
        self._pcam_root = hcfg["pcam"]["h5_dir"]
        self._nct_root  = hcfg["nct_crc"]["root"]
        # Quilt-1M shares the histo modality; its root is under quilt config.
        qcfg = self.params["Convergence"].get("quilt", {})
        self._quilt_root = qcfg.get("root", "")

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        dataset = str(row.get("dataset", ""))
        key     = str(row.get("image_key", ""))
        if dataset == "pcam":
            return os.path.join(self._pcam_root, key)
        if dataset == "nct_crc":
            return os.path.join(self._nct_root, key)
        if dataset == "quilt":
            return os.path.join(self._quilt_root, key)
        raise KeyError(f"[HistoEmbeddingDataset] Unknown histo dataset: '{dataset}'")



class FundusEmbeddingDataset(BaseEmbeddingDataset):

    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
    ):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        self._image_root = self.params["Convergence"]["fundus"]["image_root"]

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        res    = "preprocessed224" if self.resolution == 224 else "preprocessed"
        subdir = str(row.get("image_subdir", "")) if row.get("image_subdir") else ""
        key    = str(row.get("image_key", ""))
        return os.path.join(self._image_root, res, subdir, key)


class DermEmbeddingDataset(BaseEmbeddingDataset):

    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
    ):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        self._image_root = self.params["Convergence"]["derm"]["image_root"]

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        res    = "preprocessed224" if self.resolution == 224 else "preprocessed"
        subdir = str(row.get("image_subdir", "")) if row.get("image_subdir") else ""
        key    = str(row.get("image_key", ""))
        # Ensure .jpg extension (derm manifests store bare image_id)
        if not key.lower().endswith(".jpg"):
            key = key + ".jpg"
        return os.path.join(self._image_root, res, subdir, key)


class MammoEmbeddingDataset(BaseEmbeddingDataset):

    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
    ):
        super().__init__(cfg_path, manifest_csv, resolution, label_cols)
        self._image_root = self.params["Convergence"]["mammo"]["image_root"]

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        res      = "preprocessed224" if self.resolution == 224 else "preprocessed"
        study_id = str(row.get("image_subdir", "")) if row.get("image_subdir") else ""
        key      = str(row.get("image_key", ""))
        return os.path.join(self._image_root, res, study_id, key)



LOADER_REGISTRY: Dict[str, type] = {
    "cxr":    CXREmbeddingDataset,
    "histo":  HistoEmbeddingDataset,
    "fundus": FundusEmbeddingDataset,
    "derm":   DermEmbeddingDataset,
    "mammo":  MammoEmbeddingDataset,
}


def get_embedding_loader(modality: str) -> type:
    """Return the loader class for the given modality key."""
    if modality not in LOADER_REGISTRY:
        raise KeyError(
            f"Unknown modality '{modality}'. "
            f"Expected one of: {sorted(LOADER_REGISTRY.keys())}"
        )
    return LOADER_REGISTRY[modality]