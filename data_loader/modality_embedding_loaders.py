"""
data_loader/modality_embedding_loaders.py

Thin per-modality embedding loaders and the loader registry.

Each loader subclasses BaseEmbeddingDataset and overrides only _resolve_path.
The CXR loader lives in cxr_embedding_loader.py; the registry here includes it.

Path conventions (must mirror what each pool builder writes as image_key and
image_subdir):

  Histo / PCam:
    image_root = h5_dir (pcam) or nct_root (nct_crc)
    image_key  = relative path from image_root (e.g. patches/pcam_test_N.png
                 for PCam, or NCT-CRC-HE-100K/ADI/file.tif for NCT-CRC)
    Resolved:  image_root / image_key

  Fundus:
    image_root = Convergence.fundus.image_root
    image_key  = <image_id>.<ext>   (e.g. 0001.png)
    image_subdir = <source_tag>     (e.g. aptos, messidor)
    Resolved:  image_root / preprocessed224 / image_subdir / image_key

  Derm:
    image_root = Convergence.derm.image_root
    image_key  = <image_id>          (no extension stored)
    image_subdir = <source_tag>      (e.g. isic2019)
    Resolved:  image_root / preprocessed224 / image_subdir / image_key.jpg

  Mammo:
    image_root = Convergence.mammo.image_root
    image_key  = <image_id>.png
    image_subdir = <study_id>
    Resolved:  image_root / preprocessed224 / image_subdir / image_key

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Any, Dict, List, Optional

from data_loader.base_embedding_loader import BaseEmbeddingDataset
from data_loader.cxr_embedding_loader import CXREmbeddingDataset


# ---------------------------------------------------------------------------
# Histo
# ---------------------------------------------------------------------------

class HistoEmbeddingDataset(BaseEmbeddingDataset):
    """Covers both PCam (PNG patches under h5_dir/patches/) and NCT-CRC
    (TIF files under nct_root/NCT-CRC-HE-100K/<cls>/ or CRC-VAL-HE-7K/<cls>/).

    The manifest image_key is already relative to the correct source root
    (h5_dir for PCam, nct_root for NCT-CRC). The loader stores both roots
    and selects by the 'dataset' field in each row.
    """

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

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        dataset = str(row.get("dataset", ""))
        key     = str(row.get("image_key", ""))
        if dataset == "pcam":
            return os.path.join(self._pcam_root, key)
        if dataset == "nct_crc":
            return os.path.join(self._nct_root, key)
        raise KeyError(f"[HistoEmbeddingDataset] Unknown histo dataset: '{dataset}'")


# ---------------------------------------------------------------------------
# Fundus
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Derm
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Mammo
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
