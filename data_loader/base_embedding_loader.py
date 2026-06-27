"""
data_loader/base_embedding_loader.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively


class BaseEmbeddingDataset(Dataset):
    """
    Args:
        cfg_path    : path to config.yaml
        manifest_csv: absolute path to the pool manifest CSV
        resolution  : target resolution passed to _resolve_path (default 224)
        label_cols  : if provided, only these columns are returned in 'labels'
    """

    def __init__(
        self,
        cfg_path: str,
        manifest_csv: str,
        resolution: int = 224,
        label_cols: Optional[List[str]] = None,
    ):
        self.params     = read_config(cfg_path)
        self.resolution = resolution
        self.label_cols = label_cols

        df = read_csv_defensively(manifest_csv)
        self.records: List[Dict[str, Any]] = df.reset_index(drop=True).to_dict("records")

        print(
            f"[{type(self).__name__}] manifest={os.path.basename(manifest_csv)} | "
            f"resolution={resolution} | {len(self.records)} cases"
        )


    def _resolve_path(self, row: Dict[str, Any]) -> str:
        """Return absolute on-disk path for this manifest row.
        Must be overridden by every concrete subclass."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement _resolve_path(row)."
        )


    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        path = self._resolve_path(row)

        try:
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, OSError) as e:
            raise RuntimeError(
                f"[{type(self).__name__}] Cannot open image at {path} "
                f"(case_id={row.get('case_id')}): {e}"
            )

        item: Dict[str, Any] = {
            "case_id":  str(row.get("case_id", "")),
            "image":    image,
            "dataset":  str(row.get("dataset", "")),
            "modality": str(row.get("modality", "")),
            "split":    str(row.get("split", "")),
        }

        # Attach requested label columns as a dict (float, NaN preserved)
        if self.label_cols:
            item["labels"] = {
                col: float(row[col]) if col in row and not _is_nan(row[col]) else float("nan")
                for col in self.label_cols
            }

        return item


def _is_nan(v: Any) -> bool:
    try:
        return bool(np.isnan(float(v)))
    except (TypeError, ValueError):
        return False



def embedding_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "case_ids":  [b["case_id"]  for b in batch],
        "images":    [b["image"]    for b in batch],
        "datasets":  [b["dataset"]  for b in batch],
        "modalities":[b["modality"] for b in batch],
        "splits":    [b["split"]    for b in batch],
    }
    if "labels" in batch[0]:
        # Merge per-sample label dicts into per-finding lists
        all_keys = list(batch[0]["labels"].keys())
        out["labels"] = {k: [b["labels"][k] for b in batch] for k in all_keys}
    return out
