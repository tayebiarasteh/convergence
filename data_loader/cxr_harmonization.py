"""
data_loader/cxr_harmonization.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

from data_loader.build_utils import binarize_presence


CANONICAL_CXR_FINDINGS: List[str] = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
    "no_finding", "pleural_effusion", "pleural_other", "pneumonia",
    "pneumothorax", "support_devices",
]


MIMIC_LABEL_MAP: Dict[str, str] = {
    "atelectasis": "atelectasis",
    "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation",
    "edema": "edema",
    "enlarged_cardiomediastinum": "enlarged_cardiomediastinum",
    "fracture": "fracture",
    "lung_lesion": "lung_lesion",
    "lung_opacity": "lung_opacity",
    "no_finding": "no_finding",
    "pleural_effusion": "pleural_effusion",
    "pleural_other": "pleural_other",
    "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax",
    "support_devices": "support_devices",
}

# CheXpert master list carries all 14 findings under the same snake_case names
# as MIMIC, so the map is identical.
CHEXPERT_LABEL_MAP: Dict[str, str] = dict(MIMIC_LABEL_MAP)

VINDR_CXR_LABEL_MAP: Dict[str, str] = {
    "Cardiomegaly": "cardiomegaly",
    "Pleural effusion": "pleural_effusion",
    "Pneumonia": "pneumonia",
    "Atelectasis": "atelectasis",
    "No finding": "no_finding",
    "Consolidation": "consolidation",
    "Edema": "edema",
    "Pneumothorax": "pneumothorax",
    "Pleural thickening": "pleural_other",
    "Lung Opacity": "lung_opacity",
    "Nodule/Mass": "lung_lesion",
    # Two native fracture columns both fold into canonical fracture.
    "Rib fracture": "fracture",
    "Clavicle fracture": "fracture",
}

NIH_CXR14_LABEL_MAP: Dict[str, str] = {
    "cardiomegaly": "cardiomegaly",
    "effusion": "pleural_effusion",
    "pneumonia": "pneumonia",
    "atelectasis": "atelectasis",
    "no_finding": "no_finding",
    "consolidation": "consolidation",
    "pneumothorax": "pneumothorax",
    "pleural_thickening": "pleural_other",
    "edema": "edema",
    "nodule": "lung_lesion",
    "mass": "lung_lesion",
}

PADCHEST_LABEL_MAP: Dict[str, str] = {
    "cardiomegaly": "cardiomegaly",
    "pleural_effusion": "pleural_effusion",
    "pneumonia": "pneumonia",
    "atelectasis": "atelectasis",
    "no_finding": "no_finding",
    "consolidation": "consolidation",
    "pneumothorax": "pneumothorax",
    "pleural_thickening": "pleural_other",
    "nodule_mass": "lung_lesion",
    "pulmonary_edema": "edema",
    "pacemaker": "support_devices",
}

VINDR_PCXR_LABEL_MAP: Dict[str, str] = {
    "No finding": "no_finding",
    "Pneumonia": "pneumonia",
}


EXTENDED_MAPS: Dict[str, Dict[str, str]] = {
    "vindr_cxr": {
        "Aortic enlargement": "aortic_enlargement",
        "Calcification": "calcification",
        "Emphysema": "emphysema",
        "Enlarged PA": "enlarged_pa",
        "ILD": "ild",
        "Infiltration": "infiltration",
        "Lung cavity": "lung_cavity",
        "Lung cyst": "lung_cyst",
        "Mediastinal shift": "mediastinal_shift",
        "Pulmonary fibrosis": "pulmonary_fibrosis",
        "Other lesion": "other_lesion",
        "COPD": "copd",
        "Lung tumor": "lung_tumor",
        "Tuberculosis": "tuberculosis",
        "Other diseases": "other_diseases",
    },
    "nih_cxr14": {
        "infiltration": "infiltration",
        "emphysema": "emphysema",
        "fibrosis": "pulmonary_fibrosis",
        "hernia": "hernia",
    },
    "padchest": {
        "pulmonary_fibrosis": "pulmonary_fibrosis",
        "chronic_changes": "chronic_changes",
        "kyphosis": "kyphosis",
        "sternotomy": "sternotomy",
        "infiltrates": "infiltrates",
        "scoliosis": "scoliosis",
        "hernia": "hernia",
        "COPD_signs": "copd_signs",
        "emphysema": "emphysema",
        "aortic_elongation": "aortic_elongation",
        "cavitation": "cavitation",
        "volume_loss": "volume_loss",
        "congestion": "pulmonary_congestion",
        "bronchiectasis": "bronchiectasis",
        "air_trapping": "air_trapping",
    },
    "vindr_pcxr": {
        "Bronchitis/Bronchiolitis": "bronchitis_bronchiolitis",
        "Other disease": "other_disease",
        "Situs inversus": "situs_inversus",
        # The PCXR master misspells this column; key must match exactly.
        "Diagphramatic hernia": "diaphragmatic_hernia",
        "Tuberculosis": "tuberculosis",
        "Congenital emphysema": "congenital_emphysema",
        "CPAM": "cpam",
        "Hyaline membrane disease": "hyaline_membrane_disease",
        "Mediastinal tumor": "mediastinal_tumor",
        "Lung tumor": "lung_tumor",
    },
}


_CHEXPERT_POLICY = dict(positive_code=1, negative_codes=(0, 3), exclude_codes=(2,))
_BINARY_POLICY = dict(positive_code=1, negative_codes=(0,), exclude_codes=())

LABEL_POLICY: Dict[str, dict] = {
    "mimic": _CHEXPERT_POLICY,
    "chexpert": _CHEXPERT_POLICY,
    "vindr_cxr": _BINARY_POLICY,
    "nih_cxr14": _BINARY_POLICY,
    "padchest": _BINARY_POLICY,
    "vindr_pcxr": _BINARY_POLICY,
}

LABEL_MAPS: Dict[str, Dict[str, str]] = {
    "mimic": MIMIC_LABEL_MAP,
    "chexpert": CHEXPERT_LABEL_MAP,
    "vindr_cxr": VINDR_CXR_LABEL_MAP,
    "nih_cxr14": NIH_CXR14_LABEL_MAP,
    "padchest": PADCHEST_LABEL_MAP,
    "vindr_pcxr": VINDR_PCXR_LABEL_MAP,
}


def decode_site_labels(site: str, row: dict) -> Dict[str, float]:
    policy = LABEL_POLICY[site]
    out: Dict[str, float] = {f: float("nan") for f in CANONICAL_CXR_FINDINGS}
    for native_col, canonical in LABEL_MAPS[site].items():
        if native_col not in row:
            continue
        v = binarize_presence(row[native_col], **policy)
        prev = out[canonical]
        if v == 1.0:
            out[canonical] = 1.0
        elif v == 0.0 and prev != 1.0:
            out[canonical] = 0.0
    return out


def decode_extended_labels(site: str, row: dict) -> Dict[str, float]:
    """Return {extended_finding: presence_value} for one row, for sites that
    contribute rare-tail findings. Empty for sites without an extended map."""
    policy = LABEL_POLICY[site]
    emap = EXTENDED_MAPS.get(site, {})
    out: Dict[str, float] = {}
    for native_col, ext_name in emap.items():
        if native_col in row:
            out[ext_name] = binarize_presence(row[native_col], **policy)
    return out



def _res_dirname(resolution: int) -> str:
    """Preprocessed sibling-folder name for a resolution. 224 -> preprocessed224;
    anything else (512) -> preprocessed, matching the existing on-disk trees."""
    return "preprocessed224" if int(resolution) == 224 else "preprocessed"


def _split_subdir(split: str) -> str:
    """VinDr-CXR and VinDr-PCXR store train and valid under train/, test under
    test/."""
    return "test" if str(split) == "test" else "train"


def resolve_cxr_image_path(
    dataset: str,
    image_root: str,
    image_key: str,
    resolution: int = 224,
    split: Optional[str] = None,
    image_subdir: Optional[str] = None,
) -> str:
    res = _res_dirname(resolution)
    key = str(image_key)

    if dataset == "mimic":
        # jpg_rel_path contains 'files/'; swap to the resolution folder.
        token = "preprocessed224/" if int(resolution) == 224 else "preprocessed/"
        return os.path.join(image_root, key.replace("files/", token))

    if dataset == "chexpert":
        # jpg_rel_path begins 'CheXpert-v1.0/'; insert the resolution folder.
        token = ("CheXpert-v1.0/preprocessed224/" if int(resolution) == 224
                 else "CheXpert-v1.0/preprocessed/")
        return os.path.join(image_root, key.replace("CheXpert-v1.0/", token, 1))

    if dataset == "nih_cxr14":
        # img_rel_path is relative to CXR14/<res>/.
        return os.path.join(image_root, "CXR14", res, key)

    if dataset == "padchest":
        if image_subdir is None or str(image_subdir) in ("", "nan", "None"):
            sub = ""
        else:
            sub_raw = str(image_subdir)
            try:
                f = float(sub_raw)
                sub = str(int(f)) if f == int(f) else sub_raw
            except (ValueError, TypeError):
                sub = sub_raw
        return os.path.join(image_root, res, sub, key)

    if dataset in ("vindr_cxr", "vindr_pcxr"):
        # <res>/<train|test>/<image_id>.jpg.
        fname = key if key.endswith(".jpg") else f"{key}.jpg"
        return os.path.join(image_root, res, _split_subdir(split), fname)

    raise KeyError(f"unknown cxr dataset '{dataset}'")


VIEW_KEEP: Dict[str, Optional[List[str]]] = {
    "mimic": ["PA", "AP"],
    "chexpert": ["Frontal"],
    "vindr_cxr": None,
    "nih_cxr14": None,
    "padchest": ["PA", "AP", "AP_horizontal"],
    "vindr_pcxr": None,
}

# Master-list column carrying the view, per site (None -> no filtering).
VIEW_COL: Dict[str, Optional[str]] = {
    "mimic": "view",
    "chexpert": "view",
    "vindr_cxr": None,
    "nih_cxr14": None,
    "padchest": "view",
    "vindr_pcxr": None,
}

# Master-list column carrying the raw image key, per site.
IMAGE_KEY_COL: Dict[str, str] = {
    "mimic": "jpg_rel_path",
    "chexpert": "jpg_rel_path",
    "vindr_cxr": "image_id",
    "nih_cxr14": "img_rel_path",
    "padchest": "ImageID",
    "vindr_pcxr": "image_id",
}

# Secondary path token column (PadChest only).
IMAGE_SUBDIR_COL: Dict[str, Optional[str]] = {
    "mimic": None, "chexpert": None, "vindr_cxr": None,
    "nih_cxr14": None, "padchest": "ImageDir", "vindr_pcxr": None,
}

CXR_SITES: List[str] = [
    "mimic", "chexpert", "vindr_cxr", "nih_cxr14", "padchest", "vindr_pcxr",
]