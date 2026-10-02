"""
ontology/build_ontology_geometry.py
Created on June 13, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS
from Inference.resume_utils import write_csv_atomic, MissingInput, append_status, status_path
from data_loader.build_utils import manifest_exists_and_valid, write_manifest


FINDING_TO_ICD10: Dict[str, str] = {
    "atelectasis":               "J9811",
    "cardiomegaly":              "I517",
    "consolidation":             "J189",
    "edema":                     "J810",
    "enlarged_cardiomediastinum": "R931",
    "fracture":                  "S2200XA",
    "lung_lesion":               "R911",
    "lung_opacity":              "R918",
    "no_finding":                "Z0000",
    "pleural_effusion":          "J90",
    "pleural_other":             "J929",
    "pneumonia":                 "J189",
    "pneumothorax":              "J939",
    "support_devices":           "Z9689",
}


def _normalize_code(code: str) -> str:
    return code.replace(".", "").strip().upper()


def _lcp_length(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def _hierarchy_distance(code_a: str, code_b: str) -> int:
    a, b = _normalize_code(code_a), _normalize_code(code_b)
    lcp  = _lcp_length(a, b)
    return (len(a) - lcp) + (len(b) - lcp)


def _parse_icd10_codes(txt_path: str) -> set:
    codes = set()
    if not os.path.exists(txt_path):
        return codes
    with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 15:
                continue
            code_raw = line[6:13].strip()
            if code_raw:
                codes.add(_normalize_code(code_raw))
    return codes


def _build_icd10_distances(valid_codes: set) -> pd.DataFrame:
    findings = CANONICAL_CXR_FINDINGS
    rows = []
    for fa in findings:
        code_a = FINDING_TO_ICD10.get(fa, "")
        norm_a = _normalize_code(code_a)
        if valid_codes and norm_a not in valid_codes:
            norm_a = norm_a[:3]
        for fb in findings:
            code_b = FINDING_TO_ICD10.get(fb, "")
            norm_b = _normalize_code(code_b)
            if valid_codes and norm_b not in valid_codes:
                norm_b = norm_b[:3]
            dist = _hierarchy_distance(norm_a, norm_b)
            rows.append({
                "finding_a": fa,
                "finding_b": fb,
                "code_a":    code_a,
                "code_b":    code_b,
                "icd10_distance": dist,
            })
    return pd.DataFrame(rows)


def _build_comorbidity(pool_csv: str) -> pd.DataFrame:
    pool = read_csv_defensively(pool_csv)
    mimic = pool[pool["dataset"] == "mimic"].copy()

    findings = CANONICAL_CXR_FINDINGS
    presence: Dict[str, set] = {}
    for f in findings:
        if f not in mimic.columns:
            presence[f] = set()
            continue
        col = pd.to_numeric(mimic[f], errors="coerce")
        pos_idx = set(mimic.index[col == 1.0].tolist())
        presence[f] = pos_idx

    rows = []
    for fa in findings:
        for fb in findings:
            a_set = presence[fa]
            b_set = presence[fb]
            union = len(a_set | b_set)
            inter = len(a_set & b_set)
            jaccard = inter / union if union > 0 else 0.0
            rows.append({
                "finding_a":       fa,
                "finding_b":       fb,
                "n_positive_a":    len(a_set),
                "n_positive_b":    len(b_set),
                "n_co_occurrence": inter,
                "n_union":         union,
                "jaccard":         round(jaccard, 6),
            })
    return pd.DataFrame(rows)


def main_build_ontology_geometry(global_config_path: str):
    params   = read_config(global_config_path)
    cfg      = params["Convergence"]
    ocfg     = cfg["ontology"]
    pool_csv = cfg["cxr"]["pool_manifest_csv"]

    icd10_txt   = ocfg["icd10_order_txt"]
    icd10_csv   = ocfg["icd10_distance_csv"]
    comorbid_csv = ocfg["comorbidity_csv"]

    valid_codes = _parse_icd10_codes(icd10_txt)
    icd10_df    = _build_icd10_distances(valid_codes)
    os.makedirs(os.path.dirname(icd10_csv), exist_ok=True)
    write_csv_atomic(icd10_df, icd10_csv)

    if not os.path.exists(pool_csv):
        raise MissingInput(
            f"[build_ontology] the CXR pool manifest is absent at {pool_csv}, so the comorbidity "
            f"reference cannot be built. Run main_build_cxr_pool before this stage.")
    else:
        comorbid_df = _build_comorbidity(pool_csv)
        os.makedirs(os.path.dirname(comorbid_csv), exist_ok=True)
        write_csv_atomic(comorbid_df, comorbid_csv)
