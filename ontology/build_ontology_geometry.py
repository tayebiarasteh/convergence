"""
ontology/build_ontology_geometry.py
Created on May 26, 2026

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


FINDING_TO_ICD10: Dict[str, str] = {
    "atelectasis":               "J9811",   # J98.11 Atelectasis
    "cardiomegaly":              "I517",    # I51.7  Cardiomegaly
    "consolidation":             "J189",    # J18.9  Pneumonia, unspecified (consolidation often coded here)
    "edema":                     "J810",    # J81.0  Acute pulmonary oedema
    "enlarged_cardiomediastinum": "R931",   # R93.1  Abnormal findings on imaging of heart
    "fracture":                  "S2200XA", # S22.00XA Fracture of unspecified thoracic vertebra / rib
    "lung_lesion":               "R911",    # R91.1  Solitary pulmonary nodule
    "lung_opacity":              "R918",    # R91.8  Other nonspecific findings of diagnostic imaging
    "no_finding":                "Z0000",   # Z00.00 General adult medical examination, no abnormal findings
    "pleural_effusion":          "J90",     # J90    Pleural effusion
    "pleural_other":             "J929",    # J92.9  Pleural plaque, unspecified
    "pneumonia":                 "J189",    # J18.9  Pneumonia, unspecified (same as consolidation; close by design)
    "pneumothorax":              "J939",    # J93.9  Pneumothorax, unspecified
    "support_devices":           "Z9689",   # Z96.89 Presence of other specified functional implants
}


def _normalize_code(code: str) -> str:
    """Strip period and whitespace to get the raw code string used for LCP."""
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
        print(f"[build_ontology] ICD-10 order file not found: {txt_path}. "
              f"Skipping code validation.")
        return codes
    with open(txt_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if len(line) < 15:
                continue
            code_raw = line[6:13].strip()   # cols 7-13 (0-indexed 6-12)
            if code_raw:
                codes.add(_normalize_code(code_raw))
    print(f"[build_ontology] Parsed {len(codes)} ICD-10-CM codes from order file.")
    return codes


def _build_icd10_distances(valid_codes: set) -> pd.DataFrame:
    """Build 14x14 pairwise ICD-10 hierarchy distance matrix (long format)."""
    findings = CANONICAL_CXR_FINDINGS
    rows = []
    for fa in findings:
        code_a = FINDING_TO_ICD10.get(fa, "")
        norm_a = _normalize_code(code_a)
        if valid_codes and norm_a not in valid_codes:
            # Shorten to category (3 chars) as fallback
            norm_a = norm_a[:3]
            if norm_a not in valid_codes:
                print(f"  WARNING: ICD-10 code for '{fa}' ({code_a}) "
                      f"not found in order file; using {norm_a} for distance.")
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
    """Build 14x14 pairwise Jaccard comorbidity matrix from MIMIC pool labels."""
    pool = read_csv_defensively(pool_csv)
    mimic = pool[pool["dataset"] == "mimic"].copy()

    findings = CANONICAL_CXR_FINDINGS
    # Build binary presence arrays
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
    icd10_df.to_csv(icd10_csv, index=False)

    if not os.path.exists(pool_csv):
        print(f"  Pool manifest not found: {pool_csv}. "
              f"Skipping comorbidity (run build_cxr_pool first).")
    else:
        comorbid_df = _build_comorbidity(pool_csv)
        os.makedirs(os.path.dirname(comorbid_csv), exist_ok=True)
        comorbid_df.to_csv(comorbid_csv, index=False)
