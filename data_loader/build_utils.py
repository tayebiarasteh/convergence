"""
data_loader/build_utils.py
Created on June 13, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from Inference.resume_utils import write_csv_atomic


CORE_COLUMNS: List[str] = [
    "case_id", "dataset", "modality", "split", "image_key", "image_subdir",
]


AGE_BAND_EDGES = [0, 40, 60, 80, 200]
AGE_BAND_LABELS = ["<40", "40-60", "60-80", "80+"]


def age_band(values) -> pd.Series:
    band = pd.cut(pd.to_numeric(pd.Series(values), errors="coerce"),
                  bins=AGE_BAND_EDGES, labels=AGE_BAND_LABELS)
    return band.astype(object).where(band.notna(), None)


def finalize_manifest(
    df: pd.DataFrame,
    label_cols: Sequence[str] = (),
    meta_cols: Sequence[str] = (),
) -> pd.DataFrame:
    df = df.copy()
    declared = list(CORE_COLUMNS) + list(meta_cols) + list(label_cols)
    for col in declared:
        if col not in df.columns:
            df[col] = np.nan
    rest = [c for c in df.columns if c not in declared]
    return df[declared + rest]


def read_csv_defensively(path: str, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


def binarize_presence(
    value,
    positive_code: int = 1,
    negative_codes: Sequence[int] = (0,),
    exclude_codes: Sequence[int] = (),
) -> float:
    if pd.isna(value):
        return np.nan
    try:
        v = int(round(float(value)))
    except (TypeError, ValueError):
        return np.nan
    if v == int(positive_code):
        return 1.0
    if v in {int(c) for c in exclude_codes}:
        return np.nan
    if v in {int(c) for c in negative_codes}:
        return 0.0
    return np.nan


def cap_per_group(
    df: pd.DataFrame,
    group_col: str,
    cap: int,
    seed: int = 42,
) -> pd.DataFrame:
    if cap is None or cap <= 0:
        return df.reset_index(drop=True)
    parts = []
    for _, grp in df.groupby(group_col, sort=False):
        parts.append(grp.sample(n=cap, random_state=seed) if len(grp) > cap else grp)
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    return out.reset_index(drop=True)


def balanced_presence_sample(
    df: pd.DataFrame,
    finding_col: str,
    n_per_class: int,
    seed: int = 42,
) -> pd.DataFrame:
    rng_state = int(seed)
    pos = df[df[finding_col] == 1.0]
    neg = df[df[finding_col] == 0.0]
    n = min(n_per_class, len(pos), len(neg))
    if n == 0:
        return df.iloc[0:0]
    out = pd.concat(
        [pos.sample(n=n, random_state=rng_state),
         neg.sample(n=n, random_state=rng_state)],
        ignore_index=True,
    )
    return out.reset_index(drop=True)


def patient_split_leak(df: pd.DataFrame, split_col: str = "split",
                       subject_col: str = "subject_id",
                       dataset_col: str = "dataset") -> Dict:
    out = {"n_leaking_patients": 0, "leaking": [], "n_unidentified": 0, "unidentified_sites": {}}
    if split_col not in df.columns or subject_col not in df.columns:
        return out
    sid = df[subject_col].astype(str).str.strip()
    known = df[subject_col].notna() & (sid.str.lower() != "nan") & (sid != "")
    out["n_unidentified"] = int((~known).sum())
    if out["n_unidentified"] and dataset_col in df.columns:
        out["unidentified_sites"] = (df.loc[~known, dataset_col].astype(str)
                                     .value_counts().to_dict())
    have = df[known]
    if have.empty:
        return out
    key = ((have[dataset_col].astype(str) + "__" + sid[known]) if dataset_col in have.columns
           else sid[known])
    splits = have[split_col].astype(str).str.lower().replace({"val": "valid"})
    spread = (pd.DataFrame({"key": key.values, "split": splits.values})
              .groupby("key")["split"].nunique())
    bad = spread[spread > 1]
    out["n_leaking_patients"] = int(len(bad))
    out["leaking"] = [str(k) for k in bad.index[:10]]
    return out


def assert_patient_disjoint(df: pd.DataFrame, label: str, **kwargs) -> Dict:
    rep = patient_split_leak(df, **kwargs)
    if rep["n_leaking_patients"]:
        raise ValueError(
            f"[splits] {label}: {rep['n_leaking_patients']} patient(s) appear in more than one "
            f"split, for example {rep['leaking']}. Every score fitted on this split is inflated.")
    return rep


def assert_unique_case_ids(df: pd.DataFrame) -> None:
    dups = df["case_id"][df["case_id"].duplicated()].unique()
    if len(dups):
        raise ValueError(
            f"{len(dups)} duplicate case_id values, e.g. {list(dups[:5])}. "
            f"Fix the per-dataset key before writing the manifest."
        )


def manifest_exists_and_valid(path: str, expected: dict, owner: str,
                              legacy: dict = None) -> bool:
    import os
    from Inference.resume_utils import IncompatibleArtifact, check_build_params
    if not os.path.exists(path):
        return False
    try:
        check_build_params(path, expected, owner=owner, raise_on_mismatch=True, legacy=legacy)
    except IncompatibleArtifact as e:
        return False
    try:
        import pandas as pd
        if len(pd.read_csv(path, nrows=1)) == 0:
            return False
    except Exception:
        return False
    return True


def write_manifest(df, path: str, expected: dict, owner: str,
                      status_file: str = None, note: str = "") -> str:
    from Inference.resume_utils import append_status, write_build_params, write_csv_atomic
    if df is None or len(df) == 0:
        raise ValueError(f"[{owner}] refusing to write a zero-row manifest to {path}.")
    write_csv_atomic(df, path)
    write_build_params(path, expected)
    if status_file:
        append_status(status_file, f"{owner}: {len(df)} rows -> {path}. {note}".strip())
    return path
