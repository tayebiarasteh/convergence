"""
data_loader/build_utils.py
Created on May 25, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


CORE_COLUMNS: List[str] = [
    "case_id", "dataset", "modality", "split", "image_key", "image_subdir",
]


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
    """Master lists may come from heterogeneous exporters; try UTF-8 then
    latin-1 before giving up. low_memory is disabled by default because these
    wide label tables otherwise trigger mixed-dtype column warnings."""
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)



def new_rng(seed: int) -> np.random.Generator:
    """Single source of data-sampling randomness. Inferential statistics use a
    separate legacy RandomState elsewhere; this is for data construction only."""
    return np.random.default_rng(int(seed))



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
    """Randomly downsample each group to at most `cap` rows. Groups smaller
    than the cap pass through whole. Deterministic under `seed`."""
    if cap is None or cap <= 0:
        return df.reset_index(drop=True)
    parts = []
    for _, grp in df.groupby(group_col, sort=False):
        parts.append(grp.sample(n=cap, random_state=seed) if len(grp) > cap else grp)
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    return out.reset_index(drop=True)


def stratified_sample(
    df: pd.DataFrame,
    strata_cols: Sequence[str],
    n_per_stratum: int,
    seed: int = 42,
) -> pd.DataFrame:
    parts = []
    for _, grp in df.groupby(list(strata_cols), sort=False):
        parts.append(
            grp.sample(n=n_per_stratum, random_state=seed)
            if len(grp) > n_per_stratum else grp
        )
    out = pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]
    return out.reset_index(drop=True)


def balanced_presence_sample(
    df: pd.DataFrame,
    finding_col: str,
    n_per_class: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample an equal number of present (1) and absent (0) rows for one
    finding column, ignoring NaN (unlabeled) rows. Returns up to
    2 * n_per_class rows. Used for per-finding probe and anchor construction."""
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



def make_case_ids(dataset: str, keys: Sequence[str]) -> List[str]:
    out = []
    for k in keys:
        norm = str(k).strip().replace("/", "_").replace(" ", "_")
        out.append(f"{dataset}__{norm}")
    return out


def assert_unique_case_ids(df: pd.DataFrame) -> None:
    """Fail loudly if case_id is not unique; a duplicated id silently corrupts
    every downstream join, so this is checked at build time, not inference."""
    dups = df["case_id"][df["case_id"].duplicated()].unique()
    if len(dups):
        raise ValueError(
            f"{len(dups)} duplicate case_id values, e.g. {list(dups[:5])}. "
            f"Fix the per-dataset key before writing the manifest."
        )
