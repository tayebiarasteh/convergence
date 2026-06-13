"""
data_loader/build_utils.py

Shared construction helpers for every pool builder in the convergence project.

This project compares the representations of many independently trained image
encoders on a shared set of images. The data layer therefore produces flat
"embedding-pool" manifests: one row per image, carrying the fields needed to
(a) resolve the image on disk at a requested resolution, (b) attach harmonized
finding labels and demographics, and (c) thread a stable case_id through every
downstream output (embeddings, alignment scores, fracture tables).

There is deliberately no swap / occlusion / prompt machinery here; those belong
to behavioral-probing pipelines, not to representation alignment. The helpers
below cover only manifest assembly: defensive IO, deterministic sampling,
per-group capping, and presence-label balancing.

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd


# ----- Canonical embedding-manifest schema ---------------------------------
# Core columns are modality-agnostic and present in every pool manifest. Label
# and metadata columns are appended per modality (the builder passes their
# names to finalize_manifest so column order stays deterministic).
#
#   case_id        globally unique, stable id threaded through all outputs
#   dataset        source dataset key (e.g. mimic, padchest, pcam, nct_crc)
#   modality       imaging modality (cxr, histo, fundus, ...)
#   split          original dataset split (train/valid/test) where defined
#   image_key      the raw path token the per-dataset resolver needs
#   image_subdir   optional secondary path token (e.g. PadChest ImageDir); NaN if unused
CORE_COLUMNS: List[str] = [
    "case_id", "dataset", "modality", "split", "image_key", "image_subdir",
]


def finalize_manifest(
    df: pd.DataFrame,
    label_cols: Sequence[str] = (),
    meta_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Order columns as CORE + metadata + labels + any remaining, creating any
    missing core/declared columns as NaN so every manifest is schema-uniform.

    label_cols and meta_cols are declared explicitly by the builder so the
    column order of a manifest never depends on dict insertion accidents.
    """
    df = df.copy()
    declared = list(CORE_COLUMNS) + list(meta_cols) + list(label_cols)
    for col in declared:
        if col not in df.columns:
            df[col] = np.nan
    rest = [c for c in df.columns if c not in declared]
    return df[declared + rest]


# ----- Defensive IO ---------------------------------------------------------

def read_csv_defensively(path: str, **kwargs) -> pd.DataFrame:
    """Master lists may come from heterogeneous exporters; try UTF-8 then
    latin-1 before giving up. low_memory is disabled by default because these
    wide label tables otherwise trigger mixed-dtype column warnings."""
    kwargs.setdefault("low_memory", False)
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", **kwargs)


# ----- Determinism ----------------------------------------------------------

def new_rng(seed: int) -> np.random.Generator:
    """Single source of data-sampling randomness. Inferential statistics use a
    separate legacy RandomState elsewhere; this is for data construction only."""
    return np.random.default_rng(int(seed))


# ----- Label policy ---------------------------------------------------------

def binarize_presence(
    value,
    positive_code: int = 1,
    negative_codes: Sequence[int] = (0,),
    exclude_codes: Sequence[int] = (),
) -> float:
    """Map a raw label cell to a binary presence value.

    Returns 1.0 for positive, 0.0 for an explicit negative, and np.nan for
    cells that are missing, uncertain, or otherwise outside the declared codes
    (so "not labeled by this source" never masquerades as an explicit negative).

    The CheXpert-style integer convention (1 positive, 0 negative, 2 uncertain,
    3 not-mentioned) is handled by passing negative_codes=(0, 3) and
    exclude_codes=(2,) for an uncertain-as-missing policy, or negative_codes=
    (0, 2, 3) for an uncertain-as-negative policy.
    """
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


# ----- Sampling -------------------------------------------------------------

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
    """Take up to `n_per_stratum` rows from each unique combination of
    strata_cols. Used to build size-bounded shared pools that stay balanced
    across sites and views without over-representing the largest source."""
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


# ----- Case-id helpers ------------------------------------------------------

def make_case_ids(dataset: str, keys: Sequence[str]) -> List[str]:
    """Deterministic, collision-resistant ids of the form
    '<dataset>__<key>'. The key should already be unique within the dataset
    (a dicom_id, image_id, or relative path with separators normalized)."""
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
