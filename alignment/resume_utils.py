"""
alignment/resume_utils.py
Created on June 5, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import List, Set, Tuple, Optional

import numpy as np
import pandas as pd


def case_subsample_idx(n_full: int, max_cases, seed: int) -> np.ndarray:
    if not max_cases or n_full <= int(max_cases):
        return np.arange(n_full)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_full, size=int(max_cases), replace=False))


def load_done_units(out_dir: str, base_name: str, key_cols: List[str]
                    ) -> Tuple[Set[Tuple], str]:
    out_csv     = os.path.join(out_dir, f"{base_name}.csv")
    partial_csv = os.path.join(out_dir, f"{base_name}.partial.csv")
    done: Set[Tuple] = set()
    prior_frames = []
    for src in (out_csv, partial_csv):
        if os.path.exists(src):
            try:
                pf = pd.read_csv(src)
                if set(key_cols).issubset(pf.columns):
                    done |= set(map(tuple, pf[key_cols].itertuples(index=False, name=None)))
                    prior_frames.append(pf)
            except Exception as e:
                print(f"[resume] could not read {os.path.basename(src)} ({e}).")
    if prior_frames and not os.path.exists(partial_csv):
        pd.concat(prior_frames, ignore_index=True).to_csv(partial_csv, index=False)
    return done, partial_csv


def append_unit(partial_csv: str, rows: List[dict]) -> None:
    if not rows:
        return
    hdr = not os.path.exists(partial_csv)
    pd.DataFrame(rows).to_csv(partial_csv, mode="a", header=hdr, index=False)


def finalize_partial(partial_csv: str, out_csv: str,
                     dedup_cols: Optional[List[str]] = None) -> pd.DataFrame:
    if os.path.exists(partial_csv):
        df = pd.read_csv(partial_csv)
    else:
        df = pd.DataFrame()
    if dedup_cols and not df.empty:
        keep = [c for c in dedup_cols if c in df.columns]
        if keep:
            df = df.drop_duplicates(subset=keep, keep="last")
    df.to_csv(out_csv, index=False)
    try:
        os.remove(partial_csv)
    except OSError:
        pass
    return df