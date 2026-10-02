"""
alignment/resume_utils.py
Created on June 21, 2026

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
