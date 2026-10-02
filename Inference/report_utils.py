"""
Inference/report_utils.py
Created on June 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from Inference.stats_utils import (
    N_BOOT, N_PERM, BOOT_SEED,
    bootstrap_proportion, paired_bootstrap_diff, paired_bootstrap_statistic,
    bootstrap_spearman, bootstrap_slope,
    permutation_test_2groups, permutation_test_kgroups, bh_fdr,
)


def fmt_pct(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    v = float(x)
    if not (-1.0 - 1e-9 <= v <= 1.0 + 1e-9):
        raise ValueError(f"fmt_pct received {v!r}, which is outside [-1, 1] and was "
                         f"therefore already scaled to percent by an upstream caller.")
    return round(v * 100.0, 1)


def fmt_ratio_pct(x: float) -> float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return float("nan")
    v = float(x)
    if not (-3.0 <= v <= 3.0):
        raise ValueError(f"fmt_ratio_pct received {v!r}, which is outside [-3, 3] and is not a "
                         f"ratio to a reference; check whether the caller already scaled it.")
    return round(v * 100.0, 1)


def fmt_one(x: float) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return float("nan")
    return round(float(x), 1)


def keep_p(p: float) -> float:
    if p is None:
        return float("nan")
    return float(p)


def effective_p(row) -> float:
    for key in ("p_fdr", "p_raw"):
        try:
            v = row[key]
        except (KeyError, TypeError, IndexError):
            v = None
        if v is not None and v == v:
            return float(v)
    return 1.0


def _pack(prefix: str, point, std, lo, hi, n, is_percent: bool,
          unit: str = "percent") -> Dict:
    f = fmt_pct if is_percent else fmt_one
    return {
        f"{prefix}_unit": (unit if is_percent else "raw"),
        f"{prefix}_mean":     f(point),
        f"{prefix}_std":      f(std),
        f"{prefix}_ci_low":   f(lo),
        f"{prefix}_ci_high":  f(hi),
        f"{prefix}_mean_raw": float(point) if point is not None and np.isfinite(point) else float("nan"),
        f"{prefix}_std_raw":  float(std)   if std   is not None and np.isfinite(std)   else float("nan"),
        f"{prefix}_ci_low_raw":  float(lo) if lo    is not None and np.isfinite(lo)    else float("nan"),
        f"{prefix}_ci_high_raw": float(hi) if hi    is not None and np.isfinite(hi)    else float("nan"),
        "n": int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0,
    }


def _pack_stat(prefix: str, estimate, lo, hi, n) -> Dict:
    def _r(v):
        return round(float(v), 4) if v is not None and np.isfinite(v) else float("nan")
    return {
        f"{prefix}_estimate":    _r(estimate),
        f"{prefix}_ci_low":      _r(lo),
        f"{prefix}_ci_high":     _r(hi),
        f"{prefix}_estimate_raw": float(estimate) if estimate is not None and np.isfinite(estimate) else float("nan"),
        f"{prefix}_ci_low_raw":  float(lo) if lo is not None and np.isfinite(lo) else float("nan"),
        f"{prefix}_ci_high_raw": float(hi) if hi is not None and np.isfinite(hi) else float("nan"),
        "n": int(n) if n is not None and not (isinstance(n, float) and np.isnan(n)) else 0,
    }


def report_metric(
    values: np.ndarray,
    prefix: str = "value",
    is_percent: bool = True,
    n_boot: int = N_BOOT,
) -> Dict:
    res = bootstrap_proportion(np.asarray(values, dtype=float), n_boot=n_boot)
    return _pack(prefix, res["point"], res["std"], res["ci_lower"],
                 res["ci_upper"], res["n"], is_percent)


def report_accuracy(
    correct: np.ndarray,
    prefix: str = "accuracy",
    cluster_ids=None,
    n_boot: int = N_BOOT,
) -> Dict:
    if cluster_ids is not None:
        ids = np.asarray(cluster_ids, dtype=object)
        if any(c is None or str(c).strip().lower() in ("", "nan", "none") for c in ids):
            cluster_ids = None
    res = bootstrap_proportion(np.asarray(correct, dtype=float), n_boot=n_boot,
                               cluster_ids=cluster_ids)
    out = _pack(prefix, res["point"], res["std"], res["ci_lower"], res["ci_upper"], res["n"],
                is_percent=True)
    out[f"{prefix}_resample"] = res.get("resample", "row")
    return out


def report_ratio(
    values: np.ndarray,
    prefix: str = "ratio",
    n_boot: int = N_BOOT,
) -> Dict:
    res = bootstrap_proportion(np.asarray(values, dtype=float), n_boot=n_boot)
    return {
        f"{prefix}_unit": "percent_of_reference",
        f"{prefix}_mean":     fmt_ratio_pct(res["point"]),
        f"{prefix}_std":      fmt_ratio_pct(res["std"]),
        f"{prefix}_ci_low":   fmt_ratio_pct(res["ci_lower"]),
        f"{prefix}_ci_high":  fmt_ratio_pct(res["ci_upper"]),
        f"{prefix}_mean_raw": float(res["point"]) if np.isfinite(res["point"]) else float("nan"),
        f"{prefix}_std_raw":  float(res["std"]) if np.isfinite(res["std"]) else float("nan"),
        f"{prefix}_ci_low_raw":  float(res["ci_lower"]) if np.isfinite(res["ci_lower"]) else float("nan"),
        f"{prefix}_ci_high_raw": float(res["ci_upper"]) if np.isfinite(res["ci_upper"]) else float("nan"),
        "n": int(res["n"]),
    }


def report_paired_diff(
    a: np.ndarray,
    b: np.ndarray,
    prefix: str = "diff",
    is_percent: bool = True,
    n_boot: int = N_BOOT,
) -> Dict:
    res = paired_bootstrap_diff(np.asarray(a, float), np.asarray(b, float), n_boot=n_boot)
    out = _pack(prefix, res["point"], res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent, unit="percentage_points")
    out["p_raw"] = keep_p(res["p_value"])
    out["p_fdr"] = float("nan")
    return out


def report_spearman(
    x: np.ndarray,
    y: np.ndarray,
    prefix: str = "spearman",
    n_boot: int = N_BOOT,
) -> Dict:
    from scipy.stats import spearmanr
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    res = bootstrap_spearman(x, y, n_boot=n_boot)
    out = _pack_stat(prefix, res["point"], res["ci_lower"], res["ci_upper"], res["n"])
    if len(x) >= 4:
        _, p = spearmanr(x, y)
        out["p_raw"] = keep_p(p)
    else:
        out["p_raw"] = float("nan")
    out["p_fdr"] = float("nan")
    return out


def report_partial_spearman(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    prefix: str = "partial_spearman",
    n_boot: int = N_BOOT,
) -> Dict:
    from scipy.stats import spearmanr
    from sklearn.linear_model import LinearRegression
    x = np.asarray(x, float); y = np.asarray(y, float); z = np.asarray(z, float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[mask], y[mask], z[mask]
    if len(x) < 5:
        out = _pack_stat(prefix, np.nan, np.nan, np.nan, len(x))
        out["p_raw"] = float("nan"); out["p_fdr"] = float("nan")
        return out

    def _resid(a, b):
        lr = LinearRegression().fit(b.reshape(-1, 1), a)
        return a - lr.predict(b.reshape(-1, 1))

    def _partial(xa, ya, za):
        rx = _resid(xa, za); ry = _resid(ya, za)
        r, _ = spearmanr(rx, ry)
        return r

    res = paired_bootstrap_statistic([x, y, z], _partial, n_boot=n_boot)
    out = _pack_stat(prefix, res["point"], res["ci_lower"], res["ci_upper"], res["n"])
    rx = _resid(x, z); ry = _resid(y, z)
    r_full, p_full = spearmanr(rx, ry)
    out["p_raw"] = keep_p(p_full)
    out["p_fdr"] = float("nan")
    return out


def report_slope(
    x: np.ndarray,
    y: np.ndarray,
    prefix: str = "slope",
    n_boot: int = N_BOOT,
) -> Dict:
    from scipy.stats import linregress
    x = np.asarray(x, float); y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    res = bootstrap_slope(x, y, n_boot=n_boot)
    out = _pack_stat(prefix, res["point"], res["ci_lower"], res["ci_upper"], res["n"])
    if len(x) >= 3 and np.ptp(x) > 0:
        lr = linregress(x, y)
        out["p_raw"]      = keep_p(lr.pvalue)
        out["r_squared"]  = round(float(lr.rvalue ** 2), 4)
        out["intercept_raw"] = float(lr.intercept)
    else:
        out["p_raw"] = float("nan"); out["r_squared"] = float("nan")
        out["intercept_raw"] = float("nan")
        out["degenerate_reason"] = ("fewer than 3 finite points" if len(x) < 3
                                    else "the predictor takes one value")
    out["p_fdr"] = float("nan")
    return out


def report_permutation_2(
    a: np.ndarray,
    b: np.ndarray,
    prefix: str = "diff",
    is_percent: bool = True,
    n_perm: int = N_PERM,
    tested: str = "",
) -> Dict:
    a = np.asarray(a, float); b = np.asarray(b, float)
    point = float(np.nanmean(a) - np.nanmean(b)) if len(a) and len(b) else float("nan")
    p     = permutation_test_2groups(a, b, n_perm=n_perm)
    f = fmt_pct if is_percent else fmt_one
    out = {
        f"{prefix}_unit":     "percentage_points" if is_percent else "raw",
        f"{prefix}_mean":     f(point),
        f"{prefix}_mean_raw": point,
        f"{prefix}_group_a_mean": f(np.nanmean(a)) if len(a) else float("nan"),
        f"{prefix}_group_b_mean": f(np.nanmean(b)) if len(b) else float("nan"),
        "n_a": int(len(a)), "n_b": int(len(b)),
        "p_raw": keep_p(p), "p_fdr": float("nan"),
        "tested": tested,
    }
    return out


def report_permutation_k(
    groups: List[np.ndarray],
    n_perm: int = N_PERM,
) -> Dict:
    p = permutation_test_kgroups(groups, n_perm=n_perm)
    return {
        "k_groups": len([g for g in groups if len(g) >= 2]),
        "group_means_pct": ";".join(
            f"{fmt_pct(np.nanmean(g))}" for g in groups if len(g) >= 2
        ),
        "p_raw": keep_p(p), "p_fdr": float("nan"),
    }


def report_mantel(
    d1: np.ndarray,
    d2: np.ndarray,
    prefix: str = "mantel",
    n_perm: int = N_PERM,
    n_boot: int = N_BOOT,
    n_objects: Optional[int] = None,
) -> Dict:
    from scipy.stats import spearmanr
    d1 = np.asarray(d1, float); d2 = np.asarray(d2, float)
    mask = np.isfinite(d1) & np.isfinite(d2)
    by_objects = (n_objects is not None and bool(mask.all())
                  and len(d2) == n_objects * (n_objects - 1) // 2)
    d1, d2 = d1[mask], d2[mask]
    bs  = bootstrap_spearman(d1, d2, n_boot=n_boot)
    out = _pack_stat(prefix, bs["point"], bs["ci_lower"], bs["ci_upper"], bs["n"])
    obs = float(spearmanr(d1, d2).correlation) if len(d1) >= 4 else float("nan")
    if not np.isfinite(obs):
        out["p_raw"] = float("nan")
    else:
        rng = np.random.RandomState(BOOT_SEED)
        iu, M2 = None, None
        if by_objects:
            iu = np.triu_indices(n_objects, k=1)
            M2 = np.zeros((n_objects, n_objects))
            M2[iu] = d2
            M2 = M2 + M2.T
        count = 0
        for _ in range(n_perm):
            if by_objects:
                p = rng.permutation(n_objects)
                d2p = M2[np.ix_(p, p)][iu]
            else:
                d2p = rng.permutation(d2)
            if abs(float(spearmanr(d1, d2p).correlation)) >= abs(obs):
                count += 1
        out["p_raw"] = keep_p((count + 1) / (n_perm + 1))
    out[f"{prefix}_permutation"] = "objects" if by_objects else "entries"
    out["p_fdr"] = float("nan")
    return out


def add_fdr(
    df: pd.DataFrame,
    family_cols: Optional[List[str]] = None,
    p_col: str = "p_raw",
    out_col: str = "p_fdr",
) -> pd.DataFrame:
    df = df.copy()
    if p_col not in df.columns:
        return df
    if family_cols:
        for _, idx in df.groupby(family_cols).groups.items():
            sub = df.loc[idx, p_col].values
            df.loc[idx, out_col] = bh_fdr(sub)
    else:
        df[out_col] = bh_fdr(df[p_col].values)
    df["significant_fdr05"] = df[out_col].apply(
        lambda q: bool(q < 0.05) if pd.notna(q) else False
    )
    return df


STANDARD_STAT_COLS = [
    "value_mean", "value_std", "value_ci_low", "value_ci_high",
    "value_mean_raw", "value_std_raw", "value_ci_low_raw", "value_ci_high_raw",
    "n", "p_raw", "p_fdr", "significant_fdr05",
]


def report_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    prefix: str = "auroc",
    n_boot: int = N_BOOT,
    cluster_ids=None,
) -> Dict:
    from Inference.stats_utils import bootstrap_auroc
    if cluster_ids is not None:
        ids = np.asarray(cluster_ids, dtype=object)
        bad = sum(1 for c in ids if c is None
                  or str(c).strip().lower() in ("", "nan", "none"))
        if bad:
            cluster_ids = None
    res = bootstrap_auroc(np.asarray(y_true, float), np.asarray(y_score, float),
                          n_boot=n_boot, cluster_ids=cluster_ids)
    out = _pack(prefix, res["point"], res["std"], res["ci_lower"],
                res["ci_upper"], res["n"], is_percent=True)
    out[f"{prefix}_resample"] = res.get("resample", "row")
    return out
