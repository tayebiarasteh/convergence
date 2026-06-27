"""
Inference/stats_utils.py
Created on May 26, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import numpy as np
from typing import List, Optional

N_BOOT    = 10000
N_PERM    = 1000
BOOT_SEED = 0


def bootstrap_proportion(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, n, size=(n_boot, n))
    boot_means = values[idx].mean(axis=1)
    return {
        "point":    float(values.mean()),
        "std":      float(boot_means.std(ddof=1)),
        "ci_lower": float(np.percentile(boot_means, 2.5)),
        "ci_upper": float(np.percentile(boot_means, 97.5)),
        "n":        int(n),
    }


def paired_bootstrap_diff(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    n = len(values_a)
    assert len(values_b) == n, "Both arrays must have equal length."
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "p_value": np.nan, "n": 0}
    rng   = np.random.RandomState(seed)
    idx   = rng.randint(0, n, size=(n_boot, n))
    point = float(values_a.mean() - values_b.mean())
    boot_diffs = values_a[idx].mean(axis=1) - values_b[idx].mean(axis=1)
    ci_lower   = float(np.percentile(boot_diffs, 2.5))
    ci_upper   = float(np.percentile(boot_diffs, 97.5))
    std        = float(boot_diffs.std(ddof=1))
    centered   = boot_diffs - point
    p_value    = float(np.mean(np.abs(centered) >= abs(point)))
    p_value    = max(p_value, 1.0 / n_boot)
    return {
        "point":    point,
        "std":      std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value":  p_value,
        "n":        int(n),
    }


def permutation_test_2groups(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_perm: int = N_PERM,
    seed: int = BOOT_SEED,
) -> float:
    """Two-sided permutation test for H0: mean(a) == mean(b).
    Returns p-value. Returns NaN if either group has fewer than 2 observations.
    """
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    obs    = abs(float(values_a.mean() - values_b.mean()))
    pooled = np.concatenate([values_a, values_b])
    n_a    = len(values_a)
    rng    = np.random.RandomState(seed)
    count  = 0
    for _ in range(n_perm):
        perm = rng.permutation(pooled)
        if abs(perm[:n_a].mean() - perm[n_a:].mean()) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def permutation_test_kgroups(
    groups: List[np.ndarray],
    n_perm: int = N_PERM,
    seed: int = BOOT_SEED,
) -> float:
    """Permutation F-test for H0: all group means are equal (k >= 2 groups).
    Uses the one-way ANOVA F-statistic. Returns p-value; NaN if < 2 valid groups.
    """
    groups = [np.asarray(g, dtype=float) for g in groups if len(g) >= 2]
    if len(groups) < 2:
        return float("nan")
    sizes   = [len(g) for g in groups]
    pooled  = np.concatenate(groups)
    n_total = len(pooled)
    k       = len(groups)

    def _f(gs: List[np.ndarray]) -> float:
        grand      = np.concatenate(gs).mean()
        ss_between = sum(len(g) * (g.mean() - grand) ** 2 for g in gs)
        ss_within  = sum(((g - g.mean()) ** 2).sum() for g in gs)
        if ss_within == 0:
            return float("inf")
        return (ss_between / (k - 1)) / (ss_within / (n_total - k))

    obs   = _f(groups)
    rng   = np.random.RandomState(seed)
    count = 0
    for _ in range(n_perm):
        perm  = rng.permutation(pooled)
        start = 0
        pgs   = []
        for s in sizes:
            pgs.append(perm[start:start + s])
            start += s
        if _f(pgs) >= obs:
            count += 1
    return (count + 1) / (n_perm + 1)


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([])
    nan_mask  = np.isnan(p_values)
    valid_idx = np.where(~nan_mask)[0]
    if len(valid_idx) == 0:
        return p_values.copy()
    valid_p  = p_values[valid_idx]
    m        = len(valid_p)
    order    = np.argsort(valid_p)
    p_sorted = valid_p[order]
    adj = p_sorted * m / (np.arange(m, dtype=float) + 1)
    for i in range(m - 2, -1, -1):
        adj[i] = min(adj[i], adj[i + 1])
    adj = np.minimum(adj, 1.0)
    inv_order = np.empty(m, dtype=int)
    inv_order[order] = np.arange(m)
    result = p_values.copy()
    result[valid_idx] = adj[inv_order]
    return result


def _percentile_ci(boot: np.ndarray) -> tuple:
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return (np.nan, np.nan, np.nan)
    std = float(np.std(boot, ddof=1)) if boot.size > 1 else np.nan
    return (std, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))


def paired_bootstrap_statistic(
    arrays: List[np.ndarray],
    stat_fn,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Generic paired bootstrap. All arrays are resampled with the SAME index
    per replicate so cross-column structure is preserved.
    Returns {point, std, ci_lower, ci_upper, n}.
    """
    arrays = [np.asarray(a, dtype=float) for a in arrays]
    n = len(arrays[0])
    assert all(len(a) == n for a in arrays), "all arrays must be equal length"
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    point = float(stat_fn(*arrays))
    rng   = np.random.RandomState(seed)
    idx   = rng.randint(0, n, size=(n_boot, n))
    boot  = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        resampled = [a[idx[b]] for a in arrays]
        try:
            boot[b] = stat_fn(*resampled)
        except Exception:
            boot[b] = np.nan
    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi,
            "n": int(n)}


def bootstrap_spearman(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Paired bootstrap CI for Spearman rho. NaN pairs dropped up front."""
    from scipy.stats import spearmanr
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 4:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(len(x))}

    def _rho(a, b):
        r, _ = spearmanr(a, b)
        return r
    return paired_bootstrap_statistic([x, y], _rho, n_boot=n_boot, seed=seed)


def bootstrap_slope(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    """Paired bootstrap CI for OLS slope of y on x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(len(x))}

    def _slope(a, b):
        A = np.vstack([a, np.ones_like(a)]).T
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        return coef[0]
    return paired_bootstrap_statistic([x, y], _slope, n_boot=n_boot, seed=seed)

def bootstrap_auroc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    from sklearn.metrics import roc_auc_score
    y_true  = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true, y_score = y_true[mask], y_score[mask]
    n = len(y_true)
    if n < 2 or len(np.unique(y_true)) < 2:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n)}
    point = float(roc_auc_score(y_true, y_score))
    rng   = np.random.RandomState(seed)
    idx   = rng.randint(0, n, size=(n_boot, n))
    boot  = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        yt, ys = y_true[idx[b]], y_score[idx[b]]
        if len(np.unique(yt)) < 2:
            boot[b] = np.nan
            continue
        boot[b] = roc_auc_score(yt, ys)
    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n)}
