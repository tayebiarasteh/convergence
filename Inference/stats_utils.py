"""
Inference/stats_utils.py
Created on June 17, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import numpy as np
from typing import List, Optional

N_BOOT    = 1000
N_PERM    = 1000
BOOT_SEED = 0


def bootstrap_proportion(
    values: np.ndarray,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    cluster_ids=None,
) -> dict:
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": 0}
    if cluster_ids is not None:
        boot_means = np.array([values[take].mean() if len(take) else np.nan
                               for take in cluster_bootstrap_indices(cluster_ids, n_boot, seed)])
        std, lo, hi = _percentile_ci(boot_means)
        return {"point": float(values.mean()), "std": std, "ci_lower": lo, "ci_upper": hi,
                "n": int(n), "resample": "cluster"}
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
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]
    if len(values_a) < 2 or len(values_b) < 2:
        return float("nan")
    obs    = abs(float(values_a.mean() - values_b.mean()))
    if not np.isfinite(obs):
        return float("nan")
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
    groups = [np.asarray(g, dtype=float) for g in groups]
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) >= 2]
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
    if not np.isfinite(obs):
        return float("nan")
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
    cluster_ids=None,
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
    if cluster_ids is not None:
        cl = np.asarray(cluster_ids, dtype=object)[mask]
        draws = cluster_bootstrap_indices(cl, n_boot=n_boot, seed=seed)
    else:
        idx = rng.randint(0, n, size=(n_boot, n))
        draws = (idx[b] for b in range(n_boot))
    boot  = np.empty(n_boot, dtype=float)
    for b, take in enumerate(draws):
        if b >= n_boot:
            break
        yt, ys = y_true[take], y_score[take]
        if len(np.unique(yt)) < 2:
            boot[b] = np.nan
            continue
        boot[b] = roc_auc_score(yt, ys)
    std, lo, hi = _percentile_ci(boot)
    return {"point": point, "std": std, "ci_lower": lo, "ci_upper": hi, "n": int(n),
            "resample": "cluster" if cluster_ids is not None else "row"}


def subsample_statistic(
    arrays: List[np.ndarray],
    stat_fn,
    n_sub: int = 0,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    frac: float = 0.632,
) -> dict:
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("subsample_statistic: all arrays must be the same length")
    m = int(n_sub) if n_sub else max(2, int(round(frac * n)))
    if n < 4 or m < 2 or m > n:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n), "n_sub": int(m), "resample": "subsample"}
    rng = np.random.RandomState(seed)
    boot = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.permutation(n)[:m]
        try:
            boot[b] = stat_fn(*[a[idx] for a in arrays])
        except Exception:
            boot[b] = np.nan
    good = boot[np.isfinite(boot)]
    if good.size == 0:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n), "n_sub": int(m), "resample": "subsample"}
    std, lo, hi = _percentile_ci(boot)
    return {"point": float(np.mean(good)), "std": std, "ci_lower": lo, "ci_upper": hi,
            "n": int(n), "n_sub": int(m), "resample": "subsample", "n_boot": int(n_boot)}

def jackknife_blocks_statistic(
    arrays: List[np.ndarray],
    stat_fn,
    n_blocks: int = 20,
    seed: int = BOOT_SEED,
) -> dict:
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if n < n_blocks * 2:
        return {"point": np.nan, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n), "resample": "jackknife"}
    point = float(stat_fn(*arrays))
    order = np.random.RandomState(seed).permutation(n)
    blocks = np.array_split(order, n_blocks)
    vals = np.empty(n_blocks, dtype=float)
    for i, blk in enumerate(blocks):
        keep = np.setdiff1d(order, blk, assume_unique=True)
        try:
            vals[i] = stat_fn(*[a[keep] for a in arrays])
        except Exception:
            vals[i] = np.nan
    good = vals[np.isfinite(vals)]
    if good.size < 2:
        return {"point": point, "std": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "n": int(n), "resample": "jackknife"}
    se = float(np.sqrt((good.size - 1) / good.size * np.sum((good - good.mean()) ** 2)))
    return {"point": point, "std": se, "ci_lower": point - 1.96 * se,
            "ci_upper": point + 1.96 * se, "n": int(n),
            "resample": "jackknife", "n_blocks": int(good.size)}


def jackknife_over_units(values: np.ndarray, unit_a, unit_b, stat_fn=np.nanmean) -> dict:
    values = np.asarray(values, dtype=float)
    a = np.asarray(unit_a, dtype=object)
    b = np.asarray(unit_b, dtype=object)
    units = sorted(set(a.tolist()) | set(b.tolist()))
    if len(values) < 4 or len(units) < 3:
        return {"point": float(stat_fn(values)) if len(values) else np.nan,
                "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "n": int(len(values)), "n_units": len(units),
                "resample": "jackknife_over_units"}
    point = float(stat_fn(values))
    reps = []
    for u in units:
        keep = (a != u) & (b != u)
        if keep.sum() >= 2:
            reps.append(float(stat_fn(values[keep])))
    good = np.asarray([r for r in reps if np.isfinite(r)], dtype=float)
    if good.size < 3:
        return {"point": point, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                "n": int(len(values)), "n_units": len(units),
                "resample": "jackknife_over_units"}
    se = float(np.sqrt((good.size - 1) / good.size * np.sum((good - good.mean()) ** 2)))
    return {"point": point, "std": se, "ci_lower": point - 1.96 * se,
            "ci_upper": point + 1.96 * se, "n": int(len(values)),
            "n_units": int(good.size), "resample": "jackknife_over_units"}


def jackknife_pairs_statistic(unit_a, unit_b, arrays: List[np.ndarray], stat_fn,
                              min_rows: int = 4) -> dict:
    from scipy.stats import norm
    a = np.asarray(unit_a, dtype=object)
    b = np.asarray(unit_b, dtype=object)
    arrays = [np.asarray(x) for x in arrays]
    units = sorted(set(a.tolist()) | set(b.tolist()))
    out = {"point": np.nan, "std": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
           "p_value": np.nan, "n": int(len(a)), "n_units": 0,
           "resample": "jackknife_over_units"}
    if len(a) < min_rows or len(units) < 3:
        return out
    try:
        point = float(stat_fn(*arrays))
    except Exception:
        return out
    reps = []
    for u in units:
        keep = (a != u) & (b != u)
        if keep.sum() < min_rows:
            continue
        try:
            v = float(stat_fn(*[x[keep] for x in arrays]))
        except Exception:
            continue
        if np.isfinite(v):
            reps.append(v)
    good = np.asarray(reps, dtype=float)
    out["point"] = point
    if good.size < 3 or not np.isfinite(point):
        return out
    se = float(np.sqrt((good.size - 1) / good.size * np.sum((good - good.mean()) ** 2)))
    out.update({"std": se, "ci_lower": point - 1.96 * se, "ci_upper": point + 1.96 * se,
                "n_units": int(good.size),
                "p_value": float(2.0 * norm.sf(abs(point) / se)) if se > 0 else np.nan})
    return out


def assert_interval_brackets(point, ci_lower, ci_upper, label: str = "") -> None:
    if not all(np.isfinite([point, ci_lower, ci_upper])):
        return
    if not (ci_lower - 1e-9 <= point <= ci_upper + 1e-9):
        raise ValueError(
            f"interval does not bracket its point estimate{' for ' + label if label else ''}: "
            f"point={point!r} outside [{ci_lower!r}, {ci_upper!r}]. The resampling scheme is "
            f"estimating a different statistic from the one reported.")


def check_intervals_bracket(df, point_col: str, lo_col: str, hi_col: str,
                            label: str = "") -> int:
    import pandas as pd
    sub = df[[point_col, lo_col, hi_col]].apply(pd.to_numeric, errors="coerce").dropna()
    bad = sub[(sub[lo_col] > sub[point_col] + 1e-9) | (sub[point_col] > sub[hi_col] + 1e-9)]
    if len(bad):
        raise ValueError(
            f"{len(bad)} of {len(sub)} rows have an interval excluding their point estimate"
            f"{' in ' + label if label else ''}; first offender: {bad.iloc[0].to_dict()}")
    return 0


def reference_gap_at_counts(
    overall_value: float,
    counts: List[tuple],
    n_draw: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> dict:
    counts = [(int(a), int(b)) for a, b in counts if int(a) > 0 and int(b) > 0]
    if len(counts) < 2 or not np.isfinite(overall_value):
        return {"reference_mean": np.nan, "reference_ci_lower": np.nan,
                "reference_ci_upper": np.nan, "n_subgroups": len(counts)}
    a = float(np.clip(overall_value, 1e-6, 1 - 1e-6))
    rng = np.random.RandomState(seed)
    draws = np.empty(n_draw, dtype=float)
    q = a / (2.0 - a)
    for i in range(n_draw):
        vals = []
        for npos, nneg in counts:
            var = (a * (1 - a)
                   + (npos - 1) * (q - a * a)
                   + (nneg - 1) * (2 * a * a / (1 + a) - a * a)) / (npos * nneg)
            vals.append(rng.normal(a, np.sqrt(max(var, 1e-12))))
        draws[i] = max(vals) - min(vals)
    std, lo, hi = _percentile_ci(draws)
    return {"reference_mean": float(np.mean(draws)), "reference_std": std,
            "reference_ci_lower": lo, "reference_ci_upper": hi,
            "n_subgroups": len(counts), "n_draw": int(n_draw)}


def exceedance_p(observed: float, reference_draws: np.ndarray) -> float:
    reference_draws = np.asarray(reference_draws, dtype=float)
    reference_draws = reference_draws[np.isfinite(reference_draws)]
    if not np.isfinite(observed) or reference_draws.size == 0:
        return float("nan")
    return float((np.sum(reference_draws >= observed) + 1) / (reference_draws.size + 1))


def minimum_detectable_effect(sd_of_difference: float, n: int,
                              alpha: float = 0.05, power: float = 0.80) -> float:
    from scipy.stats import norm
    if not np.isfinite(sd_of_difference) or n is None or n < 2:
        return float("nan")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * sd_of_difference / np.sqrt(n))


def minimum_detectable_effect_from_se(se: float, alpha: float = 0.05,
                                      power: float = 0.80) -> float:
    from scipy.stats import norm
    if not np.isfinite(se) or se <= 0:
        return float("nan")
    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    return float(z * se)


def cluster_bootstrap_indices(cluster_ids, n_boot: int = N_BOOT,
                              seed: int = BOOT_SEED):
    ids = np.asarray(cluster_ids, dtype=object)
    missing = np.array([c is None or (isinstance(c, float) and not np.isfinite(c))
                        or str(c).strip().lower() in ("", "nan", "none")
                        for c in ids])
    if missing.any():
        raise ValueError(f"cluster_bootstrap_indices: {int(missing.sum())} of {len(ids)} "
                         f"rows carry no cluster id, which would collapse into one cluster.")
    labels = np.asarray([str(c) for c in ids])
    uniq, inv = np.unique(labels, return_inverse=True)

    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv, minlength=len(uniq))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    rng = np.random.RandomState(seed)
    for _ in range(n_boot):
        pick = rng.randint(0, len(uniq), size=len(uniq))
        take = counts[pick]
        total = int(take.sum())
        if total == 0:
            yield np.empty(0, dtype=np.int64)
            continue
        dest = np.concatenate(([0], np.cumsum(take)[:-1]))
        src = np.ones(total, dtype=np.int64)
        src[0] = starts[pick[0]]
        boundary = dest[1:][take[1:] > 0] if len(pick) > 1 else np.empty(0, dtype=np.int64)
        if boundary.size:
            prev = np.searchsorted(np.cumsum(take), boundary, side="right")
            src[boundary] = starts[pick[prev]] - (starts[pick[prev - 1]] + take[prev - 1] - 1)
        yield order[np.cumsum(src)]
