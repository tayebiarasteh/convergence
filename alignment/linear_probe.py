"""
alignment/linear_probe.py
Created on May 27, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


LOGREG_C        = 0.1


LOGREG_MAX_ITER = 150


LOGREG_SEED     = 0


def fit_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[LogisticRegression, StandardScaler]:
    """Fit a standardized L2 logistic probe. Returns (model, scaler)."""
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train.astype(np.float64))
    model  = LogisticRegression(
        C=LOGREG_C, max_iter=LOGREG_MAX_ITER,
        random_state=LOGREG_SEED, class_weight="balanced",
    )
    model.fit(X_s, y_train)
    return model, scaler


def eval_probe(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    """Return AUROC of the probe on X_test."""
    if len(np.unique(y_test)) < 2:
        return float("nan")
    X_s   = scaler.transform(X_test.astype(np.float64))
    proba = model.predict_proba(X_s)[:, 1]
    return float(roc_auc_score(y_test, proba))


def eval_probe_scores(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    """Return (y_true, y_score) for AUROC bootstrap, instead of the scalar AUROC."""
    X_s   = scaler.transform(X_test.astype(np.float64))
    proba = model.predict_proba(X_s)[:, 1]
    return np.asarray(y_test, dtype=float), np.asarray(proba, dtype=float)


def per_finding_auroc(
    embeddings: np.ndarray,
    labels: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask:  np.ndarray,
    finding_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Fit one probe per finding and return per-finding test AUROC.

    Args:
        embeddings  (N, D) float32
        labels      dict finding_name -> (N,) float array with 1.0/0.0/NaN
        train_mask  (N,) bool
        test_mask   (N,) bool
        finding_names  subset to evaluate; None = all keys in labels

    Returns dict {finding_name: auroc}.
    """
    if finding_names is None:
        finding_names = list(labels.keys())

    # Row cap: logistic regression on hundreds of thousands of rows, fit per finding
    # (and per encoder pair in E7), is the dominant cost; AUROC is stable on a sample.
    max_train = int(globals().get("_PROBE_MAX_TRAIN", 25000))
    max_test  = int(globals().get("_PROBE_MAX_TEST", 20000))
    rng = np.random.RandomState(0)
    # Embedding-row finiteness is the same for every finding; compute once.
    emb_finite = np.isfinite(embeddings).all(axis=1)

    results: Dict[str, float] = {}
    for fname in finding_names:
        col = labels[fname]
        labeled = np.isfinite(col)
        # Require BOTH the label AND the embedding row to be finite: a partly-NaN
        # encoder (e.g. medgemma_vision ~97% finite) would otherwise feed NaN into
        # LogisticRegression, which raises.
        valid = labeled & emb_finite
        tr_idx  = np.where(train_mask & valid)[0]
        te_idx  = np.where(test_mask  & valid)[0]

        if len(tr_idx) < 20 or len(te_idx) < 10:
            results[fname] = float("nan")
            continue
        if len(np.unique(col[tr_idx])) < 2 or len(np.unique(col[te_idx])) < 2:
            results[fname] = float("nan")
            continue

        # Cap rows (keep class balance roughly by simple random subsample).
        if len(tr_idx) > max_train:
            tr_idx = np.sort(rng.choice(tr_idx, max_train, replace=False))
        if len(te_idx) > max_test:
            te_idx = np.sort(rng.choice(te_idx, max_test, replace=False))

        model, scaler = fit_probe(embeddings[tr_idx], col[tr_idx])
        results[fname] = eval_probe(model, scaler, embeddings[te_idx], col[te_idx])

    return results


def mean_auroc(auroc_dict: Dict[str, float]) -> float:
    vals = [v for v in auroc_dict.values() if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")