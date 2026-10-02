"""
alignment/linear_probe.py
Created on June 23, 2026

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
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X_train.astype(np.float64, copy=False))
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
    if len(np.unique(y_test)) < 2:
        return float("nan")
    X_s   = scaler.transform(X_test.astype(np.float64, copy=False))
    proba = model.predict_proba(X_s)[:, 1]
    return float(roc_auc_score(y_test, proba))


def eval_probe_scores(
    model: LogisticRegression,
    scaler: StandardScaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
):
    X_s   = scaler.transform(X_test.astype(np.float64, copy=False))
    proba = model.predict_proba(X_s)[:, 1]
    return np.asarray(y_test, dtype=float), np.asarray(proba, dtype=float)


def per_finding_auroc(
    embeddings: np.ndarray,
    labels: Dict[str, np.ndarray],
    train_mask: np.ndarray,
    test_mask:  np.ndarray,
    finding_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    if finding_names is None:
        finding_names = list(labels.keys())

    max_train = int(globals().get("_PROBE_MAX_TRAIN", 25000))
    max_test  = int(globals().get("_PROBE_MAX_TEST", 20000))
    rng = np.random.RandomState(0)
    emb_finite = np.isfinite(embeddings).all(axis=1)

    results: Dict[str, float] = {}
    for fname in finding_names:
        col = labels[fname]
        labeled = np.isfinite(col)
        valid = labeled & emb_finite
        tr_idx  = np.where(train_mask & valid)[0]
        te_idx  = np.where(test_mask  & valid)[0]

        if len(tr_idx) < 20 or len(te_idx) < 10:
            results[fname] = float("nan")
            continue
        if len(np.unique(col[tr_idx])) < 2 or len(np.unique(col[te_idx])) < 2:
            results[fname] = float("nan")
            continue

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
