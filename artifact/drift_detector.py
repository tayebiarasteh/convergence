"""
artifact/drift_detector.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from tqdm import tqdm
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively

import warnings
warnings.filterwarnings("ignore")


def main_drift_detector(global_config_path: str) -> str:
    cfg     = read_config(global_config_path)["Convergence"]
    aln_cfg = cfg["alignment"]
    out_dir = os.path.join(aln_cfg["results_base_dir"], "results_e7_artifact")
    os.makedirs(out_dir, exist_ok=True)

    cons_dir = aln_cfg["consensus_dir"]
    dev_path = os.path.join(cons_dir, "case_deviations.npy")
    if not os.path.exists(dev_path):
        print("Case deviations not found. Run build_consensus first.")
        return out_dir

    deviations = np.load(dev_path).astype(np.float32)
    manifest   = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])

    # case_deviations.npy is computed on the consensus CASE SUBSAMPLE, so its length
    kept_path = os.path.join(cons_dir, "consensus_kept_case_idx.npy")
    if os.path.exists(kept_path):
        kept_idx = np.load(kept_path)
        if len(kept_idx) == len(deviations):
            manifest = manifest.iloc[kept_idx].reset_index(drop=True)
        elif len(deviations) == len(manifest):
            pass  # already full-length (deviation_max_cases:0 and no subsample)
        else:
            n = min(len(deviations), len(manifest))
            print(f"[E7/drift] WARNING: deviations ({len(deviations)}) vs manifest "
                  f"({len(manifest)}) length mismatch; truncating to {n}.")
            manifest = manifest.iloc[:n].reset_index(drop=True)
            deviations = deviations[:n]
    # Drop cases whose deviation is NaN (the subsample-of-subsample left some unset).
    finite = np.isfinite(deviations)
    if not finite.all():
        manifest   = manifest.iloc[finite].reset_index(drop=True)
        deviations = deviations[finite]

    # Cache deviation scores alongside manifest metadata
    dev_df = manifest[["case_id", "dataset", "split"]].copy()
    dev_df["deviation_score"] = deviations
    dev_csv = cfg["reader_study"]["deviation_score_csv"]
    os.makedirs(os.path.dirname(dev_csv), exist_ok=True)
    dev_df.to_csv(dev_csv, index=False)

    # Site-shift AUROC: MIMIC as in-distribution, other sites as OOD
    rows_auroc = []
    mimic_mask = (manifest["dataset"] == "mimic").values
    for ood_site in tqdm(manifest["dataset"].unique(), desc="[E7/drift] OOD sites", unit="site"):
        if ood_site == "mimic":
            continue
        ood_mask = (manifest["dataset"] == ood_site).values
        mask     = mimic_mask | ood_mask
        scores   = deviations[mask]
        labels   = ood_mask[mask].astype(int)
        if labels.sum() < 5 or (1 - labels).sum() < 5:
            continue
        from Inference.report_utils import report_auroc
        auroc_rep = report_auroc(labels, scores, prefix="auroc")
        row = {
            "comparison":    f"mimic_vs_{ood_site}",
            "ood_site":      ood_site,
            "n_ood":         int(labels.sum()),
            "n_id":          int((1 - labels).sum()),
        }
        row.update(auroc_rep)
        rows_auroc.append(row)

    pd.DataFrame(rows_auroc).to_csv(
        os.path.join(out_dir, "drift_detector_auroc.csv"), index=False
    )

    _stress_tests(cfg, deviations, manifest, out_dir)

    # Radiologist validation
    gold_csv = cfg["reader_study"]["gold_csv"]
    if os.path.exists(gold_csv):
        _radiologist_validation(gold_csv, dev_df, out_dir)

    return out_dir


def _corrupt_image(img_array: np.ndarray, corruption: str, level: int) -> np.ndarray:
    """Apply a simple corruption to a numpy RGB image array (H, W, 3) uint8."""
    import cv2
    if corruption == "jpeg_compression":
        quality = max(5, 95 - level * 15)
        _, enc = cv2.imencode(".jpg", img_array, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if corruption == "gaussian_noise":
        noise = np.random.RandomState(0).normal(0, level * 20, img_array.shape)
        return np.clip(img_array.astype(float) + noise, 0, 255).astype(np.uint8)
    if corruption == "downscale_resize":
        factor = max(2, level * 3)
        small  = cv2.resize(img_array, (img_array.shape[1] // factor,
                                         img_array.shape[0] // factor))
        return cv2.resize(small, (img_array.shape[1], img_array.shape[0]))
    return img_array


def _stress_tests(cfg: dict, clean_deviations: np.ndarray,
                   manifest: pd.DataFrame, out_dir: str):
    stress_csv = os.path.join(out_dir, "stress_test_plan.csv")
    if os.path.exists(stress_csv):
        return
    corruptions = ["jpeg_compression", "gaussian_noise", "downscale_resize"]
    levels      = [1, 2, 3]
    rows = []
    for c in corruptions:
        for l in levels:
            rows.append({
                "corruption": c, "level": l,
                "status": "pending_re_embedding",
            })
    pd.DataFrame(rows).to_csv(stress_csv, index=False)


def _radiologist_validation(gold_csv: str, dev_df: pd.DataFrame, out_dir: str):
    """Compute AUROC of deviation score vs radiologist distrust label."""
    gold = read_csv_defensively(gold_csv)
    if "distrust" not in gold.columns and "suboptimal" not in gold.columns:
        print("Gold CSV has no distrust/suboptimal column; "
              "skipping reader validation.")
        return
    label_col = "distrust" if "distrust" in gold.columns else "suboptimal"
    merged = gold[["case_id", label_col]].merge(
        dev_df[["case_id", "deviation_score"]], on="case_id", how="inner"
    )
    if len(merged) < 20:
        return
    labels = pd.to_numeric(merged[label_col], errors="coerce").values
    scores = merged["deviation_score"].values
    valid  = np.isfinite(labels) & np.isfinite(scores)
    if valid.sum() < 20 or len(np.unique(labels[valid])) < 2:
        return
    from Inference.report_utils import report_auroc
    auroc_rep = report_auroc(labels[valid].astype(int), scores[valid], prefix="auroc")
    out = {"n_cases": int(valid.sum()), "label_col": label_col}
    out.update(auroc_rep)
    pd.DataFrame([out]).to_csv(
        os.path.join(out_dir, "reader_validation.csv"), index=False)
