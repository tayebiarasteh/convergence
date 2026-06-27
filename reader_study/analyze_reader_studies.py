"""
reader_study/analyze_reader_studies.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from Inference.report_utils import report_metric, report_auroc, add_fdr
from Inference.stats_utils import bootstrap_proportion, N_BOOT, BOOT_SEED


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")


def _yesno(s) -> int:
    if pd.isna(s):
        return -1
    t = str(s).strip().lower()
    if t in ("yes", "y", "1", "true"):
        return 1
    if t in ("no", "n", "0", "false"):
        return 0
    return -1


def _quality(s) -> str:
    if pd.isna(s):
        return ""
    t = str(s).strip().lower()
    if t.startswith("adeq"):
        return "adequate"
    if t.startswith("subopt"):
        return "suboptimal"
    if t.startswith("non"):
        return "non-diagnostic"
    return str(s).strip()


def _bc(s) -> str:
    """Normalize a triplet B/C choice."""
    if pd.isna(s):
        return ""
    t = str(s).strip().lower()
    if t in ("b", "option_b", "optionb"):
        return "B"
    if t in ("c", "option_c", "optionc"):
        return "C"
    return str(s).strip().upper()


def _true_label_cxr(mapping: pd.DataFrame) -> pd.Series:
    """For a CXR gold mapping, the true label of the asked finding is the value
    of that finding's column for the case (the question asks about target_finding)."""
    labels = []
    for _, r in mapping.iterrows():
        f = r["target_finding"]
        v = pd.to_numeric(pd.Series([r.get(f, np.nan)]), errors="coerce").iloc[0]
        labels.append(1 if v == 1.0 else 0)
    return pd.Series(labels, index=mapping.index)


def _true_label_histo(mapping: pd.DataFrame) -> pd.Series:
    return (mapping["tissue_class"].astype(str) ==
            mapping["target_class"].astype(str)).astype(int)


def analyze_gold(task_dir: str, modality: str, reader: str) -> Optional[dict]:
    cases = _read_csv(os.path.join(task_dir, "cases.csv"))
    mapping = _read_csv(os.path.join(task_dir, "case_mapping.csv"))
    if cases is None or mapping is None:
        return None
    present_col = "finding_present" if modality == "cxr" else "tissue_present"
    if present_col not in cases.columns or cases[present_col].isna().all():
        print(f"[{reader}] gold task not filled yet ({task_dir}).")
        return None

    disp = "display_id"
    merged = cases.merge(mapping, on=disp, how="left")
    merged["reader_yes"] = merged[present_col].map(_yesno)
    merged = merged[merged["reader_yes"] != -1].copy()
    if len(merged) == 0:
        return None

    if modality == "cxr":
        merged["true"] = _true_label_cxr(merged)
    else:
        merged["true"] = _true_label_histo(merged)

    correct = (merged["reader_yes"] == merged["true"]).astype(float).values
    pos = merged["true"] == 1
    neg = merged["true"] == 0
    acc  = report_metric(correct, prefix="accuracy")
    sens = report_metric((merged.loc[pos, "reader_yes"] == 1).astype(float).values,
                         prefix="sensitivity") if pos.any() else {}
    spec = report_metric((merged.loc[neg, "reader_yes"] == 0).astype(float).values,
                         prefix="specificity") if neg.any() else {}

    # Quality + distrust rates (descriptive performance fractions)
    q = merged["image_quality"].map(_quality) if "image_quality" in merged else pd.Series([], dtype=str)
    distrust = merged["distrust_automated_read"].map(_yesno) if "distrust_automated_read" in merged else pd.Series([], dtype=int)
    distrust = distrust[distrust != -1]
    out = {"reader": reader, "modality": modality, "n": int(len(merged))}
    out.update(acc); out.update(sens); out.update(spec)
    if len(distrust):
        out.update(report_metric((distrust == 1).astype(float).values, prefix="distrust_rate"))
    if len(q):
        out["frac_suboptimal_or_worse"] = float((q != "adequate").mean())
    return {"summary": out, "merged": merged}



def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's kappa for two binary raters."""
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1 = a.mean(); pb1 = b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")


def _bootstrap_kappa(a: np.ndarray, b: np.ndarray, n_boot: int = N_BOOT) -> dict:
    """Kappa is a STATISTICAL MEASURE: report estimate + 95% CI on natural scale."""
    a = np.asarray(a, int); b = np.asarray(b, int)
    n = len(a)
    if n < 2:
        return {"estimate": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n": int(n)}
    point = _cohens_kappa(a, b)
    rng = np.random.RandomState(BOOT_SEED)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot[i] = _cohens_kappa(a[idx], b[idx])
    boot = boot[np.isfinite(boot)]
    lo = float(np.percentile(boot, 2.5)) if len(boot) else float("nan")
    hi = float(np.percentile(boot, 97.5)) if len(boot) else float("nan")
    return {"estimate": round(float(point), 4), "ci_low": round(lo, 4),
            "ci_high": round(hi, 4), "n": int(n)}


def analyze_agreement(r1_gold_dir: str, r2_overlap_dir: str) -> Optional[dict]:
    """Pair R1's gold answers with R2's overlap answers by case_id and compute
    percent agreement (performance, percent+CI) and Cohen's kappa (statistical
    measure, natural scale + CI)."""
    r1 = _read_csv(os.path.join(r1_gold_dir, "cases.csv"))
    r1_map = _read_csv(os.path.join(r1_gold_dir, "case_mapping.csv"))
    r2 = _read_csv(os.path.join(r2_overlap_dir, "cases.csv"))
    r2_map = _read_csv(os.path.join(r2_overlap_dir, "case_mapping.csv"))
    if any(x is None for x in (r1, r1_map, r2, r2_map)):
        return None
    if r1["finding_present"].isna().all() or r2["finding_present"].isna().all():
        print("[agreement] one of the readers has not filled the overlap yet.")
        return None

    r1m = r1.merge(r1_map[["display_id", "case_id"]], on="display_id", how="left")
    r1m["r1_yes"] = r1m["finding_present"].map(_yesno)
    r2m = r2.merge(r2_map[["r2_display_id", "case_id"]],
                   left_on="display_id", right_on="r2_display_id", how="left")
    r2m["r2_yes"] = r2m["finding_present"].map(_yesno)

    paired = r1m[["case_id", "r1_yes"]].merge(
        r2m[["case_id", "r2_yes"]], on="case_id", how="inner")
    paired = paired[(paired["r1_yes"] != -1) & (paired["r2_yes"] != -1)]
    if len(paired) < 2:
        return None

    agree = (paired["r1_yes"] == paired["r2_yes"]).astype(float).values
    agree_rep = report_metric(agree, prefix="percent_agreement")
    kappa = _bootstrap_kappa(paired["r1_yes"].values, paired["r2_yes"].values)
    out = {"comparison": "reader1_vs_reader2_cxr", "n_shared": int(len(paired))}
    out.update(agree_rep)
    out.update({"kappa_estimate": kappa["estimate"],
                "kappa_ci_low": kappa["ci_low"],
                "kappa_ci_high": kappa["ci_high"]})
    return out


def analyze_distrust_vs_deviation(merged_gold: pd.DataFrame, dev_csv: str,
                                  modality: str) -> Optional[dict]:
    """If a deviation-score CSV exists, test whether the reader's distrust item
    is predicted by the model deviation score (AUROC, performance, percent+CI)."""
    if not dev_csv or not os.path.exists(dev_csv):
        return None
    if "distrust_automated_read" not in merged_gold.columns:
        return None
    dev = _read_csv(dev_csv)
    if dev is None or "case_id" not in dev.columns:
        return None
    m = merged_gold.merge(dev[["case_id", "deviation_score"]], on="case_id", how="left")
    m["distrust"] = m["distrust_automated_read"].map(_yesno)
    m = m[(m["distrust"] != -1) & m["deviation_score"].notna()]
    if len(m) < 10 or m["distrust"].nunique() < 2:
        return None
    rep = report_auroc(m["distrust"].values, m["deviation_score"].values, prefix="auroc")
    out = {"modality": modality, "endpoint": "distrust_vs_deviation"}
    out.update(rep)
    return out



def analyze_all(cfg_path: str, reader_study_root: Optional[str] = None):
    cfg = read_config(cfg_path)["Convergence"]
    if reader_study_root is None:
        reader_study_root = os.path.join(cfg["alignment"]["results_base_dir"],
                                         "reader_study")
    out_dir = os.path.join(reader_study_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    r1 = os.path.join(reader_study_root, "reader1_radiologist")
    r2 = os.path.join(reader_study_root, "reader2_radiologist")
    r3 = os.path.join(reader_study_root, "reader3_pathologist")

    summary_rows = []
    merged_for_dev = {}

    g_r1 = analyze_gold(os.path.join(r1, "task_A_gold_labels"), "cxr", "reader1")
    if g_r1:
        summary_rows.append(g_r1["summary"]); merged_for_dev["reader1"] = g_r1["merged"]
    g_r2 = analyze_gold(os.path.join(r2, "task_D_extension"), "cxr", "reader2_extension")
    if g_r2:
        summary_rows.append(g_r2["summary"])
    g_r3 = analyze_gold(os.path.join(r3, "task_F_gold_labels"), "histo", "reader3")
    if g_r3:
        summary_rows.append(g_r3["summary"]); merged_for_dev["reader3"] = g_r3["merged"]

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(
            os.path.join(out_dir, "reader_gold_metrics.csv"), index=False)

    ag = analyze_agreement(os.path.join(r1, "task_A_gold_labels"),
                           os.path.join(r2, "task_C_agreement"))
    if ag:
        pd.DataFrame([ag]).to_csv(
            os.path.join(out_dir, "reader_agreement.csv"), index=False)

    dev_rows = []
    cxr_dev = cfg["reader_study"].get("deviation_score_csv", "")
    histo_dev = cfg["reader_study"].get("histo_deviation_score_csv", "")
    if "reader1" in merged_for_dev:
        r = analyze_distrust_vs_deviation(merged_for_dev["reader1"], cxr_dev, "cxr")
        if r:
            dev_rows.append(r)
    if "reader3" in merged_for_dev:
        r = analyze_distrust_vs_deviation(merged_for_dev["reader3"], histo_dev, "histo")
        if r:
            dev_rows.append(r)
    if dev_rows:
        pd.DataFrame(dev_rows).to_csv(
            os.path.join(out_dir, "distrust_vs_deviation.csv"), index=False)


    for f in sorted(os.listdir(out_dir)):
        print(f"  {f}")

