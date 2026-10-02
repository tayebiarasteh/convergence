"""
reader_study/analyze_reader_studies.py
Created on June 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Dict, Optional

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from Inference.report_utils import report_metric, add_fdr
from Inference.stats_utils import bootstrap_proportion, N_PERM, BOOT_SEED
from Inference.resume_utils import write_csv_atomic
from Inference.resume_utils import MissingInput, append_status, status_path


def _read_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    for enc in ("utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=enc) as fh:
                first = fh.readline()
            sep = ";" if first.count(";") > first.count(",") else ","
            return pd.read_csv(path, encoding=enc, sep=sep)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(path)


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


def _true_label_cxr(mapping: pd.DataFrame) -> pd.Series:
    labels = []
    for _, r in mapping.iterrows():
        f = r["target_finding"]
        v = pd.to_numeric(pd.Series([r.get(f, np.nan)]), errors="coerce").iloc[0]
        labels.append(1 if v == 1.0 else 0)
    return pd.Series(labels, index=mapping.index)


def analyze_gold(task_dir: str, modality: str, reader: str) -> Optional[dict]:
    cases = _read_csv(os.path.join(task_dir, "cases.csv"))
    mapping = _read_csv(os.path.join(task_dir, "case_mapping.csv"))
    if cases is None or mapping is None:
        return None
    present_col = "finding_present"
    if present_col not in cases.columns or cases[present_col].isna().all():
        return None

    disp = "display_id"
    merged = cases.merge(mapping, on=disp, how="left")
    merged["reader_yes"] = merged[present_col].map(_yesno)
    merged = merged[merged["reader_yes"] != -1].copy()
    if len(merged) == 0:
        return None

    merged["true"] = _true_label_cxr(merged)

    correct = (merged["reader_yes"] == merged["true"]).astype(float, copy=False).values
    pos = merged["true"] == 1
    neg = merged["true"] == 0
    acc  = report_metric(correct, prefix="label_fidelity_accuracy")
    sens = report_metric((merged.loc[pos, "reader_yes"] == 1).astype(float, copy=False).values,
                         prefix="label_fidelity_sensitivity") if pos.any() else {}
    spec = report_metric((merged.loc[neg, "reader_yes"] == 0).astype(float, copy=False).values,
                         prefix="label_fidelity_specificity") if neg.any() else {}

    q = merged["image_quality"].map(_quality) if "image_quality" in merged else pd.Series([], dtype=str)
    distrust = merged["distrust_automated_read"].map(_yesno) if "distrust_automated_read" in merged else pd.Series([], dtype=int)
    distrust = distrust[distrust != -1]
    out = {"reader": reader, "modality": modality, "n": int(len(merged)),
           "measures": "dataset_label_vs_reader", "model_involved": False}
    out.update(acc); out.update(sens); out.update(spec)
    if len(distrust):
        for _k, _v in report_metric((distrust == 1).astype(float, copy=False).values,
                                    prefix="distrust_rate").items():
            out[_k] = _v
    if len(q):
        out["frac_suboptimal_or_worse"] = float((q != "adequate").mean())
    return {"summary": out, "merged": merged}


def _cohens_kappa(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a); b = np.asarray(b)
    n = len(a)
    if n == 0:
        return float("nan")
    po = float((a == b).mean())
    pa1 = a.mean(); pb1 = b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    return (po - pe) / (1 - pe) if (1 - pe) != 0 else float("nan")


def _kappa_with_p(a: np.ndarray, b: np.ndarray, n_perm: int = N_PERM) -> dict:
    a = np.asarray(a, int); b = np.asarray(b, int)
    n = len(a)
    point = _cohens_kappa(a, b) if n >= 2 else float("nan")
    if not np.isfinite(point):
        return {"estimate": float("nan"), "p_raw": float("nan"), "n": int(n)}
    rng = np.random.RandomState(BOOT_SEED)
    null = np.array([_cohens_kappa(a, b[rng.permutation(n)]) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    p = float((np.sum(np.abs(null) >= abs(point)) + 1) / (len(null) + 1)) if len(null) else float("nan")
    return {"estimate": float(point), "p_raw": p, "n": int(n)}


def analyze_agreement(r1_gold_dir: str, r2_overlap_dir: str) -> Optional[dict]:
    r1 = _read_csv(os.path.join(r1_gold_dir, "cases.csv"))
    r1_map = _read_csv(os.path.join(r1_gold_dir, "case_mapping.csv"))
    r2 = _read_csv(os.path.join(r2_overlap_dir, "cases.csv"))
    r2_map = _read_csv(os.path.join(r2_overlap_dir, "case_mapping.csv"))
    if any(x is None for x in (r1, r1_map, r2, r2_map)):
        return None
    for df_, where in ((r1, os.path.join(r1_gold_dir, "cases.csv")),
                       (r2, os.path.join(r2_overlap_dir, "cases.csv"))):
        if "finding_present" not in df_.columns:
            return None
    if r1["finding_present"].isna().all() or r2["finding_present"].isna().all():
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

    agree = (paired["r1_yes"] == paired["r2_yes"]).astype(float, copy=False).values
    agree_rep = report_metric(agree, prefix="percent_agreement")
    kappa = _kappa_with_p(paired["r1_yes"].values, paired["r2_yes"].values)
    out = {"comparison": "reader1_vs_reader2_cxr", "n_shared": int(len(paired))}
    out.update(agree_rep)
    kappa_row = {"comparison": "reader1_vs_reader2_cxr", "n_shared": int(len(paired)),
                 "n": kappa["n"], "cohens_kappa_unweighted": kappa["estimate"],
                 "p_raw": kappa["p_raw"]}
    return {"agreement": out, "kappa": kappa_row}


def build_triplet_scoring_file(reader_study_root: str, out_path: str) -> Optional[str]:
    pieces = []
    candidates = [
        (os.path.join(reader_study_root, "reader1_radiologist", "task_B_similarity")),
        (os.path.join(reader_study_root, "reader2_radiologist", "task_E_similarity")),
    ]
    for d in candidates:
        ans = _read_csv(os.path.join(d, "cases.csv"))
        mp  = _read_csv(os.path.join(d, "case_mapping.csv"))
        if ans is None or mp is None:
            continue
        if "more_similar_to" not in ans.columns or ans["more_similar_to"].isna().all():
            continue
        m = ans.merge(mp, on="triplet_id", how="left", suffixes=("", "_map"))
        need = {"case_a", "case_b", "case_c"}
        if not need.issubset(m.columns):
            continue
        m = m.rename(columns={"more_similar_to": "choice"})
        m["choice"] = m["choice"].astype(str).str.strip().str.lower()
        m = m[m["choice"].isin(["b", "c"])]
        pieces.append(m[["case_a", "case_b", "case_c", "choice"]])

    if not pieces:
        return None
    merged = pd.concat(pieces, ignore_index=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        try:
            same = read_csv_defensively(out_path).astype(str).reset_index(drop=True).equals(
                merged.astype(str).reset_index(drop=True))
        except Exception:
            same = False
        if same:
            return out_path
    write_csv_atomic(merged, out_path)
    return out_path


def analyze_all(cfg_path: str, reader_study_root: Optional[str] = None):
    cfg = read_config(cfg_path)["Convergence"]
    if reader_study_root is None:
        reader_study_root = os.path.join(cfg["alignment"]["results_base_dir"],
                                         "reader_study")
    out_dir = os.path.join(reader_study_root, "analysis")
    os.makedirs(out_dir, exist_ok=True)
    status = status_path(cfg, "reader_analysis")
    if not os.path.isdir(reader_study_root):
        raise MissingInput(
            f"[reader] no reader-study tree at {reader_study_root}; the files returned by the readers "
            f"have to be in place before this analysis can run.")

    r1 = os.path.join(reader_study_root, "reader1_radiologist")
    r2 = os.path.join(reader_study_root, "reader2_radiologist")

    summary_rows = []
    all_rows = []

    g_r1 = analyze_gold(os.path.join(r1, "task_A_gold_labels"), "cxr", "reader1")
    if g_r1:
        summary_rows.append(g_r1["summary"])
    g_r2 = analyze_gold(os.path.join(r2, "task_D_extension"), "cxr", "reader2_extension")
    if g_r2:
        summary_rows.append(g_r2["summary"])

    for row in summary_rows:
        r = dict(row); r["section"] = "gold_label_metrics"
        all_rows.append(r)

    ag = analyze_agreement(os.path.join(r1, "task_A_gold_labels"),
                           os.path.join(r2, "task_C_agreement"))
    if ag:
        r = dict(ag["agreement"]); r["section"] = "inter_reader_agreement"
        all_rows.append(r)
        k = dict(ag["kappa"]); k["section"] = "inter_reader_kappa"
        all_rows.append(k)

    trip_out = cfg["reader_study"].get("triplets_csv", "")
    if trip_out:
        build_triplet_scoring_file(reader_study_root, trip_out)

    trip_status = "PENDING (run main_ontology_analysis after this to score triplets)"
    trip_csv = os.path.join(cfg["alignment"]["results_base_dir"],
                            "results_e3_ontology", "triplet_accuracy.csv")
    from Inference.regimes import stale_reason
    why = stale_reason(trip_csv)
    trip = None if why else _read_csv(trip_csv)
    if why:
        trip_status = f"NOT FOLDED: {why}"
    if trip is not None and not trip.empty:
        folded = 0
        for _, tr in trip.iterrows():
            r = tr.to_dict()
            n_ok = float(r.get("n", 0) or 0) > 0
            acc_ok = str(r.get("triplet_acc_mean", "")).strip() not in ("", "nan")
            if not (n_ok and acc_ok):
                continue
            r["section"] = "triplet_grounding"
            r["modality"] = "cxr"
            all_rows.append(r)
            folded += 1
        trip_status = f"OK ({folded} rows folded of {len(trip)})"

    if all_rows:
        df = pd.DataFrame(all_rows)
        front = [c for c in ("section", "reader", "modality", "comparison", "n",
                             "n_shared") if c in df.columns]
        rest = [c for c in df.columns if c not in front]
        df = df[front + rest]
        out_csv = os.path.join(out_dir, "reader_study_results.csv")
        write_csv_atomic(df, out_csv)
        present = sorted(df["section"].unique())
        n_written = len(df)
    else:
        out_csv = None
        present = []
        n_written = 0

    def _has(sec): return "yes" if sec in present else "NO"

    for f in sorted(os.listdir(out_dir)):
        pass
    append_status(status, f"label-fidelity audit and triplets written to {out_dir}")
