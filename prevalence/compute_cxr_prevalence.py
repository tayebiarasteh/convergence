"""
prevalence/compute_cxr_prevalence.py

Computes per-finding empirical prevalence from the harmonized CXR pool
manifest for the E6 fracture-law analysis.

For each finding column in the pool manifest (14 canonical + all extended),
this script counts the fraction of labeled images (finding != NaN) that are
positive (finding == 1.0). The resulting prevalence is the E6 x-axis regressor:
alignment per finding ~ log(prevalence), controlling for linear-probe
separability. A wide prevalence range (rare tail through common findings)
gives this analysis its statistical power.

Runs after build_cxr_pool.py; reads the pool manifest directly.

Run:
    python -m prevalence.compute_cxr_prevalence

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os

import numpy as np
import pandas as pd

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS, EXTENDED_MAPS


def main_compute_cxr_prevalence(global_config_path: str) -> str:
    params   = read_config(global_config_path)
    cfg      = params["Convergence"]
    pool_csv = cfg["cxr"]["pool_manifest_csv"]
    out_csv  = cfg["prevalence"]["out_csv"]

    if not os.path.exists(pool_csv):
        raise FileNotFoundError(
            f"[compute_cxr_prevalence] Pool manifest not found: {pool_csv}. "
            f"Run build_cxr_pool first."
        )

    print(f"[compute_cxr_prevalence] Reading pool manifest: {pool_csv}")
    pool = read_csv_defensively(pool_csv)
    print(f"  {len(pool)} rows | {pool['dataset'].nunique()} sites")

    # Collect all finding columns: canonical 14 + extended
    ext_names = sorted({name for emap in EXTENDED_MAPS.values()
                        for name in emap.values()})
    all_findings = list(CANONICAL_CXR_FINDINGS) + ext_names

    # Compute prevalence for every finding column that is present in the manifest
    records = []
    for finding in all_findings:
        if finding not in pool.columns:
            continue
        col = pd.to_numeric(pool[finding], errors="coerce")
        n_labeled  = int(col.notna().sum())
        n_positive = int((col == 1.0).sum())
        n_negative = int((col == 0.0).sum())
        prevalence = round(n_positive / n_labeled, 6) if n_labeled > 0 else float("nan")
        vocab = "canonical" if finding in CANONICAL_CXR_FINDINGS else "extended"

        # Per-site counts for the canonical findings
        site_counts = {}
        for site, grp in pool.groupby("dataset"):
            sc = pd.to_numeric(grp[finding], errors="coerce")
            site_counts[f"{site}_n_labeled"]  = int(sc.notna().sum())
            site_counts[f"{site}_n_positive"] = int((sc == 1.0).sum())

        records.append({
            "finding":    finding,
            "vocabulary": vocab,
            "n_labeled":  n_labeled,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "prevalence": prevalence,
            **site_counts,
        })

    out = (pd.DataFrame(records)
             .sort_values("prevalence", ascending=False)
             .reset_index(drop=True))

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out.to_csv(out_csv, index=False)

    print(f"\n[compute_cxr_prevalence] Results ({len(out)} findings) -> {out_csv}")
    print(out[["finding", "vocabulary", "n_labeled",
               "n_positive", "prevalence"]].to_string(index=False))
    return out_csv


if __name__ == "__main__":
    main_compute_cxr_prevalence(
        "/home/homesOnMaster/sarasteh/Documents/Repositories/convergence/config/config.yaml"
    )
