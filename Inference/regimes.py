"""
Inference/regimes.py
Created on September 23, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from typing import Optional

REGIMES = {
    "consensus": "gpa_column_centered",
    "e3": "e3_units_derm_class_map",
    "e4": "e4_residual_and_mean_panel_alignment",
    "e6": "e6_positive_cases_and_loaded_subgroups",
    "e14": "e14_mean_pairwise_neighbor_overlap",
    "e9_transfer_summary": "e9_transfer_above_reference_p975",
    "e12_link": "e12_chest_pool_crossed_jackknife",
    "e5_utility": "e5_utility_accuracy_case_bootstrap",
    "e10_scanner": "e10_scanner_accuracy_case_bootstrap",
    "init_alignment": "cell_to_initialization_alignment",
    "matrix_contrasts": "h2_interpretation_contrasts",
    "e9_stitching": "e9_ladder_label_free_linear_map",
    "e12_map_budget": "e12_map_budget_nested_prefixes",
    "e4_width": "e4_partial_spearman_given_embedding_width",
}

PRODUCED = {
    ("consensus", "encoder_residuals.csv"): "consensus",
    ("results_e3_ontology", "manifold_vs_ontology.csv"): "e3",
    ("results_e3_ontology", "triplet_accuracy.csv"): "e3",
    ("results_e4_scaling", "scaling_law_data.csv"): "e4",
    ("results_e4_scaling", "scaling_law_fits.csv"): "e4",
    ("results_e6_fracture", "per_finding_alignment.csv"): "e6",
    ("results_e6_fracture", "fracture_regression.csv"): "e6",
    ("results_e6_fracture", "findings_not_measurable.csv"): "e6",
    ("results_e6_fracture", "per_demographic_alignment.csv"): "e6",
    ("results_e6_fracture", "per_demographic_reference.csv"): "e6",
    ("results_e6_fracture", "per_demographic_tests.csv"): "e6",
    ("results_e14_shared", "shared_component_attribution.csv"): "e14",
    ("results_e14_shared", "per_case_agreement.csv"): "e14",
    ("results_e9_granularity", "ladder_transfer_summary.csv"): "e9_transfer_summary",
    ("results_e12_link", "alignment_vs_retention.csv"): "e12_link",
    ("results_e5_driver", "e5_cell_utility.csv"): "e5_utility",
    ("results_e10_competing", "acquisition_predictability.csv"): "e10_scanner",
    ("results_e5_driver", "cell_to_init_alignment.csv"): "init_alignment",
    ("results_e5_driver", "matrix_contrasts.csv"): "matrix_contrasts",
    ("results_e9_granularity", "ladder_stitching.csv"): "e9_stitching",
    ("results_e9_granularity", "ladder_stitching_summary.csv"): "e9_stitching",
    ("results_e7_artifact", "stitching_budget.csv"): "e12_map_budget",
    ("results_e7_artifact", "stitching_budget_summary.csv"): "e12_map_budget",
    ("results_e4_scaling", "scaling_law_width_control.csv"): "e4_width",
}


def regime_extra(name: str) -> dict:
    return {"regime": REGIMES[name]}


def stale_reason(path: str) -> Optional[str]:
    key = (os.path.basename(os.path.dirname(os.path.abspath(path))), os.path.basename(path))
    name = PRODUCED.get(key)
    if name is None or not os.path.exists(path):
        return None
    from Inference.resume_utils import read_build_params
    got = (read_build_params(path) or {}).get("regime")
    if got == REGIMES[name]:
        return None
    return (f"{key[1]} was written by an earlier version of its stage (regime {got!r}, "
            f"current {REGIMES[name]!r}); it is not read until that stage runs again")
