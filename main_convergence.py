"""
main_convergence.py
Created on May 26, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
import warnings
warnings.filterwarnings("ignore")


GLOBAL_CONFIG_PATH = ("/PATH/convergence/config/config.yaml")



def main_preprocess_pcam(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.preprocess_histo_pcam import main_preprocess_pcam
    main_preprocess_pcam(cfg_path)


def main_build_cxr_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_cxr_pool import main_build_cxr_pool
    main_build_cxr_pool(cfg_path)


def main_build_cxr_paired_reports(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_cxr_paired_reports import main_build_cxr_paired_reports
    main_build_cxr_paired_reports(cfg_path)


def main_build_histo_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_histo_pool import main_build_histo_pool
    main_build_histo_pool(cfg_path)


def main_build_quilt_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_quilt_pool import main_build_quilt_pool
    main_build_quilt_pool(cfg_path)


def main_build_fundus_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_fundus_pool import main_build_fundus_pool
    main_build_fundus_pool(cfg_path)


def main_build_derm_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_derm_pool import main_build_derm_pool
    main_build_derm_pool(cfg_path)


def main_build_mammo_pool(cfg_path: str = GLOBAL_CONFIG_PATH):
    from data_loader.build_mammo_pool import main_build_mammo_pool
    main_build_mammo_pool(cfg_path)


def main_compute_prevalence(cfg_path: str = GLOBAL_CONFIG_PATH):
    from prevalence.compute_cxr_prevalence import main_compute_cxr_prevalence
    main_compute_cxr_prevalence(cfg_path)


def main_build_ontology(cfg_path: str = GLOBAL_CONFIG_PATH):
    from ontology.build_ontology_geometry import main_build_ontology_geometry
    main_build_ontology_geometry(cfg_path)


def main_patch_cxr_manifests(cfg_path: str = GLOBAL_CONFIG_PATH):
    from patch_cxr_manifests import main_patch_cxr_manifests as _patch
    _patch(cfg_path)



def main_extract_image_embeddings(
    cfg_path: str = GLOBAL_CONFIG_PATH,
    encoder_names=None,
    pool_names=None,
    device: str = "cuda",
):
    from encoders.extract_embeddings import main_extract_image_embeddings
    main_extract_image_embeddings(cfg_path, encoder_names, pool_names, device)


def main_extract_text_embeddings(
    cfg_path: str = GLOBAL_CONFIG_PATH,
    encoder_names=None,
    device: str = "cuda",
):
    from encoders.extract_embeddings import main_extract_text_embeddings
    main_extract_text_embeddings(cfg_path, encoder_names, device)



def main_build_consensus(cfg_path: str = GLOBAL_CONFIG_PATH):
    from alignment.consensus import main_build_consensus
    main_build_consensus(cfg_path)


def main_build_relative_reps(cfg_path: str = GLOBAL_CONFIG_PATH):
    from artifact.relative_reps import build_relative_reps
    build_relative_reps(cfg_path)



def main_convergence_map(cfg_path: str = GLOBAL_CONFIG_PATH):
    from alignment.convergence_map import main_convergence_map
    main_convergence_map(cfg_path)


def main_vision_language(cfg_path: str = GLOBAL_CONFIG_PATH):
    from alignment.vision_language import main_vision_language_alignment
    main_vision_language_alignment(cfg_path)


def main_ontology_analysis(cfg_path: str = GLOBAL_CONFIG_PATH):
    from alignment.ontology_analysis import main_ontology_analysis
    main_ontology_analysis(cfg_path)


def main_scaling_law(cfg_path: str = GLOBAL_CONFIG_PATH):
    from alignment.scaling_law import main_scaling_law
    main_scaling_law(cfg_path)



def main_build_training_mixtures(cfg_path: str = GLOBAL_CONFIG_PATH):
    from controlled.build_training_mixtures import main_build_training_mixtures
    main_build_training_mixtures(cfg_path)


def main_train_encoders(
    cfg_path: str = GLOBAL_CONFIG_PATH,
    force: bool = False,
):
    from controlled.train_encoder import run_full_e5_matrix
    run_full_e5_matrix(cfg_path, force=force)


def main_list_e5_runs(cfg_path: str = GLOBAL_CONFIG_PATH):
    from controlled.train_encoder import enumerate_e5_runs
    runs = enumerate_e5_runs(cfg_path)
    print(f"[E5] {len(runs)} cells in the configured matrix:")
    for run_id, *_ in runs:
        print(f"  {run_id}")
    return [r[0] for r in runs]


def main_train_one_run(
    run_id: str,
    cfg_path: str = GLOBAL_CONFIG_PATH,
    force: bool = False,
):
    from controlled.train_encoder import run_one_e5_cell
    run_one_e5_cell(cfg_path, run_id, force=force)


def main_converge_eval(cfg_path: str = GLOBAL_CONFIG_PATH):
    from controlled.converge_eval import main_converge_eval
    main_converge_eval(cfg_path)



def main_fracture_law(cfg_path: str = GLOBAL_CONFIG_PATH):
    from alignment.fracture import main_fracture_law
    main_fracture_law(cfg_path)



def main_universal_probe(cfg_path: str = GLOBAL_CONFIG_PATH):
    from artifact.universal_probe import main_universal_probe
    main_universal_probe(cfg_path)


def main_stitching(cfg_path: str = GLOBAL_CONFIG_PATH):
    from artifact.stitching import main_stitching
    main_stitching(cfg_path)


def main_drift_detector(cfg_path: str = GLOBAL_CONFIG_PATH):
    from artifact.drift_detector import main_drift_detector
    main_drift_detector(cfg_path)


def main_reader_study_sampling(cfg_path: str = GLOBAL_CONFIG_PATH):
    from artifact.reader_study import main_reader_study, main_reader_study_pathology
    main_reader_study(cfg_path)
    main_reader_study_pathology(cfg_path)


def main_analyze_reader_packets(cfg_path: str = GLOBAL_CONFIG_PATH):
    from reader_study.analyze_reader_studies import analyze_all
    analyze_all(cfg_path)



def main_theory_proposition(cfg_path: str = GLOBAL_CONFIG_PATH):
    from theory.proposition import main_theory_proposition
    main_theory_proposition(cfg_path)



def main_build_final_tables(cfg_path: str = GLOBAL_CONFIG_PATH):
    from aggregate.build_final_tables import main_build_final_tables
    main_build_final_tables(cfg_path)
