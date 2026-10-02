# Medical foundation models converge less under label supervision

## Overview

The code measures the agreement between the image representations of medical foundation models, tests which training choices determine that agreement, and transfers classifiers between models. Agreement between two models is the mutual k-nearest-neighbor overlap of their embeddings of the same images. It is compared with the agreement of a randomly initialized network on the same images. The code builds image pools for chest radiography, histopathology, dermoscopy, fundus photography, and mammography, extracts image and report embeddings with public models, and trains models that differ only in pretraining objective, model size, random seed, initialization, or training patients. It computes the agreement between models, how agreement depends on the source data and the findings, and the transfer of linear classifiers between models through a linear mapping or a relative representation. Both are fitted on unlabeled images.

## Installation

```bash
git clone https://github.com/tayebiarasteh/convergence.git
cd convergence
```

Install PyTorch, torchvision, Transformers, accelerate, bitsandbytes, timm, OpenCLIP, TorchXRayVision, faiss, scikit-learn, SciPy, NumPy, pandas, h5py, Pillow, PyYAML, safetensors, huggingface_hub, and tqdm. The CONCH model needs the `conch` package from its [repository](https://github.com/mahmoodlab/CONCH).

## Configuration

Every path and setting is read from `config/config.yaml`. Set the roots at the top of the file: `ocean_root` for the outputs, `datasets_root` for the datasets, and `repo_root` for this repository. Each analysis has a `main_` function that takes the path of `config/config.yaml` as its first argument.

The gated checkpoints (DINOv3, UNI, UNI2-h, Virchow, Virchow2, Prov-GigaPath, CONCH, RETFound, and MedGemma) require that you accept their terms on Hugging Face and enter your token in `hf_token`. The replication set of the controlled study consists of the models trained in [FRAME](https://arxiv.org/abs/2608.25981). Set `frame_root` to their location. The repository contains no datasets and no model weights.

## Code structure

- `data_loader/`: the image pools of the five modalities, TAIX-Ray, and ReXGradient-160K, the report pairs, and the patient-disjoint splits. `data_loader/cxr_harmonization.py` maps the chest radiograph labels of the six sources to one vocabulary, and `patch_cxr_manifests.py` corrects fields of the chest manifests.
- `encoders/`: the public image and text models with their preprocessing (`image_encoders.py`, `text_encoders.py`, `panel.py`), and the extraction and caching of the embeddings (`extract_embeddings.py`).
- `alignment/metrics.py`: mKNN, CKNNA, linear and RBF CKA, the Procrustes distance, and RSA.
- `alignment/`: agreement between the public models and against the random network (`convergence_map.py`), between images and reports (`vision_language.py`), against model size, release year, and width (`scaling_law.py`), and against accuracy (`utility_vs_alignment.py`). The dependence of agreement on the source dataset and view (`shared_component.py`), on the manufacturer, the institution, pretraining exposure, and preprocessing (`competing_explanations.py`), and on findings and demographic groups (`fracture.py`). The consensus configuration and its clinical structure (`consensus.py`, `ontology_analysis.py`), and severity, laterality, and the reporting radiologists on TAIX-Ray (`granularity_ladder.py`, `granularity_transfer.py`, `rater_axis.py`).
- `controlled/`: the runs of the controlled study (`matrix.py`), their training lists (`build_training_mixtures.py`), the five objectives, their training, and the check of each objective against a scrambled target (`train_encoder.py`), the replication set (`frame_import.py`), agreement and accuracy of the trained models (`converge_eval.py`), agreement with the initialization and the contrasts between objectives (`init_alignment.py`, `matrix_contrasts.py`), and classifier transfer between the trained models (`controlled_transfer.py`).
- `artifact/`: the anchors and the relative representation (`relative_reps.py`), the linear mapping between models and its number of unlabeled images (`stitching.py`), classifier transfer, the refitted classifiers, and the other chest datasets (`universal_probe.py`), and agreement against transfer (`transfer_link.py`).
- `theory/proposition.py`: the synthetic generative model and the positive control.
- `reader_study/analyze_reader_studies.py`: the comparison of the dataset labels with the answers of the radiologists, the agreement of the two radiologists, and the scoring file of the triplets, which `alignment/ontology_analysis.py` scores against the models.
- `ontology/` and `prevalence/`: the clinical reference structure and the prevalence of the findings.
- `Inference/`: bootstrap, permutation, jackknife, and FDR (`stats_utils.py`), reporting (`report_utils.py`), and resuming (`resume_utils.py`, `regimes.py`). Every analysis saves its results per unit of work, skips the units that are already finished, and resumes an interrupted run.
- `config/`: the configuration and its reader.

## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{arastehconvergence2026,
  title={Medical foundation models converge less under label supervision},
  author={Soroosh Tayebi Arasteh and Sebastian Ziegelmayer and Mahshad Lotfinia and Lisa Adams and Sven Nebelung and Jakob Nikolas Kather and Daniel Truhn},
  year={2026},
  eprint={2607.20274},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2607.20274},
}
```

## License

MIT License. See `LICENSE` for details.
