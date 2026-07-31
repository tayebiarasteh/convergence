# Self-supervision drives representational convergence in medical foundation models more than clinical supervision

## Overview

This is the official repository of the paper [**Self-supervision drives representational convergence in medical foundation models more than clinical supervision**](https://arxiv.org/abs/2607.20274).


Medical foundation models are trained independently, on different data, with different objectives, yet are increasingly assumed to learn a shared representation of pathology. This study asks whether that convergence actually holds, how strong it is, and what causes it. The pipeline embeds a panel of 22 image encoders and 7 text encoders over five imaging modalities and six chest-radiography sites, builds a cross-encoder consensus geometry from relative representations, and isolates the causal driver with a controlled-training arm that varies the training objective at fixed architecture and data. It measures convergence against a random-initialization floor, tests whether the training objective causes it, corroborates the mechanism with a synthetic generative model, relates the shared geometry to clinical comorbidity and coding taxonomy, tests for modulation by model scale, downstream performance, finding rarity, and patient demographics, evaluates cross-encoder and cross-site functional transfer with representation stitching, and tests whether convergence extends across the image-to-text boundary. An expert reader study grounding the geometry is reserved for a later revision.

## Encoder panel

The panel fixes a structure (a general-vision scale ladder, domain specialists per modality, vision-language towers, and a random-initialization floor) while the concrete checkpoints move. Identifiers resolve to exact Hugging Face checkpoints; the standing rule is to prefer the newest release of each family at load time and record the exact id and revision. Several checkpoints are gated and require an access request plus a Hugging Face token read from the config file. Encoders at or above 27B load in 4-bit via bitsandbytes. All encoders are open-weight and run locally; no closed or API models are used.

### Image encoders (22)

| Encoder | Identifier | Specialty | Role |
|---|---|---|---|
| `rad_dino` | `microsoft/rad-dino` | Chest radiography | Core |
| `txrv_densenet` | TorchXRayVision DenseNet-121 | Chest radiography | Core |
| `retfound` | RETFound (MAE) | Retinal fundus | Core |
| `uni` | `MahmoodLab/UNI` | Histopathology | Core (gated) |
| `uni2` | `MahmoodLab/UNI2-h` | Histopathology | Core (gated) |
| `virchow` | `paige-ai/Virchow` | Histopathology | Core (gated) |
| `virchow2` | `paige-ai/Virchow2` | Histopathology | Core (gated) |
| `phikon_v2` | `owkin/phikon-v2` | Histopathology | Core |
| `prov_gigapath` | `prov-gigapath/prov-gigapath` | Histopathology | Core (gated) |
| `conch_image` | `MahmoodLab/CONCH` | Histopathology VL tower | Core (gated) |
| `dinov2_large` | `facebook/dinov2-large` | General vision | Core |
| `dinov3_s` / `dinov3_b` / `dinov3_l` / `dinov3_hplus` | `facebook/dinov3-*` | General vision scale ladder | Core |
| `clip_vitl14` | `openai/clip-vit-large-patch14` | General VL tower | Core |
| `siglip2_large` | `google/siglip2-large` | General VL tower | Core |
| `biomedclip_image` | `microsoft/BiomedCLIP` | Biomedical VL tower | Core |
| `medgemma_vision` | `google/medgemma` (vision tower) | Medical VLM tower | Core (gated) |
| `llava_med_vision` | `microsoft/llava-med` (vision tower) | Medical VLM tower | Core |
| `llava_onevision_vision` | `lmms-lab/llava-onevision` (vision tower) | General VLM tower | Core |
| `random_init_vit_l` | Randomly initialized ViT-L/16 | None | Alignment floor |

### Text encoders (7)

| Encoder | Identifier | Role |
|---|---|---|
| `pubmedbert` | `microsoft/BiomedNLP-PubMedBERT-base` | Biomedical text |
| `medcpt_query` | `ncbi/MedCPT-Query-Encoder` | Biomedical retrieval |
| `medcpt_article` | `ncbi/MedCPT-Article-Encoder` | Biomedical retrieval |
| `biomedclip_text` | `microsoft/BiomedCLIP` (text tower) | Biomedical VL text |
| `conch_text` | `MahmoodLab/CONCH` (text tower) | Pathology VL text (gated) |
| `sapbert` | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` | Biomedical entity text |
| `medgemma_27b_text` | `google/medgemma-27b-it` (language tower) | Medical LLM text (gated) |

The controlled-training arm trains a separate set of encoders from scratch, crossing modality (chest radiography, histopathology), objective (self-supervised, supervised, image-text), and backbone (ViT-S, ViT-B) at fixed initialization and data. Convergence is read out by re-embedding a held-out shared pool and comparing the trained encoders pairwise.

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/tayebiarasteh/convergence.git
cd convergence
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The pipeline uses PyTorch, Hugging Face Transformers, bitsandbytes (4-bit loading for the large encoders), timm, faiss-cpu, scikit-learn, NumPy, SciPy, and pandas.

### 2. Configuration

All paths and run options live in a single YAML file under `config/config.yaml`. Two machine-specific roots are defined once at the top of the file, and every other path is built from them by interpolation, so callers always receive fully-resolved absolute paths and no path joining happens in application code. The relevant top-level keys are:

```yaml
ocean_root:   /path/to/storage/Documents      # datasets and outputs live here
home_root:    /path/to/home/Documents         # code lives here

Convergence:
  hf_token: "hf_YOUR_TOKEN_HERE"              # required for gated open checkpoints
  seed: 42
```

Set `hf_token` in the config (not an environment variable) and accept the terms on Hugging Face for any gated checkpoints in your panel (for example UNI, UNI2-h, Virchow, Virchow2, prov-gigapath, CONCH, MedGemma) for your account. This study uses no closed or API models, so no provider keys are needed.

> **Third-party data and model compliance (your responsibility).** This repository does not redistribute any dataset or model weights; it only points to public sources and loads checkpoints you obtain yourself. Before using any dataset, model, or service referenced here, YOU are responsible for reviewing and complying with its license, terms of use, data-use agreement, and any applicable privacy, ethics, or regulatory requirements for your jurisdiction and intended use. The authors make no representation that any particular use is permitted.

## Pipeline

The study runs as a sequence of stages, all exposed through `main_convergence.py`. Each stage reads from the shared config and writes intermediate results that the next stage consumes, so runs can be interrupted and resumed per unit (per encoder, per pair, or per cell). Open `main_convergence.py`, uncomment the stage you want, and run `python main_convergence.py`. The only stages that load model weights are image and text embedding extraction and controlled training; everything else is CPU.

**Resume caveat.** Each long stage writes a per-unit partial CSV and skips units already present on restart. Resume cannot detect code changes. If you edit a stage, delete its partial CSV (and its consolidated output) before re-running, or it will skip finished units and keep stale numbers.

### Build the data pools (CPU)

```bash
# inside main_convergence.py, uncomment and run, top to bottom:
main_preprocess_pcam(cfg)            # PCam H5 -> PNG
main_build_cxr_pool(cfg)             # six-site chest-radiography pool manifest
main_build_cxr_paired_reports(cfg)   # image-report pairs
main_patch_cxr_manifests(cfg)        # fix sex / subdir / age / report paths
main_build_histo_pool(cfg)           # PCam + NCT-CRC manifest
main_build_quilt_pool(cfg)           # Quilt-1M image-text manifest
main_build_fundus_pool(cfg)          # fundus pool
main_build_derm_pool(cfg)            # dermatology pool
main_build_mammo_pool(cfg)           # mammography pool
main_compute_prevalence(cfg)         # chest-radiography finding prevalence
main_build_ontology(cfg)             # ICD-10 hierarchy + comorbidity structure
```

### Extract embeddings (GPU)

```bash
# Image encoders, one at a time; each call embeds that encoder over all five pools.
# Resumable: re-run the same line to resume.
main_extract_image_embeddings(cfg, encoder_names=["rad_dino"])
# ... repeat per image encoder, including random_init_vit_l (the floor).

# Text encoders, one at a time; reads the chest-radiography paired reports.
main_extract_text_embeddings(cfg, encoder_names=["pubmedbert"])
# ... repeat per text encoder.
```

### Build consensus and run the experiments (CPU, except controlled training)

```bash
main_build_consensus(cfg)            # consensus geometry + per-encoder residuals
main_build_relative_reps(cfg)        # relative representations

main_convergence_map(cfg)            # within-modality convergence vs the floor
main_vision_language(cfg)            # image-to-text alignment
main_ontology_analysis(cfg)          # geometry vs comorbidity and coding taxonomy
main_scaling_law(cfg)                # alignment vs scale, performance, release year

main_build_training_mixtures(cfg)    # controlled-training mixtures (CPU)
main_train_one_run("cxr__ssl__vit_s__dinov3_s__seed0", cfg)   # train one cell (GPU)
# ... repeat per cell; main_list_e5_runs(cfg) prints all run ids.
main_converge_eval(cfg)              # re-embed + controlled alignment readout

main_fracture_law(cfg)               # fracture across rarity and demographics
main_universal_probe(cfg)            # cross-encoder transfer
main_stitching(cfg)                  # representation stitching
main_drift_detector(cfg)             # manifold-deviation drift detector
main_reader_study_sampling(cfg)      # reader sampling lists (reading is manual)
main_theory_proposition(cfg)         # synthetic generative model
```

### Merge the final tables

```bash
main_build_final_tables(cfg)
```

This produces the five long-format CSVs in `outputs_root/final_tables/` (`convergence_alignment.csv`, `convergence_structure.csv`, `deployable_artifact.csv`, `fracture_robustness.csv`, `theory_synthetic.csv`). These five files are the complete, self-contained quantitative record of the paper; the full interpretation reproduces from them alone.

## File overview

- `config/config.yaml` – Central configuration; two root variables at the top, everything else built by interpolation, the encoder panel under `Convergence.encoder_panel`, and the HF token.
- `config/serde.py` – Reads the config and resolves all `${var}` path variables.
- `data_build/` – One builder per data pool (chest radiography, histopathology, Quilt-1M, fundus, dermatology, mammography), the prevalence and ICD-10 ontology builders, and the shared manifest helpers.
- `encoders/image_encoders.py` – Image encoder registry, role selection, and the local frozen loader (4-bit for the large encoders).
- `encoders/text_encoders.py` – Text encoder registry and loader, including the Gemma-3 vision-language language-tower extraction for MedGemma.
- `alignment/metrics.py` – Alignment metrics (sparse CKNNA, mutual-kNN, linear and RBF CKA, RSA, Procrustes) with faiss fallback and case-id alignment.
- `alignment/consensus.py` – Generalized Procrustes consensus geometry, per-encoder residuals, and per-case deviations.
- `alignment/relative_reps.py` – Anchor sampling and relative representations.
- `alignment/convergence_map.py` – Within-modality convergence and the random-initialization floor.
- `alignment/vision_language.py` – Image-to-text alignment.
- `alignment/ontology_analysis.py` – Comorbidity and ICD-10 Mantel tests and the triplet readout.
- `alignment/scaling_law.py` – Alignment versus parameters, downstream performance, and release year.
- `alignment/fracture.py` – Fracture across finding rarity and demographic subgroups.
- `controlled/` – Controlled-training mixtures, per-cell training, and the re-embed alignment readout.
- `artifact/universal_probe.py` – Cross-encoder transfer in the shared space.
- `artifact/stitching.py` – Affine stitching between encoder spaces.
- `artifact/drift_detector.py` – Manifold-deviation drift detector.
- `artifact/reader_study.py` – Reader sampling lists for the reserved expert grounding.
- `theory/proposition.py` – Synthetic generative model.
- `probes/linear_probe.py` – Shared logistic readouts, layer-contiguous state loading, and direction-geometry helpers.
- `Inference/stats_utils.py` – Bootstrap, paired bootstrap, permutation tests, and Benjamini-Hochberg FDR; the only place the statistical constants live.
- `Inference/report_utils.py` – The single reporting layer enforcing the metric-versus-statistic formatting contract.
- `aggregate/build_final_tables.py` – Merges every per-experiment output into the five final tables.
- `main_convergence.py` – Stage orchestrator (uncomment-and-run; no argparse).

## Citation

If you use this repository, please cite our paper:

```bibtex
@misc{arastehconvergence2026,
  author={Soroosh Tayebi Arasteh and Sebastian Ziegelmayer and Mahshad Lotfinia and Lisa Adams and Sven Nebelung and Jakob Nikolas Kather and  Daniel Truhn},
  eprint={2607.20274},
  year={2026},
  url={https://arxiv.org/abs/2607.20274}, 
}
```

## License

MIT License. See `LICENSE` for details.
