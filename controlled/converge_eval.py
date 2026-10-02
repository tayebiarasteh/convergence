"""
controlled/converge_eval.py
Created on June 23, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from itertools import combinations
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from alignment.metrics import mknn, cknna_original as cknna
from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.modality_embedding_loaders import embedding_collate_fn

import warnings
from Inference.resume_utils import (append_status, status_path, MissingInput, write_csv_atomic, cell_is_done, check_build_params, done_flag_path,
                                    fingerprint_done_flag, write_build_params,
                                    write_npz_atomic)
warnings.filterwarnings("ignore")


def _held_out_manifest(cfg, pool_manifest: str, modality: str) -> str:
    import pandas as pd
    from data_loader.build_utils import read_csv_defensively
    out = os.path.join(cfg["controlled"]["results_e5_dir"], f"heldout_{modality}_manifest.csv")
    mix = os.path.join(cfg["controlled"]["ckpts_dir"], "mixtures",
                       f"{modality}_common_train.csv")
    from Inference.resume_utils import output_is_current, record_sources
    if output_is_current(out, [pool_manifest, mix], owner="heldout_manifest"):
        return out
    if not os.path.exists(pool_manifest):
        raise MissingInput(f"the {modality} pool manifest is absent at {pool_manifest}.")
    pool = read_csv_defensively(pool_manifest)
    split = pool["split"].astype(str).str.lower().replace({"val": "valid"})
    held = pool[split.isin(["test", "valid"])]
    if held.empty:
        raise MissingInput(f"the {modality} pool carries no test or validation split, so the "
                           f"controlled encoders have no evaluation set they did not train on.")
    trained_patients = set()
    if os.path.exists(mix) and "subject_id" in held.columns:
        tr = read_csv_defensively(mix)
        if "subject_id" in tr.columns:
            trained_patients = set(tr["subject_id"].astype(str))
        held = held[~held["subject_id"].astype(str).isin(trained_patients)]
    if held.empty:
        raise MissingInput(f"every held-out {modality} case shares a patient with the training "
                           f"mixture, so no patient-disjoint evaluation set exists.")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    write_csv_atomic(held, out)
    record_sources(out, [pool_manifest, mix])
    return out


def _load_cell_model(run_id: str, ckpt_dir: str, device: torch.device):
    model_pt = os.path.join(ckpt_dir, "model.pt")
    if not os.path.exists(model_pt):
        from transformers import AutoModel
        try:
            model = AutoModel.from_pretrained(ckpt_dir, trust_remote_code=True)
        except Exception as e:
            raise MissingInput(
                f"[E5] {run_id}: the cell directory carries neither a loadable model.pt nor a "
                f"loadable Hugging Face checkpoint ({type(e).__name__}: {e}). A cell that cannot "
                f"be loaded must not disappear from the matrix in silence; delete "
                f"{ckpt_dir} and retrain it, or fix the import.") from e
    else:
        backbone = "vit_small_patch16_224" if "vit_s" in run_id else "vit_base_patch16_224"
        import timm
        model = timm.create_model(backbone, pretrained=False, num_classes=0)
        model.load_state_dict(torch.load(model_pt, map_location="cpu"))
    return model.to(device=device).eval()


def _eval_transform():
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def _forward_batch(model, tensors: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_features"):
        feats = model.forward_features(tensors)
        emb = feats[:, 0, :] if feats.dim() == 3 else feats
    else:
        out = model(tensors)
        emb = out.last_hidden_state[:, 0, :] if hasattr(out, "last_hidden_state") else out
    return torch.nn.functional.normalize(emb.float(), p=2, dim=-1)


def _extract_anchor_embeddings(run_id: str, ckpt_dir: str, cfg: dict,
                               device: torch.device, output_dir: str) -> Optional[str]:
    out_path = os.path.join(output_dir, f"{run_id}__anchors.npz")
    expected = {"cell": fingerprint_done_flag(done_flag_path(ckpt_dir))}
    if os.path.exists(out_path):
        if check_build_params(out_path, expected, owner="converge_eval_anchors"):
            return out_path
    anchors_csv = os.path.join(cfg["alignment"]["consensus_dir"], "anchor_case_ids.csv")
    if not os.path.exists(anchors_csv):
        raise MissingInput(f"the anchor case ids are absent at {anchors_csv}; run main_build_consensus before "
                           f"the controlled transfer.")
    from alignment.consensus import read_anchor_case_ids
    from data_loader.cxr_harmonization import resolve_cxr_image_path
    anchor_ids = read_anchor_case_ids(anchors_csv)
    pool = read_csv_defensively(cfg["cxr"]["pool_manifest_csv"])
    sub = pool[pool["case_id"].astype(str).isin(set(anchor_ids))]
    key_of = dict(zip(sub["case_id"].astype(str), sub["image_key"].astype(str)))
    missing = [c for c in anchor_ids if c not in key_of]
    if missing:
        raise MissingInput(f"{len(missing)} anchor case ids are not in the chest pool manifest.")
    root = cfg["cxr"]["sites"]["mimic"]["image_root"]
    model = _load_cell_model(run_id, ckpt_dir, device)
    tfm = _eval_transform()
    from PIL import Image as _Image
    embs, bs = [], 64
    with torch.no_grad():
        for i in tqdm(range(0, len(anchor_ids), bs),
                      desc=f"[anchors] {run_id[:28]}", unit="batch"):
            imgs = [tfm(_Image.open(resolve_cxr_image_path(
                "mimic", root, key_of[c])).convert("RGB"))
                    for c in anchor_ids[i:i + bs]]
            embs.append(_forward_batch(model, torch.stack(imgs).to(device)).cpu().numpy())
    arr = np.concatenate(embs, axis=0)
    write_npz_atomic(out_path, embeddings=arr.astype(np.float32, copy=False),
                     case_ids=np.array(anchor_ids, dtype=object))
    write_build_params(out_path, expected)
    return out_path


def _extract_controlled_embeddings(
    run_id: str,
    ckpt_dir: str,
    modality: str,
    cfg: dict,
    device: torch.device,
    output_dir: str,
) -> Optional[str]:
    out_path = os.path.join(output_dir, f"{run_id}.npz")
    expected = {"cell": fingerprint_done_flag(done_flag_path(ckpt_dir))}
    if os.path.exists(out_path):
        if check_build_params(out_path, expected, owner="converge_eval"):
            return out_path

    from controlled.matrix import arm_sources
    from data_loader.modality_embedding_loaders import LOADER_REGISTRY
    image_modality, pool_block = arm_sources(modality)
    loader_cls = LOADER_REGISTRY.get(image_modality)
    if loader_cls is None:
        raise MissingInput(
            f"[E5] the {modality} arm reads {image_modality} images and LOADER_REGISTRY carries "
            f"no loader for that modality; it holds {sorted(LOADER_REGISTRY)}.")
    held_out = _held_out_manifest(cfg, cfg[pool_block]["pool_manifest_csv"], modality)

    model = _load_cell_model(run_id, ckpt_dir, device)
    tfm = _eval_transform()
    cfg_path = cfg.get("global_config_path", "")
    dataset = loader_cls(cfg_path=cfg_path, manifest_csv=held_out)
    _nw = int(cfg["controlled"].get("dataloader_workers", 4))
    loader  = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=_nw,
                         collate_fn=embedding_collate_fn,
                         **({"persistent_workers": True, "prefetch_factor": 4} if _nw > 0 else {}))

    all_emb = []
    all_ids = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"[converge_eval] {run_id[:30]}", unit="batch"):
            tensors = torch.stack([tfm(im) for im in batch["images"]]).to(device)
            all_emb.append(_forward_batch(model, tensors).cpu().numpy())
            all_ids.extend(batch["case_ids"])

    embeddings = np.concatenate(all_emb, axis=0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_npz_atomic(out_path, embeddings=embeddings.astype(np.float32, copy=False),
                     case_ids=np.array(all_ids, dtype=object))
    write_build_params(out_path, expected)
    return out_path


def _seed_pair_partials(out_dir: str, partials: str, expected_pairs: set) -> int:
    from Inference.resume_utils import ensure_dir, unit_cache_path, write_unit_rows
    legacy = os.path.join(out_dir, "e5_alignment_table.partial.csv")
    final = os.path.join(out_dir, "e5_alignment_table.csv")
    sources = [q for q in (legacy, final) if os.path.exists(q)]
    if not sources:
        return 0
    ensure_dir(partials)
    seeded = 0
    for src in sources:
        try:
            frame = pd.read_csv(src)
        except Exception as exc:
            continue
        if not {"run_id_a", "run_id_b"}.issubset(frame.columns):
            continue
        for row in frame.to_dict("records"):
            ra, rb = str(row["run_id_a"]), str(row["run_id_b"])
            if expected_pairs and frozenset((ra, rb)) not in expected_pairs:
                continue
            unit = f"{ra}|{rb}"
            if os.path.exists(unit_cache_path(partials, "e5_pairs", unit)):
                continue
            write_unit_rows(partials, "e5_pairs", unit,
                            [{k: (None if pd.isna(v) else v) for k, v in row.items()}])
            seeded += 1
    return seeded


def main_converge_eval(global_config_path: str, force: bool = False) -> str:
    cfg  = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    out_dir = ctrl["results_e5_dir"]
    emb_dir = os.path.join(out_dir, "embeddings")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(emb_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "e5_alignment_table.csv")
    status = status_path(cfg, "converge_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from controlled.matrix import (enumerate_disjoint_runs, enumerate_e5_runs,
                                   enumerate_e11_runs, enumerate_frame_runs, pairing_group)
    matrix = [(r[0], r[1]) for r in enumerate_e5_runs(global_config_path)]
    matrix += [(r[0], "taix") for r in enumerate_e11_runs(global_config_path)]
    matrix += [(r[0], "cxr") for r in enumerate_frame_runs(global_config_path)]
    matrix += [(r[0], r[1]) for r in enumerate_disjoint_runs(global_config_path)]
    cells = []
    for run_id, mod in matrix:
        ckpt_dir = os.path.join(ctrl["ckpts_dir"], run_id)
        if cell_is_done(ckpt_dir):
            cells.append((run_id, ckpt_dir, mod))
    if not cells:
        raise MissingInput(
            f"no trained cell carries a done flag under {ctrl['ckpts_dir']}. "
            f"Train the controlled models before this stage.")
    from Inference.resume_utils import (clear_partial_dir, heartbeat_claim, run_units_resumable,
                                        unit_cache_path, write_unit_rows)
    partials = os.path.join(out_dir, "e5_partials")
    if force:
        clear_partial_dir(partials)

    expected_pairs = {frozenset((ra, rb))
                      for (ra, _, ma), (rb, _, mb) in combinations(cells, 2)
                      if pairing_group(ra) == pairing_group(rb)}
    if os.path.exists(out_csv) and not force:
        have = pd.read_csv(out_csv)
        got = set()
        if {"run_id_a", "run_id_b"}.issubset(have.columns):
            got = {frozenset((str(r), str(t)))
                   for r, t in zip(have["run_id_a"], have["run_id_b"])}
        if expected_pairs and expected_pairs <= got:
            done_cache = {rid: os.path.join(emb_dir, f"{rid}.npz") for rid, _c, _m in cells
                          if os.path.exists(os.path.join(emb_dir, f"{rid}.npz"))}
            _cell_utility(cfg, global_config_path, cells, done_cache, out_dir, partials, status)
            return out_dir

    _seed_pair_partials(out_dir, partials, expected_pairs)

    transfer_on = bool((ctrl.get("transfer", {}) or {}).get("enabled", True))
    cell_of = {run_id: (ckpt_dir, mod) for run_id, ckpt_dir, mod in cells}

    def _embed_unit(run_id: str) -> List[Dict]:
        ckpt_dir, mod = cell_of[run_id]
        path = _extract_controlled_embeddings(run_id, ckpt_dir, mod, cfg, device, emb_dir)
        if transfer_on and mod in ("cxr", "cxrh1", "cxrh2"):
            _extract_anchor_embeddings(run_id, ckpt_dir, cfg, device, emb_dir)
        return [{"run_id": run_id, "modality": mod, "path": path}]

    emb_frame = run_units_resumable(
        partial_dir=partials, group="e5_embed", units=[c[0] for c in cells],
        compute_unit=_embed_unit,
        build_params={"seed": int(cfg.get("seed", 42)),
                      "matched_n": int(cfg["alignment"]["matched_n"])},
        progress_desc="[E5] cells", use_claims=True, status_file=status)
    if int(emb_frame.attrs.get("n_skipped", 0)):
        raise MissingInput(f"{emb_frame.attrs['n_skipped']} of {len(cells)} cells could not be "
                           f"embedded; the finished ones are cached and skip.")

    emb_cache: Dict[str, str] = {}
    for run_id, _ckpt, _mod in cells:
        path = os.path.join(emb_dir, f"{run_id}.npz")
        if os.path.exists(path):
            emb_cache[run_id] = path

    all_pairs = [(ra, rb) for (ra, _c, _m), (rb, _d, _n) in combinations(cells, 2)
                 if pairing_group(ra) == pairing_group(rb)]

    _loaded: Dict[str, tuple] = {}

    def _emb(rid: str):
        if rid not in _loaded:
            if rid not in emb_cache:
                raise MissingInput(f"no held-out embedding cache for {rid}; another job is "
                                   f"still embedding that cell.")
            d = np.load(emb_cache[rid], allow_pickle=True)
            _loaded[rid] = (d["embeddings"].astype(np.float32, copy=False),
                            d["case_ids"].astype(str))
        return _loaded[rid]

    matched_n = int(cfg["alignment"]["matched_n"])
    n_boot_e5 = int(cfg["stats"]["n_boot_neighbor"])
    k_fraction = float(cfg["alignment"]["k_fraction"])

    def _pair_unit(unit: str) -> List[Dict]:
        rid_a, rid_b = unit.split("|", 1)
        heartbeat_claim(partials, f"e5_pairs__{unit}")
        for old in [r for r in _loaded if r not in (rid_a, rid_b)]:
            _loaded.pop(old, None)
        ea, ids_a = _emb(rid_a)
        eb, ids_b = _emb(rid_b)
        parts_a, parts_b = rid_a.split("__"), rid_b.split("__")

        from alignment.metrics import align_by_case_ids
        try:
            a, b, _ = align_by_case_ids(ea, ids_a, eb, ids_b)
        except ValueError:
            return []
        finite = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
        a, b = a[finite], b[finite]
        if len(a) < 100:
            return []
        n = min(len(a), 4 * matched_n)
        rng = np.random.RandomState(42)
        idx = rng.choice(len(a), size=n, replace=False)
        a_s, b_s = a[idx], b[idx]

        from Inference.stats_utils import assert_interval_brackets, subsample_statistic
        if n < 2 * matched_n:
            raise MissingInput(
                f"[E5] {rid_a} vs {rid_b}: {len(a)} shared held-out rows allow no fresh draw of "
                f"the matched N of {matched_n}, so this pair cannot be measured at the size every "
                f"other alignment statement uses. Enlarge the held-out pool or lower "
                f"alignment.matched_n for the whole study; a pair at its own size is not comparable.")
        n_sub = matched_n
        k_align = max(1, int(round(k_fraction * n_sub)))
        mknn_bs = subsample_statistic(
            [a_s, b_s], lambda x, y: mknn(x, y, k=k_align), n_sub=n_sub, n_boot=n_boot_e5)
        cknna_bs = subsample_statistic(
            [a_s, b_s], lambda x, y: cknna(x, y, k=k_align), n_sub=n_sub, n_boot=n_boot_e5)
        assert_interval_brackets(mknn_bs["point"], mknn_bs["ci_lower"], mknn_bs["ci_upper"],
                                 label=f"mknn {rid_a} vs {rid_b}")
        assert_interval_brackets(cknna_bs["point"], cknna_bs["ci_lower"], cknna_bs["ci_upper"],
                                 label=f"cknna {rid_a} vs {rid_b}")

        is_e11 = parts_a[0] == "taix"
        arm = pairing_group(rid_a)
        return [{
            "run_id_a":    rid_a, "run_id_b": rid_b,
            "modality":    arm,
            "arm":         arm,
            "provenance":  "frame" if arm.startswith("frame_") else "trained_here",
            "experiment":  ("E11" if is_e11 else
                            "E5_disjoint" if arm == "cxr_disjoint" else "E5"),
            "same_training_data": parts_a[0] == parts_b[0],
            "objective_a": None if is_e11 else parts_a[1],
            "objective_b": None if is_e11 else parts_b[1],
            "label_form_a": parts_a[1] if is_e11 else None,
            "label_form_b": parts_b[1] if is_e11 else None,
            "backbone_a":  parts_a[2], "backbone_b": parts_b[2],
            "init_a":      parts_a[3], "init_b": parts_b[3],
            "same_axis_level": parts_a[1] == parts_b[1],
            "same_objective": (None if is_e11 else parts_a[1] == parts_b[1]),
            "same_seed":   rid_a.split("seed")[1] == rid_b.split("seed")[1],
            "n":           n,
            "n_sub":       n_sub,
            "k":           k_align,
            "resample":    "subsample_without_replacement",
            "mknn":            round(mknn_bs["point"], 6),
            "mknn_std":        round(mknn_bs["std"], 6),
            "mknn_ci_lower":   round(mknn_bs["ci_lower"], 6),
            "mknn_ci_upper":   round(mknn_bs["ci_upper"], 6),
            "cknna":           round(cknna_bs["point"], 6),
            "cknna_std":       round(cknna_bs["std"], 6),
            "cknna_ci_lower":  round(cknna_bs["ci_lower"], 6),
            "cknna_ci_upper":  round(cknna_bs["ci_upper"], 6),
        }]

    df = run_units_resumable(
        partial_dir=partials, group="e5_pairs",
        units=[f"{ra}|{rb}" for ra, rb in all_pairs], compute_unit=_pair_unit,
        build_params={"matched_n": matched_n, "n_boot_neighbor": n_boot_e5,
                      "k_fraction": k_fraction, "seed": 42,
                      "statistic": "mknn_and_cknna_subsampled_without_replacement"},
        progress_desc="[E5] align pairs", use_claims=True, status_file=status)
    n_skipped = int(df.attrs.get("n_skipped", 0))
    if n_skipped:
        raise MissingInput(
            f"{n_skipped} of {len(all_pairs)} pairs could not run because a cell another job "
            f"owns is not embedded yet, so the table would be short of pairs. Every finished "
            f"pair is cached and skips; run this stage again once those cells are done.")
    write_csv_atomic(df, out_csv)
    append_status(status, f"E5 and E11 alignment: {len(df)} pair rows over {len(cells)} cells")

    _cell_utility(cfg, global_config_path, cells, emb_cache, out_dir, partials, status)

    if not df.empty:
        summary = df.groupby(["modality","objective_a","objective_b"])["mknn"].agg(
            ["mean","std","count"]
        ).reset_index()
        write_csv_atomic(summary, os.path.join(out_dir, "e5_summary.csv"))

    return out_dir

def _cell_utility(cfg: dict, global_config_path: str, cells, emb_cache: Dict[str, str],
                  out_dir: str, partials: str, status: str) -> str:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from Inference.regimes import regime_extra
    from Inference.report_utils import report_accuracy, report_metric
    from Inference.resume_utils import (read_jsonl, record_sources, run_units_resumable,
                                        unit_cache_path)
    from data_loader.cxr_harmonization import CANONICAL_CXR_FINDINGS

    out_csv = os.path.join(out_dir, "e5_cell_utility.csv")
    seed = int(cfg.get("seed", 42))
    manifests = {}
    mod_of = {run_id: mod for run_id, _ckpt, mod in cells}
    acc_rows: Dict[str, Dict] = {}
    extra = regime_extra("e5_utility")
    if os.path.exists(out_csv):
        from Inference.resume_utils import read_build_params
        if (read_build_params(out_csv) or {}).get("regime") == extra["regime"]:
            have = set(pd.read_csv(out_csv, usecols=["run_id"])["run_id"].astype(str))
            if set(mod_of) <= have:
                return out_csv

    def _utility_unit(run_id: str) -> List[Dict]:
        rows: List[Dict] = []
        mod = mod_of[run_id]
        path = emb_cache.get(run_id)
        if not path:
            raise MissingInput(f"no held-out embedding cache for {run_id}; another job is still "
                               f"embedding that cell.")
        if mod not in manifests:
            held = os.path.join(cfg["controlled"]["results_e5_dir"], f"heldout_{mod}_manifest.csv")
            if not os.path.exists(held):
                return rows
            manifests[mod] = read_csv_defensively(held)
        man = manifests[mod]
        if mod == "histo":
            targets = ["tissue_class"] if "tissue_class" in man.columns else []
        else:
            targets = [f for f in CANONICAL_CXR_FINDINGS if f in man.columns]
        if not targets:
            return rows
        d = np.load(path, allow_pickle=True)
        emb = d["embeddings"].astype(np.float32, copy=False)
        pos = {str(c): i for i, c in enumerate(d["case_ids"].astype(str))}
        sub = man[man["case_id"].astype(str).isin(pos)].reset_index(drop=True)
        if len(sub) < 400:
            return rows
        idx = np.array([pos[c] for c in sub["case_id"].astype(str)])
        X = emb[idx]
        rng = np.random.RandomState(seed)
        if "subject_id" in sub.columns and sub["subject_id"].notna().any():
            subs = sub["subject_id"].astype(str).unique()
            rng.shuffle(subs)
            tr = sub["subject_id"].astype(str).isin(set(subs[: int(0.7 * len(subs))])).values
        else:
            tr = rng.random_sample(len(sub)) < 0.7
        te = ~tr
        aurocs = []
        for t in targets:
            if t == "tissue_class":
                y = sub[t].astype(str).values
                declared = {str(c) for c in cfg["histo"]["nct_crc"]["classes"]}
                classes = [c for c, n in pd.Series(y[tr]).value_counts().items()
                           if n >= 20 and c in declared]
                m = pd.Series(y).isin(classes).values & np.isfinite(X).all(axis=1)
                if (m & tr).sum() < 100 or (m & te).sum() < 50 or len(classes) < 2:
                    continue
                sc = StandardScaler().fit(X[m & tr])
                clf = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                                         random_state=seed).fit(sc.transform(X[m & tr]), y[m & tr])
                correct = (clf.predict(sc.transform(X[m & te])) == y[m & te]).astype(float)
                acc = float(correct.mean())
                aurocs.append(acc)
                pat = (sub["subject_id"].astype(str).values[m & te]
                       if "subject_id" in sub.columns and sub["subject_id"].notna().all() else None)
                acc_rep = report_accuracy(correct, prefix="utility", cluster_ids=pat)
                acc_rows[run_id] = acc_rep
                rows.append({"run_id": run_id, "modality": mod, "target": t,
                             "metric": "accuracy", "value_raw": acc,
                             "n_train": int((m & tr).sum()), "n_test": int((m & te).sum()),
                             **acc_rep})
                continue
            y = pd.to_numeric(sub[t], errors="coerce").values
            m = np.isfinite(y) & np.isfinite(X).all(axis=1)
            if (m & tr).sum() < 100 or (m & te).sum() < 50:
                continue
            if len(np.unique(y[m & tr])) < 2 or len(np.unique(y[m & te])) < 2:
                continue
            sc = StandardScaler().fit(X[m & tr])
            clf = LogisticRegression(C=0.1, max_iter=200, class_weight="balanced",
                                     random_state=seed).fit(sc.transform(X[m & tr]), y[m & tr])
            a = float(roc_auc_score(y[m & te], clf.predict_proba(sc.transform(X[m & te]))[:, 1]))
            aurocs.append(a)
            rows.append({"run_id": run_id, "modality": mod, "target": t, "metric": "auroc",
                         "value_raw": a, "n_train": int((m & tr).sum()),
                         "n_test": int((m & te).sum())})
        if aurocs:
            if len(aurocs) == 1 and run_id in acc_rows:
                rep = dict(acc_rows[run_id])
            else:
                rep = report_metric(np.array(aurocs), prefix="utility")
            parts = run_id.split("__")
            rep.update({"run_id": run_id, "modality": parts[0], "axis_level": parts[1],
                        "backbone": parts[2], "init": parts[3], "target": "MEAN",
                        "metric": "mean_over_targets", "n_targets": len(aurocs)})
            rows.append(rep)
        return rows

    def _utility_unit_current(run_id: str) -> List[Dict]:
        if mod_of[run_id] != "histo":
            old = unit_cache_path(partials, "e5_utility", run_id)
            carried = read_jsonl(old) if os.path.exists(old) else []
            if carried and not any(r.get("metric") == "accuracy" for r in carried):
                return carried
        return _utility_unit(run_id)

    df = run_units_resumable(
        partial_dir=partials, group="e5_utility_v2", units=[c[0] for c in cells],
        compute_unit=_utility_unit_current,
        build_params={"seed": seed, "split": "patient_disjoint_0.7", **extra},
        progress_desc="[E5] cell utility", use_claims=True, status_file=status)
    if int(df.attrs.get("n_skipped", 0)):
        raise MissingInput(f"{df.attrs['n_skipped']} of {len(cells)} utility units could not run "
                           f"because their cell is not embedded in this job; the finished ones "
                           f"are cached and skip.")
    if df.empty:
        raise MissingInput("no controlled cell produced a utility row; check the held-out "
                           "manifests written by _held_out_manifest.")
    write_csv_atomic(df, out_csv)
    record_sources(out_csv, [], extra=extra)
    return out_csv
