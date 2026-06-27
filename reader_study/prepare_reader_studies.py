"""
reader_study/prepare_reader_studies.py
Created on May 29, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
import random
import shutil
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image

from config.serde import read_config
from data_loader.build_utils import read_csv_defensively
from data_loader.cxr_harmonization import resolve_cxr_image_path


RANDOM_SEED = 20260617


def _set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)



CXR_CORE_FINDINGS = [
    "atelectasis", "cardiomegaly", "consolidation", "edema",
    "enlarged_cardiomediastinum", "fracture", "lung_lesion", "lung_opacity",
    "pleural_effusion", "pleural_other", "pneumonia", "pneumothorax",
    "support_devices",
]
CXR_RARE_FINDINGS = [
    "emphysema", "pulmonary_fibrosis", "bronchiectasis", "tuberculosis",
    "ild", "cavitation", "lung_tumor",
]

CXR_DISPLAY = {
    "atelectasis": "atelectasis", "cardiomegaly": "cardiomegaly",
    "consolidation": "consolidation", "edema": "pulmonary edema",
    "enlarged_cardiomediastinum": "enlarged cardiomediastinum",
    "fracture": "rib fracture", "lung_lesion": "lung lesion",
    "lung_opacity": "lung opacity", "pleural_effusion": "pleural effusion",
    "pleural_other": "pleural abnormality", "pneumonia": "pneumonia",
    "pneumothorax": "pneumothorax", "support_devices": "support device",
    "emphysema": "emphysema", "pulmonary_fibrosis": "pulmonary fibrosis",
    "bronchiectasis": "bronchiectasis", "tuberculosis": "tuberculosis",
    "ild": "interstitial lung disease", "cavitation": "cavitation",
    "lung_tumor": "lung tumor",
}

HISTO_CLASS_DISPLAY = {
    "ADI": "adipose tissue", "BACK": "background / empty",
    "DEB": "debris", "LYM": "lymphocytes", "MUC": "mucus",
    "MUS": "smooth muscle", "NORM": "normal colon mucosa",
    "STR": "cancer-associated stroma", "TUM": "colorectal adenocarcinoma epithelium",
    "tumor": "tumor (metastatic)", "non_tumor": "non-tumor",
}


def _cxr_question(finding: str) -> str:
    return f"Is {CXR_DISPLAY.get(finding, finding.replace('_', ' '))} present in this chest X-ray?"


def _histo_question(tissue_class: str) -> str:
    disp = HISTO_CLASS_DISPLAY.get(tissue_class, str(tissue_class))
    return f"Is the dominant tissue in this patch {disp}?"


def _render_cxr(site_roots: dict, row: pd.Series, dst_path: str,
                resolution: int = 512):
    dataset = str(row["dataset"])
    image_root = site_roots.get(dataset)
    if image_root is None:
        raise KeyError(f"No image_root configured for CXR site '{dataset}'")
    subdir = row.get("image_subdir", None)
    if pd.isna(subdir) or str(subdir).strip().lower() in ("", "nan", "none"):
        subdir = None
    else:
        subdir = str(subdir)
        if subdir.endswith(".0"):
            subdir = subdir[:-2]
    src = resolve_cxr_image_path(
        dataset=str(row["dataset"]), image_root=image_root,
        image_key=str(row.get("image_key", "")), resolution=resolution,
        split=str(row.get("split", "")) or None, image_subdir=subdir,
    )
    img = Image.open(src).convert("RGB")
    img.save(dst_path)


def _render_histo(histo_root: str, quilt_root: str, nct_root: str, pcam_root: str,
                  row: pd.Series, dst_path: str, display_size: int = 512):
    ds  = str(row["dataset"])
    key = str(row["image_key"])
    if ds == "pcam":
        src = os.path.join(pcam_root, key)
    elif ds == "nct_crc":
        src = os.path.join(nct_root, key)
    elif ds == "quilt":
        src = os.path.join(quilt_root, key)
    else:
        raise ValueError(f"Unknown histo dataset: {ds}")
    img = Image.open(src).convert("RGB")
    if display_size and display_size != img.size[0]:
        img = img.resize((display_size, display_size), Image.LANCZOS)
    img.save(dst_path)



def _build_triplets(gold: pd.DataFrame, n_triplets: int, id_prefix: str,
                    seed: int) -> pd.DataFrame:
    rng  = np.random.RandomState(seed)
    ids  = gold["case_id"].tolist()
    n    = len(ids)
    seen = set()
    rows = []
    attempts = 0
    while len(rows) < n_triplets and attempts < n_triplets * 20:
        attempts += 1
        ia, ib, ic = rng.choice(n, size=3, replace=False)
        keytrip = tuple(sorted((ids[ia], ids[ib], ids[ic])))
        if keytrip in seen:
            continue
        seen.add(keytrip)
        rows.append({
            "triplet_id": f"{id_prefix}_{len(rows)+1:04d}",
            "case_a": ids[ia], "case_b": ids[ib], "case_c": ids[ic],
        })
    return pd.DataFrame(rows)


def _assign_triplet_sessions(trips: pd.DataFrame, n_sessions: int,
                             seed: int) -> pd.DataFrame:
    """Split triplets evenly across sessions and shuffle within each session."""
    rng = np.random.RandomState(seed)
    trips = trips.sample(frac=1, random_state=seed).reset_index(drop=True)
    trips["session"] = (np.arange(len(trips)) % n_sessions) + 1
    blocks = []
    for s in range(1, n_sessions + 1):
        blk = trips[trips["session"] == s].sample(
            frac=1, random_state=rng.randint(0, 1_000_000))
        blocks.append(blk)
    return pd.concat(blocks, ignore_index=True)


CXR_USECOLS = (["case_id", "dataset", "split", "image_key", "image_subdir",
                "sex", "age", "race", "view"]
               + CXR_CORE_FINDINGS + CXR_RARE_FINDINGS)


def _load_cxr_manifest(path: str) -> pd.DataFrame:
    cols = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [c for c in CXR_USECOLS if c in cols]
    return pd.read_csv(path, usecols=usecols, low_memory=False)


def _sample_cxr_gold(manifest: pd.DataFrame, n_target: int, seed: int,
                     core_cap: int, rare_cap: int,
                     exclude_ids: Optional[set] = None) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    df = manifest[manifest["split"] == "test"].copy()
    if exclude_ids:
        df = df[~df["case_id"].astype(str).isin(exclude_ids)]
    chosen_rows = []
    used_ids = set()

    plan = [(f, core_cap) for f in CXR_CORE_FINDINGS] + \
           [(f, rare_cap) for f in CXR_RARE_FINDINGS]
    for finding, cap in plan:
        if finding not in df.columns:
            continue
        col = pd.to_numeric(df[finding], errors="coerce")
        n_pos = cap - cap // 2
        n_neg = cap // 2

        def _site_spread(candidate_df, n_take):
            if len(candidate_df) == 0 or n_take == 0:
                return candidate_df.iloc[0:0]
            picks = []
            sites = sorted(candidate_df["dataset"].unique())
            per_site = max(1, n_take // max(1, len(sites)))
            for site in sites:
                sub = candidate_df[candidate_df["dataset"] == site]
                picks.append(sub.sample(min(per_site, len(sub)),
                                        random_state=rng.randint(0, 1_000_000)))
            out = pd.concat(picks, ignore_index=True)
            if len(out) > n_take:
                out = out.sample(n_take, random_state=rng.randint(0, 1_000_000))
            return out

        pos_cand = df[(col == 1.0) & (~df["case_id"].astype(str).isin(used_ids))]
        # Negatives: the asked finding is absent AND the case is not no_finding-only
        # noise; we simply require this finding == 0 (any other findings allowed).
        neg_cand = df[(col == 0.0) & (~df["case_id"].astype(str).isin(used_ids))]
        picked_pos = _site_spread(pos_cand, n_pos)
        used_ids.update(picked_pos["case_id"].astype(str).tolist())
        neg_cand = neg_cand[~neg_cand["case_id"].astype(str).isin(used_ids)]
        picked_neg = _site_spread(neg_cand, n_neg)
        used_ids.update(picked_neg["case_id"].astype(str).tolist())

        for picked in (picked_pos, picked_neg):
            if len(picked) == 0:
                continue
            picked = picked.copy()
            picked["target_finding"] = finding
            chosen_rows.append(picked)

    gold = pd.concat(chosen_rows, ignore_index=True)
    if len(gold) > n_target:
        gold = gold.sample(n_target, random_state=seed)
    gold = gold.sample(frac=1, random_state=seed).reset_index(drop=True)
    gold["display_id"] = [f"G_{i+1:04d}" for i in range(len(gold))]
    return gold



def _sample_histo_gold(manifest: pd.DataFrame, n_target: int, seed: int,
                       per_class: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    classes = sorted(manifest["tissue_class"].dropna().astype(str).unique())
    picks = []
    for cls in classes:
        sub = manifest[manifest["tissue_class"].astype(str) == cls]
        take = sub.sample(min(per_class, len(sub)),
                          random_state=rng.randint(0, 1_000_000)).copy()
        n = len(take)
        n_yes = n - n // 2
        # First n_yes asked about their own class; rest asked about another class.
        target = [cls] * n_yes
        others = [c for c in classes if c != cls]
        for _ in range(n - n_yes):
            target.append(others[rng.randint(0, len(others))])
        take = take.sample(frac=1, random_state=rng.randint(0, 1_000_000)).reset_index(drop=True)
        take["target_class"] = target
        picks.append(take)
    gold = pd.concat(picks, ignore_index=True)
    if len(gold) > n_target:
        gold = gold.sample(n_target, random_state=seed)
    gold = gold.sample(frac=1, random_state=seed).reset_index(drop=True)
    gold["display_id"] = [f"H_{i+1:04d}" for i in range(len(gold))]
    return gold



class Reader1Preparer:
    N_GOLD     = 480
    N_TRIPLETS = 300
    CORE_CAP   = 40
    RARE_CAP   = 14
    N_SESSIONS = 3

    def __init__(self, cfg, site_roots: dict, out_root: str, manifest: pd.DataFrame):
        self.cfg = cfg
        self.site_roots = site_roots
        self.out_root = out_root
        self.manifest = manifest

    def prepare(self) -> pd.DataFrame:
        _set_seed()
        gold = _sample_cxr_gold(self.manifest, self.N_GOLD, RANDOM_SEED,
                                self.CORE_CAP, self.RARE_CAP)

        # ---- Task A: gold labels ----
        a_dir = os.path.join(self.out_root, "task_A_gold_labels")
        a_img = os.path.join(a_dir, "images")
        os.makedirs(a_img, exist_ok=True)
        for _, r in gold.iterrows():
            _render_cxr(self.site_roots, r, os.path.join(a_img, f"{r['display_id']}.png"))

        reader = pd.DataFrame({
            "display_id": gold["display_id"],
            "question":   gold["target_finding"].map(_cxr_question),
            "finding_present": "",          # Yes / No
            "image_quality":   "",          # adequate / suboptimal / non-diagnostic
            "distrust_automated_read": "",  # Yes / No
            "comment": "",
        })
        reader.to_csv(os.path.join(a_dir, "cases.csv"), index=False)
        _save_cxr_mapping(gold, os.path.join(a_dir, "case_mapping.csv"))

        b_dir = os.path.join(self.out_root, "task_B_similarity")
        os.makedirs(os.path.join(b_dir, "images"), exist_ok=True)
        trips = _build_triplets(gold, self.N_TRIPLETS, "T1", RANDOM_SEED + 1)
        trips = _assign_triplet_sessions(trips, self.N_SESSIONS, RANDOM_SEED + 2)
        # Render the three member images per triplet under triplet-scoped ids.
        gold_by_id = gold.set_index("case_id")
        for _, t in trips.iterrows():
            for slot, cid in [("a", t["case_a"]), ("b", t["case_b"]), ("c", t["case_c"])]:
                r = gold_by_id.loc[cid]
                dst = os.path.join(b_dir, "images", f"{t['triplet_id']}_{slot}.png")
                if not os.path.exists(dst):
                    _render_cxr(self.site_roots, r, dst)
        reader_t = trips[["triplet_id", "session"]].copy()
        reader_t["images"] = reader_t["triplet_id"].map(
            lambda t: f"reference={t}_a.png | option_B={t}_b.png | option_C={t}_c.png")
        reader_t["more_similar_to"] = ""   # B / C
        reader_t["confidence"] = ""        # 1 - 5
        reader_t["comment"] = ""
        reader_t.to_csv(os.path.join(b_dir, "cases.csv"), index=False)
        trips.to_csv(os.path.join(b_dir, "case_mapping.csv"), index=False)
        return gold



class Reader2Preparer:
    N_OVERLAP        = 150
    N_EXTENSION      = 150
    N_TRIPLET_OVERLAP = 100
    CORE_CAP   = 14
    RARE_CAP   = 6
    N_SESSIONS = 3

    def __init__(self, cfg, site_roots: dict, out_root: str, manifest: pd.DataFrame,
                 r1_gold: pd.DataFrame, r1_out_root: str):
        self.cfg = cfg
        self.site_roots = site_roots
        self.out_root = out_root
        self.manifest = manifest
        self.r1_gold = r1_gold
        self.r1_out_root = r1_out_root

    def prepare(self):
        _set_seed()

        c_dir = os.path.join(self.out_root, "task_C_agreement")
        c_img = os.path.join(c_dir, "images")
        os.makedirs(c_img, exist_ok=True)
        overlap = self.r1_gold.head(self.N_OVERLAP).copy()
        # Re-anonymize so R2 cannot align to R1 display ids.
        overlap = overlap.sample(frac=1, random_state=RANDOM_SEED + 5).reset_index(drop=True)
        overlap["r2_display_id"] = [f"C_{i+1:04d}" for i in range(len(overlap))]
        for _, r in overlap.iterrows():
            _render_cxr(self.site_roots, r, os.path.join(c_img, f"{r['r2_display_id']}.png"))
        reader_c = pd.DataFrame({
            "display_id": overlap["r2_display_id"],
            "question":   overlap["target_finding"].map(_cxr_question),
            "finding_present": "",
            "image_quality":   "",
            "distrust_automated_read": "",
            "comment": "",
        })
        reader_c.to_csv(os.path.join(c_dir, "cases.csv"), index=False)
        mapping_c = overlap.rename(columns={"display_id": "r1_display_id"})
        _save_cxr_mapping(mapping_c, os.path.join(c_dir, "case_mapping.csv"),
                          display_col="r2_display_id",
                          extra_cols=["r1_display_id"])

        d_dir = os.path.join(self.out_root, "task_D_extension")
        d_img = os.path.join(d_dir, "images")
        os.makedirs(d_img, exist_ok=True)
        ext = _sample_cxr_gold(self.manifest, self.N_EXTENSION, RANDOM_SEED + 6,
                               self.CORE_CAP, self.RARE_CAP,
                               exclude_ids=set(self.r1_gold["case_id"].astype(str)))
        ext["display_id"] = [f"D_{i+1:04d}" for i in range(len(ext))]
        for _, r in ext.iterrows():
            _render_cxr(self.site_roots, r, os.path.join(d_img, f"{r['display_id']}.png"))
        reader_d = pd.DataFrame({
            "display_id": ext["display_id"],
            "question":   ext["target_finding"].map(_cxr_question),
            "finding_present": "",
            "image_quality":   "",
            "distrust_automated_read": "",
            "comment": "",
        })
        reader_d.to_csv(os.path.join(d_dir, "cases.csv"), index=False)
        _save_cxr_mapping(ext, os.path.join(d_dir, "case_mapping.csv"))

        e_dir = os.path.join(self.out_root, "task_E_triplet_overlap")
        e_img = os.path.join(e_dir, "images")
        os.makedirs(e_img, exist_ok=True)
        r1_trip_map = read_csv_defensively(
            os.path.join(self.r1_out_root, "task_B_similarity", "case_mapping.csv"))
        ov = r1_trip_map.head(self.N_TRIPLET_OVERLAP).copy()
        # Re-id and reshuffle session order for blinding.
        ov = _assign_triplet_sessions(
            ov.rename(columns={"triplet_id": "r1_triplet_id"}).assign(
                triplet_id=[f"E_{i+1:04d}" for i in range(len(ov))]),
            self.N_SESSIONS, RANDOM_SEED + 7)
        gold_by_id = self.r1_gold.set_index("case_id")
        for _, t in ov.iterrows():
            for slot, cid in [("a", t["case_a"]), ("b", t["case_b"]), ("c", t["case_c"])]:
                r = gold_by_id.loc[cid]
                dst = os.path.join(e_img, f"{t['triplet_id']}_{slot}.png")
                if not os.path.exists(dst):
                    _render_cxr(self.site_roots, r, dst)
        reader_e = ov[["triplet_id", "session"]].copy()
        reader_e["images"] = reader_e["triplet_id"].map(
            lambda t: f"reference={t}_a.png | option_B={t}_b.png | option_C={t}_c.png")
        reader_e["more_similar_to"] = ""
        reader_e["confidence"] = ""
        reader_e["comment"] = ""
        reader_e.to_csv(os.path.join(e_dir, "cases.csv"), index=False)
        ov.to_csv(os.path.join(e_dir, "case_mapping.csv"), index=False)



class Reader3Preparer:
    N_GOLD     = 480
    N_TRIPLETS = 300
    PER_CLASS  = 44       # 11 classes x 44 ~= 480
    N_SESSIONS = 3

    def __init__(self, cfg, out_root: str, manifest: pd.DataFrame,
                 histo_roots: Dict[str, str]):
        self.cfg = cfg
        self.out_root = out_root
        self.manifest = manifest
        self.roots = histo_roots

    def prepare(self):
        _set_seed()
        gold = _sample_histo_gold(self.manifest, self.N_GOLD, RANDOM_SEED + 10,
                                  self.PER_CLASS)

        f_dir = os.path.join(self.out_root, "task_F_gold_labels")
        f_img = os.path.join(f_dir, "images")
        os.makedirs(f_img, exist_ok=True)
        for _, r in gold.iterrows():
            _render_histo(self.roots["histo"], self.roots["quilt"],
                          self.roots["nct"], self.roots["pcam"],
                          r, os.path.join(f_img, f"{r['display_id']}.png"))
        reader = pd.DataFrame({
            "display_id": gold["display_id"],
            "question":   gold["target_class"].map(_histo_question),
            "tissue_present": "",            # Yes / No
            "image_quality":  "",            # adequate / suboptimal / non-diagnostic
            "distrust_automated_read": "",   # Yes / No
            "comment": "",
        })
        reader.to_csv(os.path.join(f_dir, "cases.csv"), index=False)
        gold[["display_id", "case_id", "dataset", "tissue_class", "tumor_label",
              "target_class"]].to_csv(
            os.path.join(f_dir, "case_mapping.csv"), index=False)

        g_dir = os.path.join(self.out_root, "task_G_similarity")
        g_img = os.path.join(g_dir, "images")
        os.makedirs(g_img, exist_ok=True)
        trips = _build_triplets(gold, self.N_TRIPLETS, "T3", RANDOM_SEED + 11)
        trips = _assign_triplet_sessions(trips, self.N_SESSIONS, RANDOM_SEED + 12)
        gold_by_id = gold.set_index("case_id")
        for _, t in trips.iterrows():
            for slot, cid in [("a", t["case_a"]), ("b", t["case_b"]), ("c", t["case_c"])]:
                r = gold_by_id.loc[cid]
                dst = os.path.join(g_img, f"{t['triplet_id']}_{slot}.png")
                if not os.path.exists(dst):
                    _render_histo(self.roots["histo"], self.roots["quilt"],
                                  self.roots["nct"], self.roots["pcam"], r, dst)
        reader_t = trips[["triplet_id", "session"]].copy()
        reader_t["images"] = reader_t["triplet_id"].map(
            lambda t: f"reference={t}_a.png | option_B={t}_b.png | option_C={t}_c.png")
        reader_t["more_similar_to"] = ""
        reader_t["confidence"] = ""
        reader_t["comment"] = ""
        reader_t.to_csv(os.path.join(g_dir, "cases.csv"), index=False)
        trips.to_csv(os.path.join(g_dir, "case_mapping.csv"), index=False)



def _save_cxr_mapping(gold: pd.DataFrame, path: str,
                      display_col: str = "display_id",
                      extra_cols: Optional[List[str]] = None):
    keep = [display_col, "case_id", "dataset", "split", "target_finding",
            "image_key", "image_subdir",
            "sex", "age", "race", "view"] + CXR_CORE_FINDINGS + CXR_RARE_FINDINGS
    if extra_cols:
        keep = extra_cols + keep
    keep = [c for c in keep if c in gold.columns]
    gold[keep].to_csv(path, index=False)



def prepare_all_reader_studies(cfg_path: str, output_root: Optional[str] = None):
    """Build the three clinician folder trees under reader_study/<reader>/.
    Each is a self-contained directory you can zip and hand to one clinician."""
    params = read_config(cfg_path)
    cfg = params["Convergence"]
    if output_root is None:
        output_root = os.path.join(cfg["alignment"]["results_base_dir"],
                                   "reader_study")

    r1_root = os.path.join(output_root, "reader1_radiologist")
    r2_root = os.path.join(output_root, "reader2_radiologist")
    r3_root = os.path.join(output_root, "reader3_pathologist")
    for d in (r1_root, r2_root, r3_root):
        os.makedirs(d, exist_ok=True)

    # CXR per-site image roots (each site lives under its own folder) + manifest
    cxr_sites = cfg["cxr"]["sites"]
    cxr_site_roots = {site: scfg["image_root"]
                      for site, scfg in cxr_sites.items()}
    cxr_manifest = _load_cxr_manifest(cfg["cxr"]["pool_manifest_csv"])

    # Histo roots and manifest (mirror modality_embedding_loaders resolution)
    histo_manifest = read_csv_defensively(cfg["histo"]["pool_manifest_csv"])
    qcfg = cfg.get("quilt", {})
    histo_roots = {
        "quilt": qcfg.get("root", ""),
        "nct":   cfg["histo"]["nct_crc"]["root"],
        "pcam":  cfg["histo"]["pcam"]["h5_dir"],
    }
    histo_roots["histo"] = histo_roots["nct"]  # unused fallback

    # Reader 1 first (its gold set seeds Reader 2's overlap + triplet overlap).
    r1_gold = Reader1Preparer(cfg, cxr_site_roots, r1_root, cxr_manifest).prepare()
    Reader2Preparer(cfg, cxr_site_roots, r2_root, cxr_manifest,
                    r1_gold, r1_root).prepare()
    Reader3Preparer(cfg, r3_root, histo_manifest, histo_roots).prepare()

    for root, who in [(r1_root, "Reader 1 (radiologist, CXR)"),
                      (r2_root, "Reader 2 (radiologist, CXR)"),
                      (r3_root, "Reader 3 (pathologist, histo)")]:
        for d in sorted(os.listdir(root)):
            full = os.path.join(root, d)
            if os.path.isdir(full):
                img = os.path.join(full, "images")
                n = len(os.listdir(img)) if os.path.isdir(img) else 0
                print(f"  {d}: {n} images")

