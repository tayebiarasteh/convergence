"""
controlled/matrix.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

from typing import Dict, List

from config.serde import read_config

OBJECTIVES = ("mae", "dino", "simclr", "supervised", "image_text")
TWO_VIEW_OBJECTIVES = ("simclr", "dino")

LABEL_FORMS = ("presence", "laterality", "severity")
GRADE_CLASSES = 5

CXR_LIKE = ("cxr", "cxrh1", "cxrh2")

ARM_SOURCES = {
    "cxr":   ("cxr", "cxr"),
    "cxrh1": ("cxr", "cxr"),
    "cxrh2": ("cxr", "cxr"),
    "taix":  ("cxr", "taix"),
    "histo": ("histo", "histo"),
}


def arm_sources(arm: str) -> tuple:
    key = arm.split("__")[0]
    if key.startswith("frame_"):
        key = key[len("frame_"):]
    src = ARM_SOURCES.get(key)
    if src is None:
        raise ValueError(f"arm '{key}' declares no image source; add it to ARM_SOURCES beside "
                         f"{', '.join(sorted(ARM_SOURCES))}. A reader cannot guess which loader "
                         f"reads its images or which pool its cells are scored on.")
    return src


def pairing_group(run_id: str) -> str:
    mod = run_id.split("__")[0]
    return "cxr_disjoint" if mod in ("cxrh1", "cxrh2") else mod


def taix_label_columns(form: str, taix_cfg: dict) -> List[str]:
    if form == "presence":
        return list(taix_cfg["binary_label_map"].values())
    if form == "laterality":
        return [f"side__{f}__{side}" for f in taix_cfg["side_resolved"]
                for side in ("right", "left")]
    if form == "severity":
        return [f"grade__{g}" for g in taix_cfg["graded_columns"]]
    raise ValueError(f"unknown label form {form!r}; expected one of {', '.join(LABEL_FORMS)}")


def enumerate_e5_runs(global_config_path: str) -> List[tuple]:
    cfg  = read_config(global_config_path)["Convergence"]
    ctrl = cfg["controlled"]
    objectives = ctrl.get("objectives", list(OBJECTIVES))
    backbones  = ctrl.get("backbones",  ["vit_s", "vit_b"])
    inits      = ctrl.get("inits",      ["dinov3", "random"])
    modalities = ctrl.get("modalities", ["cxr", "histo"])
    n_seeds    = int(ctrl.get("n_seeds", 3))
    unknown = [o for o in objectives if o not in OBJECTIVES]
    if unknown:
        raise ValueError(f"[E5] config declares objectives with no training branch: {unknown}. "
                         f"Implemented: {', '.join(OBJECTIVES)}.")
    exclusions = ctrl.get("modality_objective_exclusions", {}) or {}
    rnd = ctrl.get("random_arm", {}) or {}
    rnd_backbones = list(rnd.get("backbones", backbones))
    rnd_seeds = int(rnd.get("n_seeds", n_seeds))
    runs = []
    for mod in modalities:
        excluded = set(exclusions.get(mod, []))
        for obj in objectives:
            if obj in excluded:
                continue
            for init in inits:
                bbs = rnd_backbones if init == "random" else backbones
                sds = rnd_seeds if init == "random" else n_seeds
                for bb in bbs:
                    for seed in range(sds):
                        runs.append((f"{mod}__{obj}__{bb}__{init}__seed{seed}",
                                     mod, obj, bb, init, seed))
    return runs


def frame_run_id(frame_id: str, objective_map: dict) -> str:
    parts = frame_id.split("__")
    if len(parts) != 6:
        raise ValueError(f"[frame] '{frame_id}' is not a FRAME run id "
                         f"(expected modality__objective__backbone__init__composition__seedN).")
    mod, obj, backbone, _init, comp, seed = parts
    mapped = objective_map.get(obj)
    if mapped is None:
        raise ValueError(f"[frame] no objective_map entry for FRAME objective '{obj}'.")
    if mapped not in OBJECTIVES:
        raise ValueError(f"[frame] objective_map sends '{obj}' to '{mapped}', which has no "
                         f"training branch here. Implemented: {', '.join(OBJECTIVES)}.")
    level = mapped if comp == "natural" else f"{mapped}_{comp}"
    return f"frame_{mod}__{level}__{backbone}__dinov3__{seed}"


def enumerate_frame_runs(global_config_path: str) -> List[tuple]:
    cfg = read_config(global_config_path)["Convergence"]
    fr = cfg.get("frame", {}) or {}
    if not fr.get("enabled", False):
        return []
    omap = dict(fr.get("objective_map", {}))
    out = []
    for fid in fr.get("runs", []):
        rid = frame_run_id(fid, omap)
        parts = rid.split("__")
        seed = int(parts[4].replace("seed", ""))
        out.append((rid, fid, parts[0], parts[1], parts[2], seed))
    return out


def enumerate_disjoint_runs(global_config_path: str) -> List[tuple]:
    cfg = read_config(global_config_path)["Convergence"]
    dp = (cfg.get("controlled", {}) or {}).get("disjoint_pair", {}) or {}
    if not dp.get("enabled", False):
        return []
    obj = str(dp.get("objective", "simclr"))
    if obj not in OBJECTIVES:
        raise ValueError(f"[disjoint] objective '{obj}' has no training branch.")
    bb = str(dp.get("backbone", "vit_s"))
    n_seeds = int(dp.get("n_seeds", 2))
    runs = []
    for half in ("cxrh1", "cxrh2"):
        for seed in range(n_seeds):
            runs.append((f"{half}__{obj}__{bb}__dinov3__seed{seed}",
                         half, obj, bb, "dinov3", seed))
    return runs


def enumerate_e11_runs(global_config_path: str) -> List[tuple]:
    cfg = read_config(global_config_path)["Convergence"]
    gr = cfg.get("granularity", {}) or {}
    if not gr.get("enabled", True):
        return []
    forms     = list(gr.get("forms", LABEL_FORMS))
    backbones = list(gr.get("backbones", ["vit_s", "vit_b"]))
    n_seeds   = int(gr.get("n_seeds", 3))
    init      = str(gr.get("init", "dinov3"))
    unknown = [f for f in forms if f not in LABEL_FORMS]
    if unknown:
        raise ValueError(f"[E11] config declares label forms with no branch: {unknown}. "
                         f"Implemented: {', '.join(LABEL_FORMS)}.")
    runs = []
    for form in forms:
        for bb in backbones:
            for seed in range(n_seeds):
                runs.append((f"taix__{form}__{bb}__{init}__seed{seed}", form, bb, seed))
    return runs


def e5_family_status(global_config_path: str, align_csv: str, util_csv: str) -> Dict:
    import os
    from Inference.resume_utils import cell_is_done
    from data_loader.build_utils import read_csv_defensively
    cfg = read_config(global_config_path)["Convergence"]
    ckpts = cfg["controlled"]["ckpts_dir"]
    matrix = ([r[0] for r in enumerate_e5_runs(global_config_path)]
              + [r[0] for r in enumerate_e11_runs(global_config_path)]
              + [r[0] for r in enumerate_frame_runs(global_config_path)]
              + [r[0] for r in enumerate_disjoint_runs(global_config_path)])
    trained = [r for r in matrix if cell_is_done(os.path.join(ckpts, r))]
    want_pairs = {frozenset((a, b)) for i, a in enumerate(trained) for b in trained[i + 1:]
                  if pairing_group(a) == pairing_group(b)}
    have_pairs, have_cells = set(), set()
    if os.path.exists(align_csv):
        al = read_csv_defensively(align_csv)
        if {"run_id_a", "run_id_b"}.issubset(al.columns):
            have_pairs = {frozenset((str(a), str(b)))
                          for a, b in zip(al["run_id_a"], al["run_id_b"])}
    if os.path.exists(util_csv):
        ut = read_csv_defensively(util_csv)
        if "run_id" in ut.columns:
            have_cells = {str(x) for x in ut["run_id"]}
    missing_pairs = want_pairs - have_pairs
    want_cells = {str(x) for pr in want_pairs for x in pr}
    return {"trained": len(trained), "want_pairs": len(want_pairs),
            "missing_pairs": len(missing_pairs),
            "missing_cells": sorted(want_cells - have_cells),
            "example": sorted("|".join(sorted(pr)) for pr in missing_pairs)[:3]}


def require_e5_family_complete(global_config_path: str, align_csv: str, util_csv: str,
                               owner: str) -> None:
    import os
    from Inference.resume_utils import MissingInput
    for path, what in ((align_csv, "the alignment table"), (util_csv, "the cell utility table")):
        if not os.path.exists(path):
            raise MissingInput(f"[{owner}] {what} is absent at {path}; main_converge_eval has not consolidated "
                               f"yet. With several jobs, wait until one of them does.")
    from Inference.regimes import stale_reason
    for path in (align_csv, util_csv):
        why = stale_reason(path)
        if why:
            raise MissingInput(f"[{owner}] {why}; main_converge_eval rewrites it, and this stage waits.")
    st = e5_family_status(global_config_path, align_csv, util_csv)
    if not st["trained"]:
        raise MissingInput(f"[{owner}] no controlled cell carries a done flag, so main_converge_eval has "
                           f"produced nothing.")
    if st["missing_pairs"]:
        raise MissingInput(
            f"[{owner}] the alignment table covers {st['want_pairs'] - st['missing_pairs']} of "
            f"{st['want_pairs']} pairs over the {st['trained']} trained cells, so main_converge_eval is still "
            f"running or was interrupted (first missing: {st['example']}). A result computed "
            f"over a fraction of the matrix is the wrong answer, so this stage waits.")
    if st["missing_cells"]:
        raise MissingInput(
            f"[{owner}] {len(st['missing_cells'])} cell(s) named by the alignment table have no "
            f"utility row (first: {st['missing_cells'][:3]}), so main_converge_eval's utility units have not "
            f"all run. This stage waits.")
