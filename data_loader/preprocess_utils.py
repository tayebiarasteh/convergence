"""
data_loader/preprocess_utils.py
Created on June 13, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm
from Inference.resume_utils import MissingInput
from Inference.resume_utils import check_build_params, write_build_params

DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")


def _resize_one(src: str, dst_224: str, dst_512: str) -> str:
    try:
        img = Image.open(src).convert("RGB")
        for res, dst in ((224, dst_224), (512, dst_512)):
            if dst and not os.path.exists(dst):
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                img.resize((res, res), Image.LANCZOS).save(dst)
        return "ok"
    except (OSError, ValueError) as e:
        return f"ERROR {src}: {e}"


def resize_tree(
    raw_root: str,
    out_root_224: str,
    out_root_512: str,
    exts: Sequence[str] = DEFAULT_EXTS,
    num_workers: int = 8,
    tag: str = "",
) -> None:
    if not raw_root or not os.path.isdir(raw_root):
        raise MissingInput(
            f"[preprocess{tag}] the raw image root is absent or unset ({raw_root}). On a machine "
            f"where the preprocessed trees already exist this stage has nothing to do; "
            f"everywhere else the source tree has to be present.")

    stamp = os.path.join(out_root_224, ".preprocess_build_params.json")
    params = {"size_224": 224, "size_512": 512, "exts": sorted(exts)}
    if not check_build_params(stamp, params, owner=f"preprocess{tag}"):
        pass

    jobs: List[Tuple[str, str, str]] = []
    for dirpath, _, files in os.walk(raw_root):
        for fn in files:
            if not fn.lower().endswith(tuple(exts)):
                continue
            src = os.path.join(dirpath, fn)
            rel = os.path.relpath(src, raw_root)
            stem = os.path.splitext(rel)[0] + ".png"
            d224 = os.path.join(out_root_224, stem)
            d512 = os.path.join(out_root_512, stem)
            if os.path.exists(d224) and os.path.exists(d512):
                continue
            jobs.append((src, d224, d512))

    if not jobs:
        return

    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_resize_one, *j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
            r = fut.result()
            if r.startswith("ERROR"):
                errors.append(r)
    os.makedirs(out_root_224, exist_ok=True)
    write_build_params(stamp, params)
