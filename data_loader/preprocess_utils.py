"""
data_loader/preprocess_utils.py

Shared image-resize helper for the reproduction preprocessing scripts.

The chest-radiograph and fundus images used by this project were resized to 224
and 512 px offline, and those preprocessed trees already exist on the system, so
the per-modality preprocessing scripts do not run here. They are kept for
transparency and reproduction: pointed at a raw image tree, they reproduce the
exact resize convention (LANCZOS to a square target, relative structure
preserved). This module holds the one resize routine they share.

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Sequence, Tuple

from PIL import Image
from tqdm import tqdm

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
    """Resize every image under raw_root to 224 and 512 px, mirroring the
    relative directory structure into out_root_224 and out_root_512. Resumable:
    targets that already exist are skipped, so on a system where the
    preprocessed trees are already present this is a no-op.
    """
    if not raw_root or not os.path.isdir(raw_root):
        print(f"[preprocess{tag}] raw root absent or not set ({raw_root}); "
              f"skipping (preprocessed trees already exist).")
        return

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
        print(f"[preprocess{tag}] nothing to do under {raw_root}.")
        return

    print(f"[preprocess{tag}] resizing {len(jobs)} images under {raw_root}.")
    errors = []
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_resize_one, *j): j[0] for j in jobs}
        for fut in tqdm(as_completed(futs), total=len(futs), unit="img"):
            r = fut.result()
            if r.startswith("ERROR"):
                errors.append(r)
    print(f"[preprocess{tag}] done. errors={len(errors)}")
    for e in errors[:10]:
        print(" ", e)
