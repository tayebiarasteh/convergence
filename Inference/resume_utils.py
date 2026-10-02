"""
Inference/resume_utils.py
Created on August 28, 2026

@author: Soroosh Tayebi Arasteh
https://github.com/tayebiarasteh
"""

import json
import os
import shutil
import socket
import subprocess
import time
import uuid
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


class MissingInput(Exception):
    pass


class IncompatibleArtifact(Exception):
    pass


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(s))


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def ensure_parent(path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return parent


def _atomic_write(path: str, write: Callable[[str], None], suffix: str = "") -> None:
    ensure_parent(path)
    tmp = f"{path}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp{suffix}"
    try:
        write(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _fsync_file(path: str) -> None:
    with open(path, "rb+") as f:
        f.flush()
        os.fsync(f.fileno())


def write_json_atomic(path: str, obj) -> None:
    def _write(tmp):
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=2, default=_json_default, ensure_ascii=True)
            f.flush()
            os.fsync(f.fileno())
    _atomic_write(path, _write)


def write_csv_atomic(df: pd.DataFrame, path: str, index: bool = False) -> None:
    def _write(tmp):
        df.to_csv(tmp, index=index)
        _fsync_file(tmp)
    _atomic_write(path, _write)


def write_npz_atomic(path: str, **arrays) -> None:
    _atomic_write(path, lambda tmp: np.savez(tmp, **arrays), suffix=".npz")


def write_npy_atomic(path: str, arr) -> None:
    _atomic_write(path, lambda tmp: np.save(tmp, arr), suffix=".npy")


def write_torch_atomic(obj, path: str) -> None:
    import torch
    _atomic_write(path, lambda tmp: torch.save(obj, tmp))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fingerprint_file(path: str) -> Optional[Dict]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return {"path": os.path.abspath(path), "size": int(st.st_size),
            "mtime": round(float(st.st_mtime), 3)}


def digest_file(path: str) -> Optional[str]:
    import hashlib
    try:
        with open(path, "rb") as f:
            return hashlib.blake2b(f.read(), digest_size=16).hexdigest()
    except OSError:
        return None


def fingerprint_done_flag(done_flag_path: str) -> Optional[str]:
    try:
        with open(done_flag_path) as f:
            return f.read().strip()
    except OSError:
        return None


def params_path(artifact_path: str) -> str:
    return artifact_path + ".params.json"


def write_build_params(artifact_path: str, params: Dict) -> None:
    write_json_atomic(params_path(artifact_path), params)


def read_build_params(artifact_path: str) -> Optional[Dict]:
    p = params_path(artifact_path)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _moved_keys(stored: Dict, expected: Dict, prefix: str = "") -> List:
    moved = []
    for k, want in expected.items():
        if k not in stored:
            continue
        was = stored[k]
        name = f"{prefix}{k}"
        if isinstance(was, dict) and isinstance(want, dict):
            moved.extend(_moved_keys(was, want, prefix=f"{name}."))
        elif was != want:
            moved.append((name, was, want))
    return moved


def output_is_current(artifact_path: str, sources: Sequence[str], owner: str,
                      extra: Optional[Dict] = None) -> bool:
    if not os.path.exists(artifact_path):
        return False
    stored = read_build_params(artifact_path)
    if not stored or "sources" not in stored:
        return False
    for k, want in (extra or {}).items():
        if stored.get(k) != want:
            return False
    for p, fp in stored["sources"].items():
        if fingerprint_file(p) != fp:
            return False
    for p in sources:
        if p not in stored["sources"]:
            return False
    return True


def record_sources(artifact_path: str, sources: Sequence[str],
                   extra: Optional[Dict] = None) -> None:
    rec = {"sources": {p: fingerprint_file(p) for p in sources}}
    if extra:
        rec.update(extra)
    write_build_params(artifact_path, rec)


def check_build_params(artifact_path: str, expected: Dict, owner: str,
                       raise_on_mismatch: bool = False,
                       legacy: Optional[Dict] = None) -> bool:
    stored = read_build_params(artifact_path)
    if stored is None:
        return True
    if legacy:
        for k, was in legacy.items():
            if k not in stored:
                stored = dict(stored)
                stored[k] = was
    moved = _moved_keys(stored, expected)
    if not moved:
        return True
    lines = "\n  ".join(f"{k}: {was!r} -> {now!r}" for k, was, now in moved)
    msg = (f"[{owner}] REBUILD: {os.path.basename(artifact_path)} was built "
           f"under different parameters, so it does not answer the question "
           f"being asked now.\n  {lines}")
    if raise_on_mismatch:
        raise IncompatibleArtifact(msg)
    return False


_JSONL_WARNED = set()


def read_jsonl(path: str, owner: str = "resume_utils") -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows
    bad = 0
    with open(path, "rb") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad += 1
    if bad and path not in _JSONL_WARNED:
        _JSONL_WARNED.add(path)
    return rows


def done_keys(path: str, key_fields: Sequence[str],
              owner: str = "resume_utils") -> set:
    return {tuple(r.get(k) for k in key_fields) for r in read_jsonl(path, owner)}


def unit_cache_path(partial_dir: str, group: str, unit: str) -> str:
    return os.path.join(partial_dir, f"{_safe(group)}__{_safe(unit)}.jsonl")


def unit_done(partial_dir: str, group: str, unit: str,
              require_rows: bool = False) -> bool:
    path = unit_cache_path(partial_dir, group, unit)
    if not os.path.exists(path):
        return False
    if not require_rows:
        return True
    return len(read_jsonl(path)) > 0


def write_unit_rows(partial_dir: str, group: str, unit: str,
                    rows: List[Dict]) -> None:
    ensure_dir(partial_dir)

    def _write(tmp):
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=True, default=_json_default) + "\n")
            f.flush()
            os.fsync(f.fileno())
    _atomic_write(unit_cache_path(partial_dir, group, unit), _write)


def load_group_rows(partial_dir: str, group: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isdir(partial_dir):
        return rows
    prefix = f"{_safe(group)}__"
    for name in sorted(os.listdir(partial_dir)):
        if not name.startswith(prefix) or not name.endswith(".jsonl"):
            continue
        rows.extend(read_jsonl(os.path.join(partial_dir, name)))
    return rows


def run_units_resumable(
    partial_dir: str,
    group: str,
    units: Sequence[str],
    compute_unit: Callable[[str], List[Dict]],
    progress_desc: Optional[str] = None,
    require_rows: bool = False,
    use_claims: bool = False,
    status_file: Optional[str] = None,
    build_params: Optional[Dict] = None,
) -> pd.DataFrame:
    ensure_dir(partial_dir)
    if build_params is not None:
        stamp = os.path.join(partial_dir, f"{group}__build_params.json")
        if not check_build_params(stamp, build_params, owner=group):
            if use_claims and not claim_unit(partial_dir, f"{group}__reset"):
                raise MissingInput(f"[{group}] another job is clearing the partials that were "
                                   f"built under other parameters.")
            try:
                if not check_build_params(stamp, build_params, owner=group):
                    clear_partial_dir(partial_dir)
                    ensure_dir(partial_dir)
            finally:
                if use_claims:
                    release_claim(partial_dir, f"{group}__reset")
        write_build_params(stamp, build_params)
    todo = [u for u in units if not unit_done(partial_dir, group, u, require_rows)]
    n_skipped = 0
    mine = set()

    def _run(candidates, desc):
        nonlocal n_skipped
        try:
            from tqdm import tqdm
            iterator = tqdm(candidates, desc=desc, unit="unit")
        except ImportError:
            iterator = candidates
        for unit in iterator:
            if unit_done(partial_dir, group, unit, require_rows):
                continue
            tag = f"{group}__{unit}"
            if use_claims and not claim_unit(partial_dir, tag):
                continue
            try:
                try:
                    rows = compute_unit(unit)
                except MissingInput as e:
                    n_skipped += 1
                    mine.add(unit)
                    continue
                write_unit_rows(partial_dir, group, unit, rows)
                mine.add(unit)
            finally:
                if use_claims:
                    release_claim(partial_dir, tag)
            if status_file:
                append_status(status_file, f"{group}: {unit} wrote {len(rows)} row(s)")

    _run(todo, progress_desc or f"[{group}]")
    left = [u for u in units if u not in mine and not unit_done(partial_dir, group, u, require_rows)
            and not (use_claims and claim_is_live(partial_dir, f"{group}__{u}"))]
    if left:
        _run(left, f"[{group}] released units")
    elsewhere = [u for u in units if u not in mine
                 and not unit_done(partial_dir, group, u, require_rows)]
    if elsewhere:
        raise MissingInput(f"[{group}] {len(elsewhere)} of {len(units)} unit(s) are still with "
                           f"another job (first: {elsewhere[0]}), and the job that finishes the "
                           f"last one writes the table.")
    out = pd.DataFrame(load_group_rows(partial_dir, group))
    out.attrs["n_skipped"] = n_skipped
    out.attrs["n_units"] = len(units)
    return out


def clear_partial_dir(partial_dir: str) -> None:
    shutil.rmtree(partial_dir, ignore_errors=True)


def done_flag_path(cell_dir: str) -> str:
    return os.path.join(cell_dir, "done.flag")


def cell_is_done(cell_dir: str, required_files: Iterable[str] = ()) -> bool:
    flag = done_flag_path(cell_dir)
    if not os.path.exists(flag):
        return False
    for name in required_files:
        p = os.path.join(cell_dir, name)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            return False
    return True


def write_done_flag(cell_dir: str, meta: Optional[Dict] = None) -> None:
    ensure_dir(cell_dir)
    payload = {"finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "host": socket.gethostname()}
    if meta:
        payload.update(meta)
    write_json_atomic(done_flag_path(cell_dir), payload)


def read_done_meta(cell_dir: str) -> Dict:
    try:
        with open(done_flag_path(cell_dir)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def clear_cell(cell_dir: str) -> None:
    for name in ("done.flag", "resume.pt", "final.pt"):
        p = os.path.join(cell_dir, name)
        if os.path.exists(p):
            os.remove(p)
    for name in os.listdir(cell_dir) if os.path.isdir(cell_dir) else []:
        if name.endswith(".params.json"):
            os.remove(os.path.join(cell_dir, name))


def append_status(status_file: str, message: str) -> None:
    ensure_parent(status_file)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {message}\n"
    with open(status_file, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def status_path(cfg: Dict, stage: str) -> str:
    return os.path.join(cfg["outputs"]["status_dir"], f"{stage}_status.txt")


_CLAIM_STALE_AFTER_S = 21600.0


def _claim_path(partial_dir: str, tag: str) -> str:
    return os.path.join(partial_dir, f"claim__{_safe(tag)}.json")


def _slurm_job_state(job_id: str) -> str:
    if not job_id:
        return "unknown"
    try:
        out = subprocess.run(["squeue", "-h", "-j", str(job_id)],
                             capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "gone" if "invalid job id" in (out.stderr or "").lower() else "unknown"
    return "gone" if out.stdout.strip() == "" else "alive"


_LIVE_CLAIMS: set = set()
_CLAIM_TOKENS: Dict[str, str] = {}


def _read_claim(path: str) -> Dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _remove_own_claim(path: str) -> None:
    token = _CLAIM_TOKENS.pop(path, None)
    _LIVE_CLAIMS.discard(path)
    if not os.path.exists(path):
        return
    if token is not None and _read_claim(path).get("token") not in (None, token):
        return
    try:
        os.remove(path)
    except OSError:
        pass


def release_all_live_claims(reason: str = "") -> int:
    n = 0
    for path in sorted(_LIVE_CLAIMS):
        existed = os.path.exists(path)
        _remove_own_claim(path)
        n += int(existed and not os.path.exists(path))
    _LIVE_CLAIMS.clear()
    return n


_LINK_WARNED = False


def _create_claim(path: str, token: str) -> bool:
    global _LINK_WARNED
    tmp = f"{path}.{token}.tmp"
    with open(tmp, "w") as f:
        json.dump({"host": socket.gethostname(), "pid": os.getpid(),
                   "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                   "claimed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "token": token},
                  f, ensure_ascii=True)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    except OSError as exc:
        if not _LINK_WARNED:
            _LINK_WARNED = True
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f, open(tmp) as src:
            f.write(src.read())
        return True
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def claim_unit(partial_dir: str, tag: str,
               stale_after_s: float = _CLAIM_STALE_AFTER_S) -> bool:
    ensure_dir(partial_dir)
    path = _claim_path(partial_dir, tag)
    token = uuid.uuid4().hex
    if not _create_claim(path, token):
        claim = _read_claim(path)
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return False
        owner = _slurm_job_state(str(claim.get("slurm_job_id", "")))
        if owner == "alive" or (owner == "unknown" and age < stale_after_s):
            return False
        aside = f"{path}.{token}.expired"
        try:
            os.rename(path, aside)
        except OSError:
            return False
        moved = _read_claim(aside)
        if moved != claim:
            try:
                os.link(aside, path)
            except OSError:
                pass
            os.remove(aside)
            return False
        os.remove(aside)
        if not _create_claim(path, token):
            return False
    _LIVE_CLAIMS.add(path)
    _CLAIM_TOKENS[path] = token
    return True


def claim_is_live(partial_dir: str, tag: str,
                  stale_after_s: float = _CLAIM_STALE_AFTER_S) -> bool:
    path = _claim_path(partial_dir, tag)
    if not os.path.exists(path) or path in _LIVE_CLAIMS:
        return False
    try:
        with open(path) as f:
            claim = json.load(f)
    except (OSError, json.JSONDecodeError):
        claim = {}
    try:
        age = time.time() - os.path.getmtime(path)
    except OSError:
        return False
    owner = _slurm_job_state(str(claim.get("slurm_job_id", "")))
    return owner == "alive" or (owner == "unknown" and age < stale_after_s)


def claim_cell(cell_dir: str, tag: str, owner: str) -> bool:
    if claim_unit(cell_dir, tag):
        return True
    return False


def heartbeat_claim(partial_dir: str, tag: str) -> None:
    path = _claim_path(partial_dir, tag)
    if os.path.exists(path):
        os.utime(path, None)


def release_claim(partial_dir: str, tag: str) -> None:
    _remove_own_claim(_claim_path(partial_dir, tag))


def _fmt_size_gib(x: float) -> str:
    return f"{x * 1024:.0f} MiB" if x < 1.0 else f"{x:.1f} GiB"


def print_projected_peak(owner: str, parts: Dict[str, float], note: str = "") -> float:
    total = float(sum(parts.values()))
    detail = ", ".join(f"{k} {_fmt_size_gib(v)}" for k, v in sorted(parts.items(), key=lambda kv: -kv[1]))
    ask = (f"Request at least {total * 1.3:.0f} GiB." if total >= 1.0
           else "The loaded model and not this block decides the job's memory.")
    return total
