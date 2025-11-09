import os
import io
import zipfile
import subprocess
import shutil
import hashlib
import fnmatch
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor

DEFAULT_REMOTE_ROOT = os.environ.get("REMOTE_ROOT", "/workspace")
DEFAULT_HF_HOME = os.environ.get("HF_HOME", f"{DEFAULT_REMOTE_ROOT}/hf")

def _iter_files(base_path: str, exclude_files: Optional[List[str]] = None, exclude_dirs: Optional[List[str]] = None) -> Iterable[Tuple[str, str]]:
    base_path = os.path.abspath(base_path)
    if os.path.isfile(base_path):
        rel = os.path.basename(base_path)
        if not _excluded(rel, exclude_files):
            yield rel.replace("\\", "/"), base_path
        return
    for root, dirs, files in os.walk(base_path):
        if exclude_dirs and dirs:
            keep: List[str] = []
            for d in dirs:
                rel_dir = os.path.relpath(os.path.join(root, d), base_path).replace("\\", "/")
                if any(fnmatch.fnmatch(rel_dir, pat) or fnmatch.fnmatch(rel_dir + "/", pat) for pat in exclude_dirs):
                    continue
                keep.append(d)
            dirs[:] = keep
        for name in files:
            abs_path = os.path.join(root, name)
            rel_path = os.path.relpath(abs_path, base_path).replace("\\", "/")
            if _excluded(rel_path, exclude_files):
                continue
            yield rel_path, abs_path

def _excluded(rel_path: str, exclude: Optional[List[str]]) -> bool:
    if not exclude:
        return False
    for pat in exclude:
        if fnmatch.fnmatch(rel_path, pat):
            return True
    return False

def _hash_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()

def compute_hashes(path: str, exclude_files: Optional[List[str]] = None, exclude_dirs: Optional[List[str]] = None) -> Dict[str, str]:
    pairs = list(_iter_files(path, exclude_files, exclude_dirs))
    results: Dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        futs = {ex.submit(_hash_file, ap): rp for rp, ap in pairs}
        for fut in futs:
            results[futs[fut]] = fut.result()
    return results

def get_remote_hashes(path: str) -> Dict[str, str]:
    # Backend-specific: implement fetching hashes from remote storage
    # For now, placeholder assuming local (override in backend)
    if not os.path.exists(path):
        return {}
    if os.path.isfile(path):
        return {os.path.basename(path): _hash_file(path)}
    out: Dict[str, str] = {}
    for root, _, files in os.walk(path):
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, path).replace("\\", "/")
            out[rel] = _hash_file(p)
    return out

def zip_dir(path: str) -> Optional[bytes]:
    # Backend-agnostic zipping of dir
    if not os.path.exists(path):
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(path):
            for name in files:
                p = os.path.join(root, name)
                arc = os.path.relpath(p, path)
                zf.write(p, arc)
    buf.seek(0)
    return buf.read()

def run_script(script_path: str, venv: Optional[str] = None, workdir: str = DEFAULT_REMOTE_ROOT, gpu: Optional[str] = None, timeout: Optional[int] = 86400) -> str:
    # Local run; backend handles remote
    os.chdir(workdir)
    if venv:
        act = venv if venv.endswith("/bin/activate") else os.path.join(venv, "bin/activate")
        cmd = f"source {act} && python {script_path}"
    else:
        cmd = f"python {script_path}"
    subprocess.run(cmd, shell=True, check=True, executable="/bin/bash", timeout=timeout)
    return workdir

def prefetch_hf(repos: List, hf_home: str = DEFAULT_HF_HOME) -> str:
    # Local prefetch; backend can run remotely
    from pathlib import Path as _P
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise ImportError("huggingface_hub not installed; install with pip install huggingface_hub")
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", hf_home)
    _P(hf_home).mkdir(parents=True, exist_ok=True)
    for item in repos:
        if isinstance(item, (list, tuple)) and len(item) >= 1:
            repo_id = item[0]
            alias = item[1] if len(item) > 1 else None
        else:
            repo_id = str(item)
            alias = None
        name = alias or repo_id.replace("/", "--")
        dest = os.path.join(hf_home, name)
        _P(dest).mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=repo_id, local_dir=dest, local_dir_use_symlinks=False)
    return hf_home

def sync_workspace(paths: List[str], exclude_files_global: Optional[List[str]] = None, exclude_dirs_global: Optional[List[str]] = None, exclude_files_map: Optional[Dict[str, List[str]]] = None, exclude_dirs_map: Optional[Dict[str, List[str]]] = None, remote_root: str = DEFAULT_REMOTE_ROOT, upload_func=None, allowed_extensions: Optional[Sequence[str]] = None, get_remote_hashes_func=None) -> None:
    # Backend provides upload_func for remote upload
    allowed_set = None
    if allowed_extensions:
        allowed_set = {ext.lower().lstrip(".") for ext in allowed_extensions}
    for src in paths:
        src_abs = os.path.abspath(src)
        if not os.path.exists(src_abs):
            continue
        is_dir = os.path.isdir(src_abs)
        base = os.path.basename(src_abs.rstrip(os.sep)) if is_dir else os.path.basename(src_abs)
        # Per-src excludes
        per_src_files_exclude = exclude_files_global or []
        if exclude_files_map:
            per_src_files_exclude += exclude_files_map.get(src, []) or exclude_files_map.get(src_abs, []) or exclude_files_map.get(base, [])
        per_src_dirs_exclude = exclude_dirs_global or []
        if exclude_dirs_map:
            per_src_dirs_exclude += exclude_dirs_map.get(src, []) or exclude_dirs_map.get(src_abs, []) or exclude_dirs_map.get(base, [])
        local_hashes = compute_hashes(src_abs, exclude_files=per_src_files_exclude, exclude_dirs=per_src_dirs_exclude)
        remote_target = os.path.join(remote_root, base) if is_dir else os.path.join(remote_root, base)
        remote_hashes_func = get_remote_hashes_func or get_remote_hashes
        remote_hashes = remote_hashes_func(remote_target)
        files_to_upload: List[Tuple[str, str]] = []
        for rel, h in local_hashes.items():
            if allowed_set is not None:
                ext = Path(rel).suffix.lower().lstrip(".")
                if ext not in allowed_set:
                    continue
            if rel not in remote_hashes or remote_hashes[rel] != h:
                local_file = os.path.join(src_abs, rel) if is_dir else src_abs
                remote_file = os.path.join(base, rel) if is_dir else base
                files_to_upload.append((local_file, remote_file.replace("\\", "/")))
        for local_file, remote_file in files_to_upload:
            if upload_func:
                upload_func(local_file, remote_file)
            else:
                # Local copy
                target_path = os.path.join(remote_root, remote_file)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(local_file, target_path)

def sync_outputs(local_dir: str, remote_dir: str = f"{DEFAULT_REMOTE_ROOT}/out_local", allowed_extensions: Optional[Sequence[str]] = None) -> None:
    local_path = Path(local_dir).expanduser().resolve()
    remote_hashes = get_remote_hashes(remote_dir)
    if not remote_hashes:
        if local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        return
    local_hashes = compute_hashes(str(local_path)) if local_path.exists() else {}
    if local_hashes == remote_hashes:
        return
    zip_bytes = zip_dir(remote_dir)
    if local_path.exists():
        shutil.rmtree(local_path)
    local_path.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        zf.extractall(local_path)
    if allowed_extensions:
        allowed = {ext.lower().lstrip(".") for ext in allowed_extensions}
        for file_path in local_path.rglob("*"):
            if file_path.is_file():
                ext = file_path.suffix.lower().lstrip(".")
                if ext not in allowed:
                    file_path.unlink()
        for dir_path in sorted(local_path.rglob("*"), reverse=True):
            if dir_path.is_dir():
                try:
                    next(dir_path.iterdir())
                except StopIteration:
                    dir_path.rmdir()
