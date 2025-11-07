import atexit
import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import modal
from modal import Image, Volume, gpu as modal_gpu

from mlrunner.utils import *

CONFIG_FILENAME = "modal_config.txt"


def _deep_merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = deepcopy(base)
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


class ModalBackend:
    def __init__(self, config: Dict[str, Any], app: Optional[modal.App] = None):
        self.config = self._load_config(config or {})
        app_name = self.config.get("app", {}).get("name") or "mlrunner"
        self.app = app or modal.App(app_name)
        volume_name = self.config.get("volume", {}).get("name") or "workspace"
        self.volume = Volume.from_name(volume_name, create_if_missing=True)
        self.image = self._build_image()
        self.gpu_info = self._resolve_gpu()
        self.remote_outputs = self.config.get("storage", {}).get(
            "outputs_remote_dir", f"{DEFAULT_REMOTE_ROOT}/out_local"
        )
        # Set globals before binding
        global _image, _vol, _gpu
        _image = self.image
        _vol = self.volume
        _gpu = self.gpu_info["decorator"]
        app_proxy.bind(self.app)
        # Start the app context so deferred functions hydrate correctly.
        self._app_ctx = self.app.run()
        self._app_ctx.__enter__()
        atexit.register(self._cleanup_app)

    def _load_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        overrides = deepcopy(overrides)
        config_hint = (
            overrides.pop("modal_config_path", None)
            or overrides.pop("config_path", None)
            or os.environ.get("MLRUNNER_MODAL_CONFIG")
        )
        config_path = Path(config_hint) if config_hint else Path(__file__).parent / CONFIG_FILENAME
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_path
        defaults: Dict[str, Any] = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as fh:
                defaults = json.load(fh)
        return _deep_merge(defaults, overrides)

    def _build_image(self) -> Image:
        image_config = self.config.get("image", {})
        base = image_config.get("base", "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel")
        image = Image.from_registry(base)

        env_vars = image_config.get("env")
        if env_vars:
            image = image.env(env_vars)

        for cmd_group in image_config.get("run_commands", []):
            if isinstance(cmd_group, (list, tuple)):
                image = image.run_commands(*cmd_group)
            else:
                image = image.run_commands(cmd_group)

        apt_pkgs = image_config.get("apt_packages") or image_config.get("system_libs") or []
        if apt_pkgs:
            image = image.apt_install(*apt_pkgs)

        pip_pkgs = image_config.get("pip_packages", [])
        if pip_pkgs:
            image = image.pip_install(*pip_pkgs)

        uv_pkgs = image_config.get("uv_pip_packages", [])
        if uv_pkgs:
            image = image.run_commands(f"uv pip install --system {' '.join(uv_pkgs)}")

        requirements = image_config.get("requirements", {})
        extra_pip = requirements.get("pip", [])
        if extra_pip:
            image = image.run_commands(f"uv pip install --system {' '.join(extra_pip)}")

        conda_cfg = image_config.get("conda_env")
        if conda_cfg:
            installer_url = (
                conda_cfg.get("installer_url")
                if isinstance(conda_cfg, dict)
                else "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
            )
            image = image.run_commands(
                f"wget {installer_url} -O /tmp/miniconda.sh",
                "bash /tmp/miniconda.sh -b -p /opt/conda",
                "export PATH=/opt/conda/bin:$PATH",
            )
            conda_pkgs = []
            if isinstance(conda_cfg, dict):
                conda_pkgs = conda_cfg.get("packages", [])
            else:
                conda_pkgs = requirements.get("conda", [])
            for pkg in conda_pkgs:
                image = image.run_commands(f"conda install -y {pkg}")

        for cmd_group in image_config.get("post_run_commands", []):
            if isinstance(cmd_group, (list, tuple)):
                image = image.run_commands(*cmd_group)
            else:
                image = image.run_commands(cmd_group)

        return image

    def _resolve_gpu(self) -> Dict[str, Any]:
        gpu_config = self.config.get("gpu", {})
        requested = (
            gpu_config.get("type")
            or os.environ.get("MODAL_GPU")
            or gpu_config.get("default_type")
            or "L4"
        )
        requested = str(requested).upper()
        alias_map = {k.upper(): v for k, v in gpu_config.get("type_aliases", {}).items()}
        resolved_label = alias_map.get(requested, requested)
        spec_map = {k.upper(): v for k, v in gpu_config.get("modal_object_map", {}).items()}
        spec_value = spec_map.get(requested) or spec_map.get(resolved_label.upper()) or resolved_label
        return {"type": requested, "label": resolved_label, "decorator": spec_value}

    def run(self, code, inputs, output_dir):
        # Sync workspace using modal volume
        def upload_to_volume(local_file, remote_file):
            subprocess.run(["modal", "volume", "rm", self.volume.name, remote_file], check=False)
            subprocess.run(["modal", "volume", "put", self.volume.name, local_file, remote_file], check=True)
        
        code_sync = self.config.get("storage", {}).get("code_sync", {})
        sync_workspace(
            paths=code_sync.get("dirs") or [],
            exclude_files_global=code_sync.get("exclude_files_global"),
            exclude_dirs_global=code_sync.get("exclude_dirs_global"),
            exclude_files_map=code_sync.get("exclude_files_map"),
            exclude_dirs_map=code_sync.get("exclude_dirs_map"),
            upload_func=upload_to_volume
        )

        # Prefetch models remotely
        storage_cfg = self.config.get("storage", {})
        repos = storage_cfg.get("models", [])
        if repos:
            prefetch_kwargs: Dict[str, Any] = {"repos": repos}
            hf_home = storage_cfg.get("hf_home")
            if hf_home:
                prefetch_kwargs["hf_home"] = hf_home
            prefetch_hf_remote.remote(**prefetch_kwargs)

        # Run script remotely
        script = code
        venv = self.config.get("build", {}).get("venv", {}).get("path", f"{DEFAULT_REMOTE_ROOT}/venv")
        workdir = self.config.get("run", {}).get("workdir", DEFAULT_REMOTE_ROOT)
        result_info: Optional[Dict[str, Any]] = None
        stream = run_script_remote.remote_gen(
            script_path=script,
            venv=venv,
            workdir=workdir
        )
        try:
            for event in stream:
                if isinstance(event, dict):
                    kind = event.get("event")
                    if kind == "log":
                        message = event.get("data", "")
                        if message:
                            print(message, end="" if message.endswith("\n") else "\n")
                    elif kind == "result":
                        result_info = event
                else:
                    print(event)
        finally:
            if hasattr(stream, "close"):
                stream.close()

        if not result_info:
            raise RuntimeError("Modal execution did not return completion metadata")

        returncode = result_info.get("returncode", 0)
        if returncode:
            raise subprocess.CalledProcessError(returncode, script)

        self.sync_outputs(local_dir=output_dir)
        return {"status": "success", "result": result_info}

    def sync_outputs(self, local_dir: str, remote_dir: Optional[str] = None) -> None:
        target_remote = remote_dir or self.remote_outputs
        local_path = Path(local_dir).expanduser().resolve()
        remote_hashes = get_remote_hashes_remote.remote(target_remote)
        if not remote_hashes:
            if local_path.exists():
                shutil.rmtree(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            return
        zip_bytes = zip_remote_dir_remote.remote(target_remote)
        if not zip_bytes:
            return
        if local_path.exists():
            shutil.rmtree(local_path)
        local_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(local_path)

    def _cleanup_app(self) -> None:
        ctx = getattr(self, "_app_ctx", None)
        if ctx:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._app_ctx = None

# Modal-specific functions (adapted from modal_utils)
class _Defer:
    def __init__(self, name: str):
        self.name = name

class _AppProxy:
    def __init__(self):
        self._app = None

    def _resolve_kwargs(self, dkwargs: Dict) -> Dict:
        resolved = {}
        for k, v in dkwargs.items():
            if isinstance(v, _Defer):
                v = globals().get(v.name)
            if k == "volumes" and isinstance(v, dict):
                nv = {}
                for mp, vol in v.items():
                    if isinstance(vol, _Defer):
                        vol = globals().get(vol.name)
                    nv[mp] = vol
                if any(val is None for val in nv.values()):
                    v = None
                else:
                    v = nv
            if v is None:
                continue
            resolved[k] = v
        return resolved

    def bind(self, app_obj):
        import sys
        self._app = app_obj
        mod = sys.modules[__name__]
        for name, obj in list(vars(mod).items()):
            if callable(obj) and hasattr(obj, "_modal_defer"):
                dargs, dkwargs = getattr(obj, "_modal_defer")
                resolved = self._resolve_kwargs(dkwargs)
                wrapped = self._app.function(*dargs, **resolved)(obj)
                setattr(mod, name, wrapped)
        return self

    def function(self, *dargs, **dkwargs):
        def decorator(fn):
            setattr(fn, "_modal_defer", (dargs, dkwargs))
            return fn
        return decorator

app_proxy = _AppProxy()
_image = None
_vol = None
_gpu = None
DEFER_IMAGE = _Defer("_image")
DEFER_VOLUME = _Defer("_vol")
DEFER_GPU = _Defer("_gpu")

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME})
def get_remote_hashes_remote(remote_path: str) -> Dict[str, str]:
    if not os.path.exists(remote_path):
        return {}
    if os.path.isfile(remote_path):
        return {os.path.basename(remote_path): _hash_file_local(remote_path)}
    out: Dict[str, str] = {}
    for root, _, files in os.walk(remote_path):
        for name in files:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, remote_path).replace("\\", "/")
            out[rel] = _hash_file_local(p)
    return out

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME})
def zip_remote_dir_remote(remote_dir: str) -> Optional[bytes]:
    if not os.path.exists(remote_dir):
        return None
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(remote_dir):
            for name in files:
                p = os.path.join(root, name)
                arc = os.path.relpath(p, remote_dir)
                zf.write(p, arc)
    buf.seek(0)
    return buf.read()

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME}, gpu=DEFER_GPU, timeout=86400)
def run_script_remote(script_path: str, venv: Optional[str] = None, workdir: str = DEFAULT_REMOTE_ROOT):
    os.chdir(workdir)
    if venv:
        act = venv if venv.endswith("/bin/activate") else os.path.join(venv, "bin/activate")
        cmd = f"source {act} && python {script_path}"
    else:
        cmd = f"python {script_path}"
    proc = subprocess.Popen(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    logs: List[str] = []
    try:
        if proc.stdout:
            for line in proc.stdout:
                logs.append(line)
                yield {"event": "log", "data": line}
        returncode = proc.wait()
    finally:
        if proc.stdout:
            proc.stdout.close()
    yield {
        "event": "result",
        "returncode": returncode,
        "workdir": workdir,
        "logs": "".join(logs),
    }

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME})
def prefetch_hf_remote(repos: List, hf_home: str = DEFAULT_HF_HOME) -> str:
    from pathlib import Path as _P
    from huggingface_hub import snapshot_download
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
def _hash_file_local(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
