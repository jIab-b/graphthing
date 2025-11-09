import atexit
import hashlib
import io
import json
import os
import posixpath
import shlex
import sys
import shutil
import subprocess
import zipfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import modal
from modal import Image, Volume

from mlrunner.shell_options import ShellOptions
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
        return {"type": requested, "label": resolved_label, "decorator": str(spec_value).upper()}

    def run(self, code, inputs, output_dir):
        code_sync = self.config.get("storage", {}).get("code_sync", {})
        # Use remote hash checking for incremental sync
        allowed_exts = None  # For code sync, allow all extensions
        def get_remote_hashes_func(path: str):
            return get_remote_hashes_remote.remote(path, allowed_extensions=allowed_exts)
        sync_workspace(
            paths=code_sync.get("dirs") or [],
            exclude_files_global=code_sync.get("exclude_files_global"),
            exclude_dirs_global=code_sync.get("exclude_dirs_global"),
            exclude_files_map=code_sync.get("exclude_files_map"),
            exclude_dirs_map=code_sync.get("exclude_dirs_map"),
            upload_func=self._upload_to_volume,
            get_remote_hashes_func=get_remote_hashes_func,
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

    def shell(self, options: ShellOptions):
        workdir = self._normalize_remote_path(options.workdir)
        if options.sync_code:
            code_sync = self.config.get("storage", {}).get("code_sync", {})
            # Use remote hash checking for incremental sync
            def get_remote_hashes_func(path: str):
                return get_remote_hashes_remote.remote(path, allowed_extensions=None)
            sync_workspace(
                paths=code_sync.get("dirs") or [],
                exclude_files_global=code_sync.get("exclude_files_global"),
                exclude_dirs_global=code_sync.get("exclude_dirs_global"),
                exclude_files_map=code_sync.get("exclude_files_map"),
                exclude_dirs_map=code_sync.get("exclude_dirs_map"),
                upload_func=self._upload_to_volume,
                get_remote_hashes_func=get_remote_hashes_func,
            )
        if options.prefetch_models:
            storage_cfg = self.config.get("storage", {})
            repos = storage_cfg.get("models", [])
            if repos:
                prefetch_kwargs: Dict[str, Any] = {"repos": repos}
                hf_home = storage_cfg.get("hf_home")
                if hf_home:
                    prefetch_kwargs["hf_home"] = hf_home
                prefetch_hf_remote.remote(**prefetch_kwargs)
        gpu_spec = self._select_gpu_decorator(options.gpu)
        session = ModalShellSession(self, workdir, options.env, gpu_spec, stream=options.stream)
        exit_code = 0
        if options.startup_cmd:
            exit_code = session.run_command(options.startup_cmd)
            if not options.interactive:
                return exit_code
            if exit_code:
                return exit_code
        if not options.interactive:
            return exit_code
        return session.interactive()

    def sync_outputs(self, local_dir: str, remote_dir: Optional[str] = None, allowed_extensions: Optional[List[str]] = None) -> None:
        target_remote = remote_dir or self.remote_outputs
        local_path = Path(local_dir).expanduser().resolve()
        zip_bytes = zip_remote_dir_remote.remote(target_remote, allowed_extensions=allowed_extensions)
        if not zip_bytes:
            return
        local_path.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            zf.extractall(local_path)
        if allowed_extensions:
            _filter_local_extensions(local_path, allowed_extensions)
        clear_remote_outputs_remote.remote(target_remote, allowed_extensions=allowed_extensions)

    def _cleanup_app(self) -> None:
        ctx = getattr(self, "_app_ctx", None)
        if ctx:
            try:
                ctx.__exit__(None, None, None)
            except Exception:
                pass
            self._app_ctx = None

    def _upload_to_volume(self, local_file: str, remote_file: str):
        subprocess.run(["modal", "volume", "rm", self.volume.name, remote_file], check=False)
        subprocess.run(["modal", "volume", "put", self.volume.name, local_file, remote_file], check=True)

    def _select_gpu_decorator(self, override: Optional[Any]) -> Any:
        if override is None:
            return self.gpu_info["decorator"]
        if isinstance(override, bool):
            return self.gpu_info["decorator"] if override else None
        gpu_config = self.config.get("gpu", {})
        requested = str(override).upper()
        alias_map = {k.upper(): v for k, v in gpu_config.get("type_aliases", {}).items()}
        resolved = alias_map.get(requested, requested)
        spec_map = {k.upper(): v for k, v in gpu_config.get("modal_object_map", {}).items()}
        spec_value = spec_map.get(requested) or spec_map.get(resolved.upper()) or spec_map.get(resolved) or resolved
        return str(spec_value).upper()

    def close(self):
        self._cleanup_app()

    def _normalize_remote_path(self, path: Optional[str]) -> str:
        base = path or DEFAULT_REMOTE_ROOT
        if not base:
            base = DEFAULT_REMOTE_ROOT
        candidate = str(base)
        if not candidate.startswith("/"):
            candidate = posixpath.join(DEFAULT_REMOTE_ROOT, candidate)
        normalized = posixpath.normpath(candidate)
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        return normalized

    @contextmanager
    def _gpu_context(self, spec: Any):
        previous = globals().get("_gpu")
        globals()["_gpu"] = spec
        try:
            yield
        finally:
            globals()["_gpu"] = previous

    def push_directory(
        self,
        local_dir: str,
        remote_root: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ) -> None:
        remote = self._normalize_remote_path(remote_root)
        # Use remote hash checking for proper incremental sync
        def get_remote_hashes_func(path: str):
            return get_remote_hashes_remote.remote(path, allowed_extensions=allowed_extensions)
        sync_workspace(
            paths=[local_dir],
            exclude_dirs_global=exclude_dirs,
            remote_root=remote,
            upload_func=self._upload_to_volume,
            allowed_extensions=allowed_extensions,
            get_remote_hashes_func=get_remote_hashes_func,
        )


class ModalShellSession:
    def __init__(self, backend: ModalBackend, workdir: str, env: Dict[str, str], gpu_spec: Any, stream: bool = True):
        self.backend = backend
        self.base_dir = backend._normalize_remote_path(workdir)
        self.workdir = self.base_dir
        self.env = {str(k): str(v) for k, v in (env or {}).items()}
        self.gpu_spec = gpu_spec
        self.stream = bool(stream)
        self.output = sys.stdout
        self.last_exit_code = 0

    def interactive(self) -> int:
        inp = sys.stdin
        out = self.output
        while True:
            if hasattr(inp, "isatty") and inp.isatty():
                out.write(self._prompt())
                out.flush()
            line = inp.readline()
            if line == "":
                if hasattr(inp, "isatty") and inp.isatty():
                    out.write("\n")
                    out.flush()
                return self.last_exit_code
            command = line.rstrip("\n")
            if not command.strip():
                continue
            result = self._handle_command(command)
            if result is False:
                return self.last_exit_code

    def run_command(self, command: str) -> int:
        if not command.strip():
            return self.last_exit_code
        logs = ""
        exit_code = 0
        if self.stream:
            with self.backend._gpu_context(self.gpu_spec):
                stream = execute_shell_command_remote.remote_gen(
                    command=command,
                    workdir=self.workdir,
                    env=self.env,
                )
            try:
                result_info = None
                for event in stream:
                    if isinstance(event, dict):
                        kind = event.get("event")
                        if kind == "log":
                            message = event.get("data", "")
                            if message:
                                self.output.write(message)
                                if not message.endswith("\n"):
                                    self.output.write("\n")
                                self.output.flush()
                        elif kind == "result":
                            result_info = event
                    else:
                        self.output.write(str(event))
                if result_info:
                    logs = result_info.get("logs", "") or ""
                    exit_code = int(result_info.get("returncode", 0))
            finally:
                if hasattr(stream, "close"):
                    stream.close()
        else:
            with self.backend._gpu_context(self.gpu_spec):
                result = execute_shell_command_remote.remote(
                    command=command,
                    workdir=self.workdir,
                    env=self.env,
                )
            if isinstance(result, dict):
                logs = result.get("logs", "") or ""
                exit_code = int(result.get("returncode", 0))
            else:
                logs = str(result or "")
            if logs:
                self.output.write(logs)
                if not logs.endswith("\n"):
                    self.output.write("\n")
                self.output.flush()
        self.last_exit_code = exit_code
        return exit_code

    def _handle_command(self, command: str):
        parts = shlex.split(command)
        if not parts:
            return None
        head = parts[0]
        if head in ("exit", "quit"):
            return False
        if head == "cd":
            target = parts[1] if len(parts) > 1 else None
            self._change_directory(target)
            return None
        self.last_exit_code = self.run_command(command)
        return None

    def _change_directory(self, target: Optional[str]):
        if target is None or target == "~":
            self.workdir = self.base_dir
            return
        candidate = target
        if candidate.startswith("~"):
            suffix = candidate[1:].lstrip("/")
            candidate = posixpath.join(self.base_dir, suffix)
        if candidate.startswith("/"):
            new_path = candidate
        else:
            new_path = posixpath.join(self.workdir, candidate)
        normalized = posixpath.normpath(new_path)
        if not normalized.startswith("/"):
            normalized = "/" + normalized
        self.workdir = normalized

    def _prompt(self) -> str:
        return f"{self.workdir}$ "

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

def _filter_local_extensions(base_path: Path, allowed_extensions: Sequence[str]) -> None:
    allowed = {ext.lower().lstrip(".") for ext in allowed_extensions}
    for file_path in base_path.rglob("*"):
        if file_path.is_file():
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in allowed:
                file_path.unlink()
    for dir_path in sorted(base_path.rglob("*"), reverse=True):
        if dir_path.is_dir():
            try:
                next(dir_path.iterdir())
            except StopIteration:
                dir_path.rmdir()

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME}, timeout=3600)
def get_remote_hashes_remote(remote_path: str, allowed_extensions: Optional[List[str]] = None) -> Dict[str, str]:
    if not os.path.exists(remote_path):
        return {}
    if os.path.isfile(remote_path):
        return {os.path.basename(remote_path): _hash_file_local(remote_path)}
    out: Dict[str, str] = {}
    allow: Optional[set] = None
    if allowed_extensions:
        allow = {ext.lower().lstrip(".") for ext in allowed_extensions}
    for root, _, files in os.walk(remote_path):
        for name in files:
            if allow is not None:
                ext = os.path.splitext(name)[1].lower().lstrip(".")
                if ext not in allow:
                    continue
            p = os.path.join(root, name)
            rel = os.path.relpath(p, remote_path).replace("\\", "/")
            out[rel] = _hash_file_local(p)
    return out

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME}, timeout=3600)
def zip_remote_dir_remote(remote_dir: str, allowed_extensions: Optional[List[str]] = None) -> Optional[bytes]:
    if not os.path.exists(remote_dir):
        return None
    allow: Optional[set] = None
    if allowed_extensions:
        allow = {ext.lower().lstrip(".") for ext in allowed_extensions}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(remote_dir):
            for name in files:
                if allow is not None:
                    ext = os.path.splitext(name)[1].lower().lstrip(".")
                    if ext not in allow:
                        continue
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

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME}, gpu=DEFER_GPU, timeout=86400)
def execute_shell_command_remote(command: str, workdir: str, env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    os.makedirs(workdir, exist_ok=True)
    os.chdir(workdir)
    if env:
        os.environ.update({str(k): str(v) for k, v in env.items()})
    proc = subprocess.Popen(
        command,
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
        "returncode": int(returncode),
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

@app_proxy.function(image=DEFER_IMAGE, volumes={DEFAULT_REMOTE_ROOT: DEFER_VOLUME}, timeout=3600)
def clear_remote_outputs_remote(remote_dir: str, allowed_extensions: Optional[List[str]] = None) -> None:
    if not os.path.exists(remote_dir):
        return
    allow: Optional[set] = None
    if allowed_extensions:
        allow = {ext.lower().lstrip('.') for ext in allowed_extensions}
    for root, dirs, files in os.walk(remote_dir, topdown=False):
        for name in files:
            if allow is not None:
                ext = os.path.splitext(name)[1].lower().lstrip('.')
                if ext not in allow:
                    continue
            try:
                os.remove(os.path.join(root, name))
            except FileNotFoundError:
                pass
        for name in dirs:
            try:
                os.rmdir(os.path.join(root, name))
            except (FileNotFoundError, OSError):
                pass

def _hash_file_local(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()
