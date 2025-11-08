import os
import subprocess
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional, List

try:
    from .modal_backend import ModalBackend
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

from .shell_options import ShellOptions
from .utils import sync_workspace, prefetch_hf, run_script, sync_outputs, DEFAULT_REMOTE_ROOT

class MLRunner:
    def __init__(self, backend: str = "modal", task: str = "general", **config):
        self.backend = backend
        self.task = task
        self.config = self._apply_defaults(config)
        if backend == "modal" and not MODAL_AVAILABLE:
            raise ImportError("Modal backend requires 'modal' package: pip install modal")
        self.backend_impl = self._get_backend()

    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        defaults: Dict[str, Any] = {
            "gpu": {
                "count": 1,
            },
            "storage": {
                "code_sync": {
                    "dirs": [],
                    "exclude_files_global": [],
                    "exclude_dirs_global": [],
                    "exclude_files_map": {},
                    "exclude_dirs_map": {},
                },
                "models": [],
            },
            "scale": {"timeout": 86400},
            "shell": {
                "workdir": DEFAULT_REMOTE_ROOT,
                "startup_cmd": None,
                "env": {},
                "gpu": None,
                "sync_code": True,
                "prefetch_models": False,
                "interactive": True,
            },
        }

        def _merge(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in overrides.items():
                if isinstance(value, dict):
                    existing = base.get(key)
                    if isinstance(existing, dict):
                        base[key] = _merge(existing, value)
                    else:
                        base[key] = deepcopy(value)
                else:
                    base[key] = value
            return base

        merged = _merge(deepcopy(defaults), config)

        if self.task == "diffusion":
            storage = merged.setdefault("storage", {})
            if not storage.get("models"):
                storage["models"] = [
                    ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl-base"),
                    ("stabilityai/stable-diffusion-xl-refiner-1.0", "sdxl-refiner"),
                    ("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "openclip-vit-b-32"),
                ]
            gpu_conf = merged.setdefault("gpu", {})
            gpu_conf.setdefault("type", "A100")

        return merged

    def _get_backend(self):
        return self._create_backend(self.config)

    def _create_backend(self, config: Dict[str, Any]):
        if self.backend == "modal":
            return ModalBackend(config)
        if self.backend == "local":
            return LocalBackend(config)
        raise NotImplementedError(f"Backend {self.backend} not implemented")

    def run(self, code: str, inputs: Optional[Dict[str, Any]] = None, output_dir: str = "./results", pipeline: Optional[List[Dict]] = None):
        if inputs:
            # Set env/args from inputs
            os.environ.update({k: str(v) for k, v in inputs.get("env", {}).items()})
        self.backend_impl.run(code=code, inputs=inputs, output_dir=output_dir)
        if pipeline:
            # Handle multi-stage
            for stage in pipeline:
                self.run(stage["script"], inputs=inputs, output_dir=output_dir)
        self.sync_outputs(output_dir)
        return {"status": "success", "outputs": list(Path(output_dir).glob("*"))}

    def sync_outputs(self, local_dir: str, remote_dir: Optional[str] = None, allowed_extensions: Optional[List[str]] = None) -> None:
        backend_sync = getattr(self.backend_impl, "sync_outputs", None)
        if callable(backend_sync):
            backend_sync(local_dir=local_dir, remote_dir=remote_dir, allowed_extensions=allowed_extensions)
        else:
            sync_outputs(local_dir, remote_dir=remote_dir or f"{DEFAULT_REMOTE_ROOT}/out_local", allowed_extensions=allowed_extensions)

    def push_directory(
        self,
        local_dir: str,
        remote_root: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ) -> None:
        backend_push = getattr(self.backend_impl, "push_directory", None)
        if callable(backend_push):
            backend_push(
                local_dir=local_dir,
                remote_root=remote_root,
                allowed_extensions=allowed_extensions,
                exclude_dirs=exclude_dirs,
            )
        else:
            sync_workspace(
                paths=[local_dir],
                exclude_dirs_global=exclude_dirs,
                remote_root=remote_root or DEFAULT_REMOTE_ROOT,
                allowed_extensions=allowed_extensions,
            )

    def shell(
        self,
        config: Optional[str] = None,
        dir: Optional[str] = None,
        gpu: Optional[Any] = None,
        env: Optional[Dict[str, Any]] = None,
        startup_cmd: Optional[str] = None,
        sync_code: Optional[bool] = None,
        prefetch_models: Optional[bool] = None,
        interactive: Optional[bool] = None,
        stream: Optional[bool] = True,
    ):
        call_config = deepcopy(self.config)
        if config:
            call_config["config_path"] = config
        shell_cfg = call_config.get("shell", {})
        workdir = str(dir or shell_cfg.get("workdir") or DEFAULT_REMOTE_ROOT)
        base_env = shell_cfg.get("env") or {}
        merged_env: Dict[str, str] = {str(k): str(v) for k, v in base_env.items()}
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})
        effective_gpu = gpu if gpu is not None else shell_cfg.get("gpu")
        effective_startup = startup_cmd or shell_cfg.get("startup_cmd")
        effective_sync_code = sync_code if sync_code is not None else shell_cfg.get("sync_code", True)
        effective_prefetch = prefetch_models if prefetch_models is not None else shell_cfg.get("prefetch_models", False)
        effective_interactive = interactive if interactive is not None else shell_cfg.get("interactive", True)
        effective_stream = stream if stream is not None else shell_cfg.get("stream", True)
        options = ShellOptions(
            workdir=workdir,
            env=merged_env,
            startup_cmd=effective_startup,
            gpu=effective_gpu,
            sync_code=bool(effective_sync_code),
            prefetch_models=bool(effective_prefetch),
            config_path=call_config.get("config_path"),
            interactive=bool(effective_interactive),
            stream=bool(effective_stream),
        )
        backend = self.backend_impl
        temp_backend = None
        should_replace_backend = config and config != self.config.get("config_path")
        if should_replace_backend:
            temp_backend = self._create_backend(call_config)
            backend = temp_backend
        try:
            return backend.shell(options)
        finally:
            if temp_backend:
                temp_backend.close()

class LocalBackend:
    def __init__(self, config):
        self.config = config

    def run(self, code, inputs, output_dir):
        # Local run using utils
        code_sync = self.config.get("storage", {}).get("code_sync", {})
        sync_workspace(
            paths=code_sync.get("dirs", []),
            exclude_files_global=code_sync.get("exclude_files_global"),
            exclude_dirs_global=code_sync.get("exclude_dirs_global"),
            exclude_files_map=code_sync.get("exclude_files_map"),
            exclude_dirs_map=code_sync.get("exclude_dirs_map")
        )
        models = self.config.get("storage", {}).get("models", [])
        if models:
            prefetch_hf(models)
        venv = self.config.get("build", {}).get("venv", {}).get("path")
        run_script(code, venv=venv)
        # No sync needed for local

    def shell(self, options: ShellOptions):
        workdir_path = Path(options.workdir).expanduser()
        if options.sync_code:
            code_sync = self.config.get("storage", {}).get("code_sync", {})
            sync_workspace(
                paths=code_sync.get("dirs", []),
                exclude_files_global=code_sync.get("exclude_files_global"),
                exclude_dirs_global=code_sync.get("exclude_dirs_global"),
                exclude_files_map=code_sync.get("exclude_files_map"),
                exclude_dirs_map=code_sync.get("exclude_dirs_map"),
                remote_root=str(workdir_path),
            )
        if options.prefetch_models:
            models = self.config.get("storage", {}).get("models", [])
            if models:
                prefetch_hf(models)
        workdir_path.mkdir(parents=True, exist_ok=True)
        env_vars = os.environ.copy()
        env_vars.update(options.env)
        exit_code = 0
        if options.startup_cmd:
            result = subprocess.run(
                options.startup_cmd,
                shell=True,
                executable="/bin/bash",
                cwd=str(workdir_path),
                env=env_vars,
                check=False,
            )
            exit_code = result.returncode if result.returncode is not None else 0
            if not options.interactive:
                return exit_code
            if exit_code:
                return exit_code
        if not options.interactive:
            return exit_code
        shell_path = env_vars.get("SHELL") or os.environ.get("SHELL") or "/bin/bash"
        result = subprocess.run(
            [shell_path],
            cwd=str(workdir_path),
            env=env_vars,
            check=False,
        )
        return result.returncode

    def close(self):
        return None

    def push_directory(
        self,
        local_dir: str,
        remote_root: Optional[str] = None,
        allowed_extensions: Optional[List[str]] = None,
        exclude_dirs: Optional[List[str]] = None,
    ) -> None:
        target_remote = remote_root or DEFAULT_REMOTE_ROOT
        sync_workspace(
            paths=[local_dir],
            exclude_dirs_global=exclude_dirs,
            remote_root=target_remote,
            allowed_extensions=allowed_extensions,
        )
