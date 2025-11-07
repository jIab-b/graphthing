import os
from pathlib import Path
from copy import deepcopy
from typing import Dict, Any, Optional, List

try:
    from .modal_backend import ModalBackend
    MODAL_AVAILABLE = True
except ImportError:
    MODAL_AVAILABLE = False

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
        if self.backend == "modal":
            return ModalBackend(self.config)
        elif self.backend == "local":
            return LocalBackend(self.config)  # Stub for local
        else:
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

    def sync_outputs(self, local_dir: str, remote_dir: Optional[str] = None) -> None:
        backend_sync = getattr(self.backend_impl, "sync_outputs", None)
        if callable(backend_sync):
            backend_sync(local_dir=local_dir, remote_dir=remote_dir)
        else:
            sync_outputs(local_dir, remote_dir=remote_dir or f"{DEFAULT_REMOTE_ROOT}/out_local")

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
