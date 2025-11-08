from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ShellOptions:
    workdir: str
    env: Dict[str, str]
    startup_cmd: Optional[str]
    gpu: Optional[Any]
    sync_code: bool
    prefetch_models: bool
    config_path: Optional[str]
    interactive: bool
    stream: bool

