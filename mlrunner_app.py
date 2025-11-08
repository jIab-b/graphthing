import os
from pathlib import Path
from mlrunner import MLRunner




runner = MLRunner(
    backend="modal",
    config_path="lang_config.txt",
    gpu={"type": "L4"},
    storage={
        "code_sync": {
            "dirs": ["commands"],
            "exclude_files_global": ["**/*.pyc"],
            "exclude_dirs_global": ["**/__pycache__/**"]
        }
    }
)

def run_model():
    runner.run(code="commands/run_sequence.py", output_dir="./out_local")


def sync_sglang_sources():
    allowed_extensions = [
        "py",
        "pyi",
        "md",
        "txt",
        "json",
        "yaml",
        "yml",
        "sh",
        "cmake",
        "hpp",
        "h",
        "c",
        "cc",
        "cpp",
        "cu",
        "ptx",
        "js",
        "ts",
    ]
    runner.sync_outputs(
        local_dir="sglang",
        remote_dir="/sgl-workspace/sglang",
        allowed_extensions=allowed_extensions,
    )

if __name__ == "__main__":
    #run_model()
    sync_sglang_sources()