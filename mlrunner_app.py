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
        remote_dir="/workspace/sglang",
        allowed_extensions=allowed_extensions,
    )


def push_sglang_changes():
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
    exclude_dirs = [
        "**/__pycache__/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/node_modules/**",
    ]
    runner.push_directory(
        local_dir="sglang",
        remote_root="/workspace",
        allowed_extensions=allowed_extensions,
        exclude_dirs=exclude_dirs,
    )

def push_commands():
    allowed_extensions = [
        "py",
        "pyi",
        "txt",
        "json",
        "yaml",
        "yml",
    ]
    exclude_dirs = [
        "**/__pycache__/**",
        "**/.git/**",
        "**/build/**",
        "**/dist/**",
        "**/node_modules/**",
    ]
    runner.push_directory(
        local_dir="commands",
        remote_root="/workspace",
        allowed_extensions=allowed_extensions,
        exclude_dirs=exclude_dirs,
    )

def run_full_capture():
    #runner.run(code="commands/export_model_graphs.py", output_dir="./out_local")
    runner.run(code="commands/trace_with_nsight.py", output_dir="./out_local")
    allowed_extensions = [
        "json",
        "pt2",
        "dot",
        "nsys-rep",
        "qdrep",
        "sqlite",
        "ncu-rep",
        "log",
        "txt",
        "csv",
    ]
    runner.sync_outputs(
        local_dir="out_local",
        remote_dir="/workspace/out_local",
        allowed_extensions=allowed_extensions,
    )

if __name__ == "__main__":
    #run_model()
    #push_commands()
    #sync_sglang_sources()
    push_commands()
    run_full_capture()