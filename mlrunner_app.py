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
    # Run commands in sequence within a single execution context
    runner.run(code="commands/run_sequence.py", output_dir="./out_local")

if __name__ == "__main__":
    run_model()
