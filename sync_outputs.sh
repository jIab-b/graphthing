#!/bin/bash
set -e
echo "Syncing remote /workspace/out_local -> local ./out_local"
python - <<'PY'
from mlrunner_app import runner
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
print("Sync outputs complete.")
PY