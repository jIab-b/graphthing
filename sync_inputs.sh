#!/bin/bash
set -e
echo "Syncing sglang -> /workspace/sglang and commands -> /workspace"
python - <<'PY'
from mlrunner import MLRunner
r = MLRunner(backend='modal', config_path='lang_config.txt')
print("Pushing sglang directory...")
r.push_directory(local_dir='sglang', remote_root='/workspace/sglang')
print("Pushing commands directory...")
r.push_directory(local_dir='commands', remote_root='/workspace')
print("Sync inputs complete.")
PY


