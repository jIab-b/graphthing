#!/bin/bash
set -e
echo "Syncing remote /workspace/out_local -> local ./out_local"
python - <<'PY'
from mlrunner import MLRunner
r = MLRunner(backend='modal', config_path='lang_config.txt')
allowed = ['json','pt2','dot','nsys-rep','qdrep','sqlite','ncu-rep','log','txt','csv']
r.sync_outputs(local_dir='out_local', remote_dir='/workspace/out_local', allowed_extensions=allowed)
print("Sync outputs complete.")
PY


