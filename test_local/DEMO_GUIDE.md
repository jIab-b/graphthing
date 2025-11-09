## Qwen3-0.6B RTX 2060 Super Demo Checklist

### 1. Environment Prep
- Ensure the repo dependencies are installed inside `test_venv` (`pip install -r requirements.txt` if needed).
- Activate the virtualenv before running any tooling:  
  ```bash
  source test_venv/bin/activate
  ```
- Confirm CUDA sees the target board (`nvidia-smi`) and no other heavy workloads are running.

### 2. Export Graph Buckets + Metadata
Run the export harness from the repo root (`/home/beed/graphthing/test_local`):
```bash
python export_model_graphs.py
```
This emits:
- TorchExport artifacts for all supported buckets (prefill/decode × batch/cache combos) under `out_local/*.pt2` with matching `.dot` visualizations.
- A manifest at `out_local/export_manifest.json` capturing scenario metadata, runtime policy, KV cache math, hardware constraints, and profiling hooks. This JSON is the hand-off to the AI compiler.

### 3. Reading the Manifest
`export_manifest.json` includes:
- `scenarios`: inputs/outputs, kv cache bytes consumed/produced, CUDA graph eligibility, and artifact paths.
- `runtime_policy`: batching window, speculative decoding depth, CUDA graph buckets, KV paging defaults.
- `hardware`: live snapshot of the RTX 2060 Super (SM count, memory, driver).
- `precision_options` & `quant_calibration`: indicate allowable precisions and calibration status.
Use this file to seed the compiler’s search space or tooling dashboards.

### 4. Profiling Hooks (Nsight Systems & Compute)
`trace_with_nsight.py` already wires up NSYS/NCU captures plus environment dumps.

1. Pick or author a serving workload script (default `sglang_workload.py`).  
2. Export env vars if you need non-defaults:
   ```bash
   export WORKLOAD=/path/to/your_workload.py
   export TRACE_BASE=qwen3_demo
   export REMOTE_OUT_DIR=/home/beed/graphthing/test_local/out_local
   ```
3. (Optional) Dry run the workload first to ensure it succeeds.
4. Launch profiling:
   ```bash
   python trace_with_nsight.py
   ```
   - Generates NSYS `.qdrep/.nsys-rep`, CSV summaries, and NCU `.ncu-rep` in `out_local/`.
   - Debug artifacts (env info, logs) land in `out_local/debug/`.

These traces provide the latency/throughput counters that the learned cost model will consume alongside the manifest.

### 5. Next Steps
- Feed `out_local/export_manifest.json` plus NSYS/NCU stats into your MLIR-based compiler prototype.
- Implement quantization calibration if INT8 KV is desired (hook into `quant_calibration` field once data is ready).
- Extend `export_model_graphs.py` scenarios if you need more buckets; the runtime policy and manifest update automatically.
