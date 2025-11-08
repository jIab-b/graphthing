import os
import subprocess
from pathlib import Path

REMOTE_OUT = os.environ.get("REMOTE_OUT_DIR", "/workspace/out_local")
WORKLOAD = os.environ.get("WORKLOAD", "/workspace/commands/sglang_workload.py")
BASE = os.environ.get("TRACE_BASE", "sglang_vllm_trace")

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def run_nsys():
    print(f"NSYS: starting profile to {REMOTE_OUT} base={BASE} (this may take a while)")
    out_base = f"{REMOTE_OUT}/{BASE}"
    cmd = [
        "nsys", "profile",
        "--force-overwrite=true",
        "-t", "cuda,cublas,cudnn,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "-o", out_base,
        "python", WORKLOAD,
    ]
    print(f"NSYS: cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"NSYS: profile complete -> {out_base}.qdrep")
    stats_cmd = [
        "nsys", "stats",
        "--report", "gpukernsum",
        "--format", "csv",
        "-o", f"{out_base}_kernsum",
        f"{out_base}.qdrep",
    ]
    print("NSYS: generating kernel summary CSV")
    subprocess.run(stats_cmd, check=True)
    print(f"NSYS: kernel summary written -> {out_base}_kernsum.csv")

def run_ncu():
    print("NCU: starting kernel metrics collection (this may take a while)")
    out_base = f"{REMOTE_OUT}/{BASE}"
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--set", "full",
        "--kernel-name-base", "demangled",
        "-o", out_base,
        "python", WORKLOAD,
    ]
    print(f"NCU: cmd={' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"NCU: metrics complete -> {out_base}.ncu-rep")

if __name__ == "__main__":
    print(f"TRACE: REMOTE_OUT_DIR={REMOTE_OUT}")
    print(f"TRACE: WORKLOAD={WORKLOAD}")
    print(f"TRACE: TRACE_BASE={BASE}")
    ensure_dir(REMOTE_OUT)
    os.environ.setdefault("CUDA_CACHE_PATH", f"{REMOTE_OUT}/compute_cache")
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", "2147483648")
    ensure_dir(os.environ["CUDA_CACHE_PATH"])
    print(f"TRACE: CUDA_CACHE_PATH={os.environ['CUDA_CACHE_PATH']}")
    run_nsys()
    run_ncu()


