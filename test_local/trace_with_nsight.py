import os
import sys
import subprocess
import shutil
import platform
from pathlib import Path

HF_HOME_PATH = os.path.expanduser(os.environ.get("HF_HOME", "~/hf"))
os.environ["HF_HOME"] = HF_HOME_PATH
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME_PATH)

BASE_DIR = Path(__file__).resolve().parent

REMOTE_OUT = os.environ.get("REMOTE_OUT_DIR", str(BASE_DIR / "out_local"))
WORKLOAD = os.environ.get("WORKLOAD", str(BASE_DIR / "sglang_workload.py"))
BASE = os.environ.get("TRACE_BASE", "sglang_vllm_trace")
DEBUG_DIR = os.environ.get("DEBUG_DIR", f"{REMOTE_OUT}/debug")
SANITY = os.environ.get("SANITY_BEFORE_PROFILE", "0") == "1"
USE_STRACE = os.environ.get("NSYS_STRACE", "0") == "1"

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def _write_file(path: str, content: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)

def _run_and_log(cmd, log_name: str) -> int:
    print(f"DBG: running {' '.join(cmd)}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_path = f"{DEBUG_DIR}/{log_name}"
    _write_file(out_path, proc.stdout or "")
    print(f"DBG: rc={proc.returncode} -> {out_path}")
    return proc.returncode

def _gather_env_debug() -> None:
    ensure_dir(DEBUG_DIR)
    env_lines = []
    env_lines.append(f"python_exe={sys.executable}")
    env_lines.append(f"python_version={sys.version}")
    env_lines.append(f"platform={platform.platform()}")
    env_lines.append(f"uname={platform.uname()}")
    env_lines.append(f"PATH={os.environ.get('PATH','')}")
    env_lines.append(f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')}")
    env_lines.append(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')}")
    env_lines.append(f"which_nsys={shutil.which('nsys')}")
    env_lines.append(f"which_ncu={shutil.which('ncu')}")
    _write_file(f"{DEBUG_DIR}/env.txt", "\n".join(map(str, env_lines)) + "\n")
    _run_and_log(["nsys", "--version"], "nsys_version.log")
    _run_and_log(["ncu", "--version"], "ncu_version.log")
    _run_and_log(["nvidia-smi", "-L"], "nvidia_smi_L.log")
    _run_and_log(["nvidia-smi"], "nvidia_smi.log")
    _run_and_log(["uname", "-a"], "uname.log")
    _run_and_log(["bash", "-lc", "cat /etc/os-release || true"], "os_release.log")
    _run_and_log([sys.executable, "-V"], "python_V.log")
    _run_and_log([
        sys.executable, "-c",
        "import json,sys; "
        "info={'torch':None,'torch_cuda':None,'sglang':None,'vllm':None}; "
        "try:\n import torch; info['torch']=torch.__version__; info['torch_cuda']=getattr(torch.version,'cuda',None)\n"
        "except Exception as e:\n info['torch']=str(e)\n"
        "try:\n import sglang; info['sglang']=getattr(sglang,'__version__', 'unknown')\n"
        "except Exception as e:\n info['sglang']=str(e)\n"
        "try:\n import vllm; info['vllm']=getattr(vllm,'__version__','unknown')\n"
        "except Exception as e:\n info['vllm']=str(e)\n"
        "print(json.dumps(info))"
    ], "py_pkgs.log")
    _run_and_log(["bash", "-lc", "ldconfig -p | grep -E 'libcuda|libnvidia-ml' || true"], "ldconfig_cuda.log")

def _exists(path: str) -> bool:
    try:
        return Path(path).exists()
    except Exception:
        return False

def run_sanity():
    print("SANITY: running workload without profiler")
    rc = _run_and_log(["python", WORKLOAD], "sanity_workload.log")
    print(f"SANITY: rc={rc}")
    return rc

def run_nsys():
    print(f"NSYS: starting profile to {REMOTE_OUT} base={BASE} (this may take a while)")
    out_base = f"{REMOTE_OUT}/{BASE}"
    rep_ns = f"{out_base}.nsys-rep"
    rep_qd = f"{out_base}.qdrep"
    cmd = [
        "nsys", "profile",
        "--force-overwrite=true",
        "-t", "cuda,cublas,cudnn,nvtx",
        "--sample=none",
        "--cpuctxsw=none",
        "-o", out_base,
        "python", WORKLOAD,
    ]
    if USE_STRACE and shutil.which("strace"):
        cmd = ["strace", "-f", "-o", f"{DEBUG_DIR}/nsys.strace"] + cmd
    print(f"NSYS: cmd={' '.join(cmd)}")
    rc = _run_and_log(cmd, "nsys_profile.log")
    used_rep = rep_ns if _exists(rep_ns) else rep_qd if _exists(rep_qd) else None
    if used_rep:
        print(f"NSYS: profile complete -> {used_rep} (rc={rc})")
    else:
        raise SystemExit(f"NSYS: no report produced, rc={rc}")
    stats_cmd = [
        "nsys", "stats",
        "--report", "gpukernsum",
        "--format", "csv",
        "-o", f"{out_base}_kernsum",
        used_rep,
    ]
    print("NSYS: generating kernel summary CSV")
    rc_stats = _run_and_log(stats_cmd, "nsys_stats.log")
    if rc_stats == 0:
        print(f"NSYS: kernel summary written -> {out_base}_kernsum.csv")
    export_cmd = [
        "nsys", "export",
        "--sqlite", "true",
        "-o", out_base,
        used_rep,
    ]
    print(f"NSYS: exporting sqlite -> {out_base}.sqlite")
    _run_and_log(export_cmd, "nsys_export.log")

def run_ncu():
    print("NCU: starting kernel metrics collection (this may take a while)")
    out_base = f"{REMOTE_OUT}/{BASE}"
    cmd = [
        "ncu",
        "--target-processes", "all",
        "--set", "roofline",
        "--profile-from-start", "off",
        "--nvtx",
        "--nvtx-include", "warmup|prefill|decode",
        "--kernel-name-base", "demangled",
        "-o", out_base,
        "python", WORKLOAD,
    ]
    print(f"NCU: cmd={' '.join(cmd)}")
    rc = _run_and_log(cmd, "ncu_profile.log")
    ncu_rep = f"{out_base}.ncu-rep"
    if _exists(ncu_rep):
        print(f"NCU: metrics complete -> {ncu_rep} (rc={rc})")
    else:
        print(f"NCU: no report produced, rc={rc}")

if __name__ == "__main__":
    print(f"TRACE: REMOTE_OUT_DIR={REMOTE_OUT}")
    print(f"TRACE: WORKLOAD={WORKLOAD}")
    print(f"TRACE: TRACE_BASE={BASE}")
    print(f"TRACE: DEBUG_DIR={DEBUG_DIR}")
    ensure_dir(REMOTE_OUT)
    ensure_dir(DEBUG_DIR)
    os.environ.setdefault("CUDA_CACHE_PATH", f"{REMOTE_OUT}/compute_cache")
    os.environ.setdefault("CUDA_CACHE_MAXSIZE", "2147483648")
    ensure_dir(os.environ["CUDA_CACHE_PATH"])
    print(f"TRACE: CUDA_CACHE_PATH={os.environ['CUDA_CACHE_PATH']}")
    _gather_env_debug()
    if SANITY:
        run_sanity()
    nsys_ok = False
    try:
        run_nsys()
        nsys_ok = True
    except SystemExit as exc:
        print(f"TRACE: NSYS SystemExit: {exc}")
        out_base = f"{REMOTE_OUT}/{BASE}"
        rep_ns = f"{out_base}.nsys-rep"
        rep_qd = f"{out_base}.qdrep"
        if Path(rep_ns).exists() or Path(rep_qd).exists():
            print("TRACE: NSYS produced a report despite error; continuing")
            nsys_ok = True
        else:
            print("TRACE: NSYS produced no report; continuing anyway")
    except Exception as exc:
        print(f"TRACE: NSYS error: {exc}")
        out_base = f"{REMOTE_OUT}/{BASE}"
        rep_ns = f"{out_base}.nsys-rep"
        rep_qd = f"{out_base}.qdrep"
        if Path(rep_ns).exists() or Path(rep_qd).exists():
            print("TRACE: NSYS produced a report despite error; continuing")
            nsys_ok = True
        else:
            print("TRACE: NSYS produced no report; continuing anyway")
    try:
        run_ncu()
    except Exception as exc:
        print(f"TRACE: NCU error: {exc}; continuing")

