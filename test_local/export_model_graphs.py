import json
import os
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import masking_utils as hf_masking_utils
from transformers.cache_utils import DynamicCache
from transformers.utils import logging as hf_logging
try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None
try:
    import safetensors.torch as st
except Exception:
    st = None

HF_HOME_PATH = os.path.expanduser(os.environ.get("HF_HOME", "~/hf"))
os.environ["HF_HOME"] = HF_HOME_PATH
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME_PATH)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ID = os.environ.get("GRAPH_MODEL_ID", "Qwen/Qwen3-0.6B")
PREFILL_SCENARIOS = [
    {"batch_size": 1, "seq_len": 512},
    {"batch_size": 2, "seq_len": 512},
    {"batch_size": 1, "seq_len": 1024},
]
DECODE_SCENARIOS = [
    {"batch_size": 1, "cache_len": 256},
    {"batch_size": 1, "cache_len": 512},
    {"batch_size": 2, "cache_len": 256},
]
RUNTIME_POLICY = {
    "dynamic_batch_window_ms": 2,
    "speculative_decode": {"enabled": True, "draft_tokens": 2},
    "cuda_graph_buckets": {
        "prefill": [{"batch_size": s["batch_size"], "seq_len": s["seq_len"]} for s in PREFILL_SCENARIOS],
        "decode": [{"batch_size": s["batch_size"], "cache_len": s["cache_len"]} for s in DECODE_SCENARIOS],
    },
    "kv_cache": {
        "policy": "paged",
        "page_size_tokens": 256,
        "eviction": "least_recent",
    },
}


def _simple_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids=None, **_):
    batch_size, q_len = input_embeds.shape[:2]
    device = input_embeds.device
    dtype = input_embeds.dtype
    if past_key_values is None:
        past_len = 0
    elif hasattr(past_key_values, "get_seq_length"):
        past_len = past_key_values.get_seq_length()
    else:
        past_len = int(cache_position[0].item())
    kv_len = past_len + q_len
    neg_value = torch.finfo(dtype).min
    q_pos = cache_position.view(1, q_len, 1).to(device)
    kv_pos = torch.arange(kv_len, device=device).view(1, 1, kv_len)
    future_mask = kv_pos > q_pos
    mask = torch.zeros((batch_size, 1, q_len, kv_len), device=device, dtype=dtype)
    mask = mask.masked_fill(future_mask, neg_value)
    if attention_mask is not None and attention_mask.ndim == 2:
        padding = attention_mask.to(device=device, dtype=torch.bool)
        padding = ~padding[:, -kv_len:]
        mask = mask.masked_fill(padding[:, None, None, :], neg_value)
    return mask

hf_masking_utils.create_causal_mask = _simple_causal_mask

def _reset(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def _bytes_per_dtype(dtype: torch.dtype) -> int:
    return torch.tensor((), dtype=dtype).element_size()

def _kv_bytes_per_token(cfg, dtype: torch.dtype) -> int:
    nkvh = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    return 2 * cfg.num_hidden_layers * nkvh * head_dim * _bytes_per_dtype(dtype)

def _gpu_snapshot():
    if not torch.cuda.is_available():
        return {"available": False}
    props = torch.cuda.get_device_properties(0)
    free_b, total_b = torch.cuda.mem_get_info()
    approx_bw_gbps = None
    mem_clock = getattr(props, "memory_clock_rate", 0)
    bus_width = getattr(props, "memory_bus_width", 0)
    if mem_clock and bus_width:
        approx_bw_gbps = 2 * mem_clock * 1e3 * (bus_width / 8) / 1e9
    return {
        "available": True,
        "name": props.name,
        "sm_count": props.multi_processor_count,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_memory_bytes": total_b,
        "free_memory_bytes": free_b,
        "shared_mem_per_block": props.shared_memory_per_block,
        "l2_cache_bytes": getattr(props, "l2_cache_size", None),
        "memory_bandwidth_gbps_est": approx_bw_gbps,
        "driver": torch.version.cuda,
    }

def _manifest_header(cfg, model_dtype, out_dir):
    gpu = _gpu_snapshot()
    kv_bpt = _kv_bytes_per_token(cfg, model_dtype)
    return {
        "model_id": MODEL_ID,
        "local_snapshot": os.path.abspath(out_dir),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "model_config": {
            "layers": getattr(cfg, "num_hidden_layers", None),
            "attention_heads": getattr(cfg, "num_attention_heads", None),
            "key_value_heads": getattr(cfg, "num_key_value_heads", None),
            "head_dim": getattr(cfg, "head_dim", None),
            "hidden_size": getattr(cfg, "hidden_size", None),
            "dtype": str(model_dtype),
        },
        "hardware": gpu,
        "kv_cache": {
            "bytes_per_token": kv_bpt,
            "dtype": str(model_dtype),
            "notes": "kv_bytes = 2 * layers * kv_heads * head_dim * dtype_bytes",
        },
        "runtime_policy": RUNTIME_POLICY,
        "precision_options": {
            "weights": ["fp16"],
            "activations": ["fp16"],
            "kv_cache": ["fp16", "int8"],
        },
        "quant_calibration": {
            "status": "not_collected",
            "notes": "int8 scales to be produced via calibration script before enabling low-precision path",
        },
        "profiling": {
            "nsight_script": os.path.abspath(os.path.join(BASE_DIR, "trace_with_nsight.py")),
            "instructions": "Activate venv, set WORKLOAD env to serving entrypoint, then run trace_with_nsight.py for NSYS/NCU traces.",
        },
        "scenarios": [],
    }

def _decorate_entry(entry, cfg, model_dtype):
    kv_bpt = _kv_bytes_per_token(cfg, model_dtype)
    seq = entry.get("sequence_length") or 0
    cache_len = entry.get("cache_length") or 0
    if entry["phase"] == "prefill":
        produced = kv_bpt * entry["batch_size"] * seq
        consumed = 0
    else:
        produced = kv_bpt * entry["batch_size"]
        consumed = kv_bpt * entry["batch_size"] * cache_len
    entry["kv_cache_bytes"] = {
        "consumed": consumed,
        "produced": produced,
    }
    entry["attention"] = {
        "implementation": "sdpa_eager",
        "mask": "causal",
        "rope_theta": getattr(cfg, "rope_theta", None),
        "rope_scaling": getattr(cfg, "rope_scaling", None),
    }
    entry["runtime_tags"] = {
        "cuda_graph_bucket": True,
        "serving_priority": "throughput" if entry["phase"] == "prefill" else "latency",
    }
    return entry

def _safe_export(model, args, kwargs):
    try:
        return torch.export.export(model, args, kwargs=kwargs)
    except AssertionError as exc:
        if "ProxyTorchDispatchMode" in str(exc) and "attention_mask" in kwargs:
            nk = {k: v for k, v in kwargs.items() if k != "attention_mask"}
            return torch.export.export(model, args, kwargs=nk)
        raise

def _maybe_write_dot(ep, label, path):
    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".dot"
        path = base + ext
    drawer = FxGraphDrawer(ep.graph_module, label)
    dg = drawer.get_main_dot_graph()
    writer = getattr(dg, f"write_{ext.lstrip('.').lower()}", None)
    if writer is not None:
        try:
            writer(path)
            return
        except Exception:
            pass
    try:
        txt = dg.to_string()
    except Exception:
        txt = dg.create_dot()
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def export_prefill(model, tok, bsz, tok_len, out_dir):
    x = torch.randint(0, tok.vocab_size, (bsz, tok_len), device="cuda", dtype=torch.long)
    m = torch.ones(bsz, tok_len, device="cuda", dtype=torch.long)
    ep = _safe_export(model, (x,), {"attention_mask": m, "use_cache": False, "return_dict": False})
    base_name = f"prefill_B{bsz}_T{tok_len}"
    p = os.path.join(out_dir, f"{base_name}.pt2")
    torch.export.save(ep, p)
    dot_path = os.path.join(out_dir, f"{base_name}.dot")
    _maybe_write_dot(ep, "prefill", dot_path)
    return {
        "id": base_name,
        "phase": "prefill",
        "batch_size": bsz,
        "sequence_length": tok_len,
        "cache_length": None,
        "artifacts": {
            "torch_export": os.path.abspath(p),
            "graphviz": os.path.abspath(dot_path),
        },
        "inputs": [
            {"name": "input_ids", "shape": [bsz, tok_len], "dtype": "torch.long"},
            {"name": "attention_mask", "shape": [bsz, tok_len], "dtype": "torch.long"},
        ],
        "outputs": [
            {"name": "logits", "shape": [bsz, tok_len, model.config.hidden_size], "dtype": str(model.dtype)},
        ],
        "notes": ["prefill path", "no kv cache inputs"],
    }

def export_decode(model, tok, bsz, cache_len, out_dir):
    cfg = model.config
    nh = cfg.num_attention_heads
    nkvh = getattr(cfg, "num_key_value_heads", nh)
    hd = getattr(cfg, "head_dim", None)
    if hd is None:
        hd = cfg.hidden_size // nh
    x = torch.randint(0, tok.vocab_size, (bsz, 1), device="cuda", dtype=torch.long)
    mask = torch.ones(bsz, cache_len + 1, device="cuda", dtype=torch.long)
    pos = torch.arange(cache_len, cache_len + 1, device="cuda", dtype=torch.long).unsqueeze(0).expand(bsz, 1)
    pkv = []
    for _ in range(cfg.num_hidden_layers):
        k = torch.randn(bsz, nkvh, cache_len, hd, device="cuda", dtype=model.dtype)
        v = torch.randn(bsz, nkvh, cache_len, hd, device="cuda", dtype=model.dtype)
        pkv.append((k, v))
    pkv = tuple(pkv)
    class DecodeWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, input_ids, attn_mask, cached_kv, position_ids):
            cache = DynamicCache(config=self.m.config)
            for li, (lk, lv) in enumerate(cached_kv):
                cache.update(lk, lv, li)
            y = self.m(input_ids, attention_mask=attn_mask, past_key_values=cache, use_cache=True, position_ids=position_ids, return_dict=False)
            return y[0]
    ep = _safe_export(DecodeWrapper(model), (x, mask, pkv, pos), {})
    base_name = f"decode_B{bsz}_L{cache_len}"
    p = os.path.join(out_dir, f"{base_name}.pt2")
    torch.export.save(ep, p)
    dot_path = os.path.join(out_dir, f"{base_name}.dot")
    _maybe_write_dot(ep, "decode", dot_path)
    return {
        "id": base_name,
        "phase": "decode",
        "batch_size": bsz,
        "sequence_length": 1,
        "cache_length": cache_len,
        "artifacts": {
            "torch_export": os.path.abspath(p),
            "graphviz": os.path.abspath(dot_path),
        },
        "inputs": [
            {"name": "input_ids", "shape": [bsz, 1], "dtype": "torch.long"},
            {"name": "attention_mask", "shape": [bsz, cache_len + 1], "dtype": "torch.long"},
            {"name": "cached_kv", "shape": [cfg.num_hidden_layers, 2, bsz, nkvh, cache_len, hd], "dtype": str(model.dtype)},
            {"name": "position_ids", "shape": [bsz, 1], "dtype": "torch.long"},
        ],
        "outputs": [
            {"name": "logits", "shape": [bsz, 1, model.config.hidden_size], "dtype": str(model.dtype)},
        ],
        "notes": ["decode path", "kv cache provided explicitly"],
    }

if __name__ == "__main__":
    out_dir = os.environ.get("REMOTE_OUT_DIR", os.path.join(BASE_DIR, "out_local"))
    _reset(out_dir)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "debug")
    hf_logging.set_verbosity_debug()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
    print("CUDA WARMUP: creating CUDA context")
    try:
        _tmp = torch.empty(1, device="cuda")
        torch.cuda.synchronize()
        print("CUDA WARMUP: ready")
    except Exception as exc:
        print(f"CUDA WARMUP: failed: {exc}")
    if st is not None and not hasattr(st, "_orig_load_file"):
        st._orig_load_file = st.load_file
        def _dbg_load_file(path, *args, **kwargs):
            t0 = time.time()
            print(f"SAFETENSORS: loading {path}")
            sd = st._orig_load_file(path, *args, **kwargs)
            dt = time.time() - t0
            try:
                k = len(sd)
            except Exception:
                k = -1
            print(f"SAFETENSORS: loaded {path} in {dt:.2f}s keys={k}")
            return sd
        st.load_file = _dbg_load_file
    repo_id = MODEL_ID
    local_root = os.environ.get("GRAPH_LOCAL_ROOT", os.path.join(BASE_DIR, "hf_local"))
    model_alias = repo_id.split("/")[-1].replace("/", "--")
    LOCAL_DIR = os.path.join(local_root, model_alias)
    os.makedirs(LOCAL_DIR, exist_ok=True)
    print(f"STAGE: ensuring local snapshot at {LOCAL_DIR}")
    if snapshot_download is not None:
        try:
            dl_t0 = time.time()
            snapshot_download(
                repo_id=repo_id,
                local_dir=LOCAL_DIR,
                local_dir_use_symlinks=False,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "config.json", "generation_config.json"],
                ignore_patterns=["*.bin"],
            )
            print(f"STAGE: snapshot ready in {time.time() - dl_t0:.2f}s")
        except Exception as exc:
            print(f"STAGE: snapshot_download failed ({exc}), proceeding if files already present")
    else:
        print("STAGE: huggingface_hub not available, assuming files present locally")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("STEP 1: loading tokenizer from local dir")
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR, use_fast=True, trust_remote_code=True)
    print("STEP 2: resolving config from local dir")
    cfg = AutoConfig.from_pretrained(LOCAL_DIR, trust_remote_code=True)
    print(f"CONFIG: loaded; layers={getattr(cfg,'num_hidden_layers',None)} heads={getattr(cfg,'num_attention_heads',None)} hidden={getattr(cfg,'hidden_size',None)}")
    print("STEP 3: loading model weights on CPU from local dir (single GPU move after)")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    print(f"LOAD: from_pretrained completed in {time.time() - t_load:.2f}s (CPU)")
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"CUDA: before move free={free_b} total={total_b}")
    print("STEP 4: moving model to CUDA (single bulk transfer)")
    t_mv = time.time()
    model.to("cuda", non_blocking=True)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free_b2, total_b2 = torch.cuda.mem_get_info()
        print(f"CUDA: after move free={free_b2} total={total_b2} (move {time.time() - t_mv:.2f}s)")
    print("STEP 5: finalizing attention implementation")
    print("STEP 6: model ready")
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    print("VERIFY: running tiny forward to confirm readiness")
    try:
        tiny = torch.randint(0, tok.vocab_size, (1, 2), device="cuda", dtype=torch.long)
        out = model(tiny, use_cache=False, return_dict=False)
        _ = out[0]
        print("VERIFY: forward ok")
    except Exception as exc:
        print(f"VERIFY: forward failed: {exc}")
    manifest = _manifest_header(cfg, model.dtype, LOCAL_DIR)
    scenarios = []
    for sc in PREFILL_SCENARIOS:
        meta = export_prefill(model, tok, sc["batch_size"], sc["seq_len"], out_dir)
        meta = _decorate_entry(meta, cfg, model.dtype)
        scenarios.append(meta)
    for sc in DECODE_SCENARIOS:
        meta = export_decode(model, tok, sc["batch_size"], sc["cache_len"], out_dir)
        meta = _decorate_entry(meta, cfg, model.dtype)
        scenarios.append(meta)
    manifest["scenarios"] = scenarios
    manifest_path = os.path.join(out_dir, "export_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"MANIFEST: wrote metadata -> {manifest_path}")
