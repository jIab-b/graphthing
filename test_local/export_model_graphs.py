import json
import os
import time
import shutil
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

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


def _write_json(path: str, payload: Dict[str, Any]):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _normalize_dim(dim):
    try:
        return int(dim)
    except (TypeError, ValueError):
        return str(dim)


def _shape_list(shape) -> List[Any]:
    if shape is None:
        return None
    dims: List[Any] = []
    for dim in shape:
        dims.append(_normalize_dim(dim))
    return dims


def _estimate_tensor_bytes(shape, dtype) -> Optional[int]:
    if not shape or dtype is None:
        return None
    numel = 1
    for dim in shape:
        if not isinstance(dim, int):
            return None
        numel *= max(dim, 0)
    try:
        elem_size = torch.tensor((), dtype=dtype).element_size()
    except Exception:
        return None
    return numel * elem_size


def _node_role(node) -> str:
    if node.op == "placeholder":
        return "input"
    if node.op == "output":
        return "output"
    return "intermediate"


def _tensor_inventory(ep, label: str, scenario_id: str) -> Dict[str, Any]:
    gm = ep.graph_module
    tensors: List[Dict[str, Any]] = []
    total_bytes = 0
    for node in gm.graph.nodes:
        meta = node.meta.get("tensor_meta") or node.meta.get("val")
        if meta is None:
            continue
        shape = _shape_list(getattr(meta, "shape", None))
        dtype = getattr(meta, "dtype", None)
        estimated_bytes = _estimate_tensor_bytes(shape, dtype)
        if isinstance(estimated_bytes, int):
            total_bytes += estimated_bytes
        tensors.append(
            {
                "node": node.name,
                "op": node.op,
                "target": str(node.target),
                "role": _node_role(node),
                "shape": shape,
                "dtype": str(dtype) if dtype is not None else None,
                "estimated_bytes": estimated_bytes,
            }
        )
    return {
        "graph_label": label,
        "scenario_id": scenario_id,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "tensors": tensors,
        "summary": {
            "total_entries": len(tensors),
            "estimated_tensor_bytes": total_bytes,
        },
    }


def _persist_tensor_inventory(ep, base_name: str, label: str, out_dir: str) -> str:
    inv = _tensor_inventory(ep, label, base_name)
    path = os.path.join(out_dir, f"{base_name}_{label}_tensors.json")
    _write_json(path, inv)
    return os.path.abspath(path)


def _preferred_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _benchmark_callable(run_fn, device: str, warmup: int = 2, iters: int = 5) -> Dict[str, Any]:
    latencies: List[float] = []
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(warmup):
            run_fn()
        torch.cuda.synchronize()
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_fn()
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        peak_mem = torch.cuda.max_memory_allocated()
    else:
        for _ in range(warmup):
            run_fn()
        for _ in range(iters):
            t0 = time.perf_counter()
            run_fn()
            latencies.append((time.perf_counter() - t0) * 1000.0)
        peak_mem = None
    if not latencies:
        stats = {"avg": None, "min": None, "max": None}
    else:
        stats = {"avg": mean(latencies), "min": min(latencies), "max": max(latencies)}
    return {
        "device": device,
        "warmup_iters": warmup,
        "sample_iters": iters,
        "latency_ms": stats,
        "peak_memory_bytes": peak_mem,
        "samples": latencies,
    }


def _measure_prefill(model, scenario_meta, x, attn_mask, kv_bpt, device):
    def _run():
        with torch.inference_mode():
            model(x, attention_mask=attn_mask, use_cache=False, return_dict=False)
    stats = _benchmark_callable(_run, device)
    produced = kv_bpt * scenario_meta["batch_size"] * scenario_meta["sequence_length"]
    stats.update(
        {
            "phase": "prefill",
            "scenario": scenario_meta,
            "kv_cache": {
                "bytes_per_token": kv_bpt,
                "produced_bytes": produced,
                "consumed_bytes": 0,
            },
            "environment": {
                "cuda_available": torch.cuda.is_available(),
                "torch": torch.__version__,
            },
        }
    )
    return stats


def _measure_decode(module, scenario_meta, x, attn_mask, cached_kv, position_ids, kv_bpt, device):
    def _run():
        with torch.inference_mode():
            module(x, attn_mask, cached_kv, position_ids)
    stats = _benchmark_callable(_run, device)
    produced = kv_bpt * scenario_meta["batch_size"]
    consumed = kv_bpt * scenario_meta["batch_size"] * scenario_meta["cache_length"]
    stats.update(
        {
            "phase": "decode",
            "scenario": scenario_meta,
            "kv_cache": {
                "bytes_per_token": kv_bpt,
                "produced_bytes": produced,
                "consumed_bytes": consumed,
            },
            "environment": {
                "cuda_available": torch.cuda.is_available(),
                "torch": torch.__version__,
            },
        }
    )
    return stats


def _build_full_inference_plan(prefill_entries, decode_entries, out_dir):
    bundles = []
    decode_by_batch: Dict[int, List[Dict[str, Any]]] = {}
    for dec in decode_entries:
        decode_by_batch.setdefault(dec["batch_size"], []).append(dec)
    for pf in prefill_entries:
        matches = decode_by_batch.get(pf["batch_size"], [])
        for dec in matches:
            kv_info = pf.get("external_state", {}).get("kv_cache", {})
            bundles.append(
                {
                    "id": f"{pf['id']}__{dec['id']}",
                    "sequence": [
                        {
                            "stage": "prefill",
                            "graph": pf["artifacts"]["torch_export"],
                            "tensor_inventory": pf["artifacts"].get("tensor_inventory"),
                            "metrics": pf["artifacts"].get("metrics"),
                        },
                        {
                            "stage": "decode",
                            "graph": dec["artifacts"]["torch_export"],
                            "tensor_inventory": dec["artifacts"].get("tensor_inventory"),
                            "metrics": dec["artifacts"].get("metrics"),
                        },
                    ],
                    "kv_cache_contract": {
                        "bytes_per_token": kv_info.get("bytes_per_token"),
                        "dtype": kv_info.get("dtype"),
                        "layers": kv_info.get("layers"),
                        "producer_scenario": pf["id"],
                        "consumer_scenario": dec["id"],
                        "cache_length_tokens": dec.get("cache_length"),
                        "flow_notes": "Prefill produces KV cache consumed/extended by decode",
                    },
                    "serving_notes": [
                        "Execute prefill once per request, then reuse cache for decode tokens",
                    ],
                }
            )
    plan = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "bundle_count": len(bundles),
        "bundles": bundles,
    }
    path = os.path.join(out_dir, "full_inference_plan.json")
    _write_json(path, plan)
    return os.path.abspath(path)


def _build_runtime_context(prefill_entries, decode_entries, gpu_snapshot):
    shape_buckets = {
        "prefill": [
            {
                "scenario_id": ent["id"],
                "batch_size": ent["batch_size"],
                "sequence_length": ent["sequence_length"],
            }
            for ent in prefill_entries
        ],
        "decode": [
            {
                "scenario_id": ent["id"],
                "batch_size": ent["batch_size"],
                "sequence_length": ent["sequence_length"],
                "cache_length": ent.get("cache_length"),
            }
            for ent in decode_entries
        ],
    }
    bucket_ids = [ent["id"] for ent in prefill_entries + decode_entries]
    ctx = {
        "shape_buckets": shape_buckets,
        "cuda_graphs": {
            "enabled": True,
            "bucket_ids": bucket_ids,
            "notes": "Static shapes exported for CUDA graph capture/replay",
        },
        "allocator": {
            "strategy": "caching",
            "reserved_bytes_cap": (gpu_snapshot or {}).get("total_memory_bytes"),
            "notes": "Rely on PyTorch CUDACachingAllocator to reuse fixed-shape pools",
        },
        "kv_cache_manager": {
            **RUNTIME_POLICY["kv_cache"],
            "bytes_per_token": (prefill_entries[0]["external_state"]["kv_cache"]["bytes_per_token"] if prefill_entries else None),
        },
        "serving": {
            "dynamic_batch_window_ms": RUNTIME_POLICY["dynamic_batch_window_ms"],
            "speculative_decode": RUNTIME_POLICY["speculative_decode"],
        },
    }
    return ctx

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
    entry["external_state"] = {
        "kv_cache": {
            "bytes_per_token": kv_bpt,
            "dtype": str(model_dtype),
            "layers": getattr(cfg, "num_hidden_layers", None),
            "policy": "produce" if entry["phase"] == "prefill" else "consume-update",
        }
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

def export_prefill(model, tok, bsz, tok_len, out_dir, device: str):
    cfg = model.config
    kv_bpt = _kv_bytes_per_token(cfg, model.dtype)
    x = torch.randint(0, tok.vocab_size, (bsz, tok_len), device=device, dtype=torch.long)
    m = torch.ones(bsz, tok_len, device=device, dtype=torch.long)
    ep = _safe_export(model, (x,), {"attention_mask": m, "use_cache": False, "return_dict": False})
    base_name = f"prefill_B{bsz}_T{tok_len}"
    p = os.path.join(out_dir, f"{base_name}.pt2")
    torch.export.save(ep, p)
    dot_path = os.path.join(out_dir, f"{base_name}.dot")
    _maybe_write_dot(ep, "prefill", dot_path)
    tensor_inventory_path = _persist_tensor_inventory(ep, base_name, "prefill", out_dir)
    scenario_meta = {"id": base_name, "batch_size": bsz, "sequence_length": tok_len}
    metrics = _measure_prefill(model, scenario_meta, x, m, kv_bpt, device)
    metrics_path = os.path.join(out_dir, f"{base_name}_metrics.json")
    _write_json(metrics_path, metrics)
    return {
        "id": base_name,
        "phase": "prefill",
        "batch_size": bsz,
        "sequence_length": tok_len,
        "cache_length": None,
        "artifacts": {
            "torch_export": os.path.abspath(p),
            "graphviz": os.path.abspath(dot_path),
            "tensor_inventory": tensor_inventory_path,
            "metrics": os.path.abspath(metrics_path),
        },
        "inputs": [
            {"name": "input_ids", "shape": [bsz, tok_len], "dtype": "torch.long"},
            {"name": "attention_mask", "shape": [bsz, tok_len], "dtype": "torch.long"},
        ],
        "outputs": [
            {"name": "logits", "shape": [bsz, tok_len, model.config.hidden_size], "dtype": str(model.dtype)},
        ],
        "measurements": metrics,
        "notes": ["prefill path", "no kv cache inputs"],
    }

def export_decode(model, tok, bsz, cache_len, out_dir, device: str):
    cfg = model.config
    kv_bpt = _kv_bytes_per_token(cfg, model.dtype)
    nh = cfg.num_attention_heads
    nkvh = getattr(cfg, "num_key_value_heads", nh)
    hd = getattr(cfg, "head_dim", None)
    if hd is None:
        hd = cfg.hidden_size // nh
    x = torch.randint(0, tok.vocab_size, (bsz, 1), device=device, dtype=torch.long)
    mask = torch.ones(bsz, cache_len + 1, device=device, dtype=torch.long)
    pos = torch.arange(cache_len, cache_len + 1, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, 1)
    pkv = []
    for _ in range(cfg.num_hidden_layers):
        k = torch.randn(bsz, nkvh, cache_len, hd, device=device, dtype=model.dtype)
        v = torch.randn(bsz, nkvh, cache_len, hd, device=device, dtype=model.dtype)
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
            y = self.m(
                input_ids,
                attention_mask=attn_mask,
                past_key_values=cache,
                use_cache=True,
                position_ids=position_ids,
                return_dict=False,
            )
            return y[0]

    decode_module = DecodeWrapper(model)
    ep = _safe_export(decode_module, (x, mask, pkv, pos), {})
    base_name = f"decode_B{bsz}_L{cache_len}"
    p = os.path.join(out_dir, f"{base_name}.pt2")
    torch.export.save(ep, p)
    dot_path = os.path.join(out_dir, f"{base_name}.dot")
    _maybe_write_dot(ep, "decode", dot_path)
    tensor_inventory_path = _persist_tensor_inventory(ep, base_name, "decode", out_dir)
    scenario_meta = {
        "id": base_name,
        "batch_size": bsz,
        "sequence_length": 1,
        "cache_length": cache_len,
    }
    metrics = _measure_decode(decode_module, scenario_meta, x, mask, pkv, pos, kv_bpt, device)
    metrics_path = os.path.join(out_dir, f"{base_name}_metrics.json")
    _write_json(metrics_path, metrics)
    return {
        "id": base_name,
        "phase": "decode",
        "batch_size": bsz,
        "sequence_length": 1,
        "cache_length": cache_len,
        "artifacts": {
            "torch_export": os.path.abspath(p),
            "graphviz": os.path.abspath(dot_path),
            "tensor_inventory": tensor_inventory_path,
            "metrics": os.path.abspath(metrics_path),
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
        "measurements": metrics,
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
    print("STEP 3: loading model weights on CPU from local dir {LOCAL_DIR}, (single GPU move after)")
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        torch_dtype="auto",
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    print(f"LOAD: from_pretrained completed in {time.time() - t_load:.2f}s (CPU)")
    target_device = _preferred_device()
    print(f"DEVICE: selected runtime device -> {target_device}")
    using_cuda = target_device == "cuda"
    if using_cuda:
        try:
            if torch.cuda.is_available():
                free_b, total_b = torch.cuda.mem_get_info()
                print(f"CUDA: before move free={free_b} total={total_b}")
            print("STEP 4: moving model to CUDA (single bulk transfer)")
            t_mv = time.time()
            model.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            free_b2, total_b2 = torch.cuda.mem_get_info()
            print(f"CUDA: after move free={free_b2} total={total_b2} (move {time.time() - t_mv:.2f}s)")
        except Exception as exc:
            print(f"DEVICE: CUDA move failed, falling back to CPU ({exc})")
            target_device = "cpu"
            using_cuda = False
            model.to("cpu")
    if not using_cuda:
        print("STEP 4: using CPU for exports")
        model.to("cpu")
    print("STEP 5: finalizing attention implementation")
    print("STEP 6: model ready")
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()
    print("VERIFY: running tiny forward to confirm readiness")
    try:
        tiny = torch.randint(0, tok.vocab_size, (1, 2), device=target_device, dtype=torch.long)
        out = model(tiny, use_cache=False, return_dict=False)
        _ = out[0]
        print("VERIFY: forward ok")
    except Exception as exc:
        print(f"VERIFY: forward failed: {exc}")
    manifest = _manifest_header(cfg, model.dtype, LOCAL_DIR)
    scenarios: List[Dict[str, Any]] = []
    prefill_entries: List[Dict[str, Any]] = []
    decode_entries: List[Dict[str, Any]] = []
    for sc in PREFILL_SCENARIOS:
        meta = export_prefill(model, tok, sc["batch_size"], sc["seq_len"], out_dir, target_device)
        meta = _decorate_entry(meta, cfg, model.dtype)
        scenarios.append(meta)
        prefill_entries.append(meta)
    for sc in DECODE_SCENARIOS:
        meta = export_decode(model, tok, sc["batch_size"], sc["cache_len"], out_dir, target_device)
        meta = _decorate_entry(meta, cfg, model.dtype)
        scenarios.append(meta)
        decode_entries.append(meta)
    manifest["scenarios"] = scenarios
    manifest["runtime_context"] = _build_runtime_context(prefill_entries, decode_entries, manifest.get("hardware"))
    plan_path = _build_full_inference_plan(prefill_entries, decode_entries, out_dir)
    manifest["full_inference_plan"] = plan_path
    manifest_path = os.path.join(out_dir, "export_manifest.json")
    _write_json(manifest_path, manifest)
    print(f"MANIFEST: wrote metadata -> {manifest_path}")
