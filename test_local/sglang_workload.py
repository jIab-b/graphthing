import os
import time
import json
import sglang as sgl
import torch
from transformers import AutoConfig

HF_HOME_PATH = os.path.expanduser(os.environ.get("HF_HOME", "~/hf"))
os.environ["HF_HOME"] = HF_HOME_PATH
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", HF_HOME_PATH)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_ID = os.environ.get("GRAPH_MODEL_ID", "Qwen/Qwen3-0.6B")

@sgl.function
def gen_text(s, text, tokens, temperature):
    s += text
    s += sgl.gen("out", max_tokens=tokens, temperature=temperature)

def text_of_len(n):
    return " ".join(["word"] * n)

def hw_profile():
    d = {}
    d["device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        i = 0
        props = torch.cuda.get_device_properties(i)
        d["name"] = torch.cuda.get_device_name(i)
        d["cc"] = f"{props.major}.{props.minor}"
        d["total_mem"] = int(props.total_memory)
        d["sm_count"] = int(props.multi_processor_count)
    return d

def kv_bytes_per_token(cfg, dtype_bytes=2):
    nh = int(getattr(cfg, "num_attention_heads"))
    nkvh = int(getattr(cfg, "num_key_value_heads", nh))
    layers = int(getattr(cfg, "num_hidden_layers"))
    head_dim = int(getattr(cfg, "hidden_size")) // nh
    return 2 * layers * nkvh * head_dim * dtype_bytes

if __name__ == "__main__":
    OUT_DIR = os.environ.get("REMOTE_OUT_DIR", os.path.join(BASE_DIR, "out_local"))
    os.makedirs(OUT_DIR, exist_ok=True)
    runtime = sgl.Runtime(model_path=MODEL_ID)
    sgl.set_default_backend(runtime)
    torch.cuda.nvtx.range_push("warmup")
    t0 = time.time()
    gen_text.run(text_of_len(64), 8, 0.0)
    warmup_time = time.time() - t0
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("prefill_B1_T1024")
    t1 = time.time()
    gen_text.run(text_of_len(1024), 8, 0.0)
    prefill_time = time.time() - t1
    torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_push("decode_B1_T128")
    t2 = time.time()
    gen_text.run(text_of_len(32), 128, 0.0)
    decode_time = time.time() - t2
    torch.cuda.nvtx.range_pop()
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    kv_bpt = kv_bytes_per_token(cfg, 2)
    mem_free, mem_total = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
    summary = {
        "model_id": MODEL_ID,
        "hardware": hw_profile(),
        "memory": {
            "mem_free": int(mem_free),
            "mem_total": int(mem_total),
            "allocated": int(torch.cuda.memory_allocated() if torch.cuda.is_available() else 0),
            "reserved": int(torch.cuda.memory_reserved() if torch.cuda.is_available() else 0),
        },
        "config": {
            "num_layers": int(getattr(cfg, "num_hidden_layers")),
            "num_heads": int(getattr(cfg, "num_attention_heads")),
            "num_kv_heads": int(getattr(cfg, "num_key_value_heads", getattr(cfg, "num_attention_heads"))),
            "hidden_size": int(getattr(cfg, "hidden_size")),
            "kv_bytes_per_token_fp16": int(kv_bpt),
        },
        "workload": {
            "warmup": {"prompt_len": 64, "tokens": 8, "time_s": warmup_time},
            "prefill": {"prompt_len": 1024, "tokens": 8, "time_s": prefill_time, "tokens_per_s": 8.0 / max(prefill_time, 1e-6)},
            "decode": {"prompt_len": 32, "tokens": 128, "time_s": decode_time, "tokens_per_s": 128.0 / max(decode_time, 1e-6)},
        },
        "nvtx": ["warmup", "prefill_B1_T1024", "decode_B1_T128"],
    }
    with open(os.path.join(OUT_DIR, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary))

