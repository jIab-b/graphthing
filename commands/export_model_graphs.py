import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import torch
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
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
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

MODEL_ID = os.environ.get("GRAPH_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


def _parse_device_map(raw: str | None):
    if not raw:
        return None
    txt = raw.strip()
    if txt.lower() == "auto":
        return "auto"
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return txt

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


def _snapshot_has_weights(local_dir: str) -> bool:
    return any(Path(local_dir).glob("*.safetensors"))


def _try_hf_cli_download(repo_id: str, local_dir: str) -> bool:
    hf_cli = shutil.which("huggingface-cli")
    if hf_cli is None:
        return False
    cmd = [
        hf_cli,
        "download",
        repo_id,
        "--local-dir",
        local_dir,
        "--local-dir-use-symlinks",
        "False",
        "--resume-download",
    ]
    include_patterns = [
        "*.safetensors",
        "*.json",
        "tokenizer*",
        "config.json",
        "generation_config.json",
    ]
    for pat in include_patterns:
        cmd.extend(["--include", pat])
    env = os.environ.copy()
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print(f"STAGE: huggingface-cli download -> {local_dir}")
    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except Exception as exc:
        print(f"STAGE: huggingface-cli failed ({exc}); will fall back")
        return False


def _ensure_local_snapshot(repo_id: str, local_root: str) -> str:
    model_alias = repo_id.split("/")[-1].replace("/", "--")
    local_dir = os.path.join(local_root, model_alias)
    force_refresh = _env_flag("GRAPH_FORCE_SNAPSHOT_REFRESH", False)
    os.makedirs(local_dir, exist_ok=True)
    if force_refresh:
        print(f"STAGE: forcing refresh of snapshot at {local_dir}")
        shutil.rmtree(local_dir, ignore_errors=True)
        os.makedirs(local_dir, exist_ok=True)

    if _snapshot_has_weights(local_dir):
        print(f"STAGE: detected existing snapshot at {local_dir}, reusing")
        return local_dir

    prefer_cli = _env_flag("GRAPH_USE_HF_CLI_DOWNLOAD", True)
    if prefer_cli and _try_hf_cli_download(repo_id, local_dir) and _snapshot_has_weights(local_dir):
        print("STAGE: snapshot ready via huggingface-cli")
        return local_dir

    if snapshot_download is not None:
        print(f"STAGE: snapshot_download fallback for {repo_id}")
        try:
            dl_t0 = time.time()
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*", "config.json", "generation_config.json"],
                ignore_patterns=["*.bin"],
            )
            print(f"STAGE: snapshot ready in {time.time() - dl_t0:.2f}s via python client")
        except Exception as exc:
            print(f"STAGE: snapshot_download failed ({exc}); continuing if files already exist")
    else:
        print("STAGE: huggingface_hub snapshot_download unavailable; assuming files exist locally")

    if not _snapshot_has_weights(local_dir):
        raise RuntimeError(f"Snapshot for {repo_id} missing in {local_dir}; download failed")
    return local_dir

def export_prefill(model, tok, bsz, tok_len, out_dir):
    x = torch.randint(0, tok.vocab_size, (bsz, tok_len), device="cuda", dtype=torch.long)
    m = torch.ones(bsz, tok_len, device="cuda", dtype=torch.long)
    ep = _safe_export(model, (x,), {"attention_mask": m, "use_cache": False, "return_dict": False})
    p = os.path.join(out_dir, f"prefill_B{bsz}_T{tok_len}.pt2")
    torch.export.save(ep, p)
    _maybe_write_dot(ep, "prefill", os.path.join(out_dir, f"prefill_B{bsz}_T{tok_len}.dot"))

def export_decode(model, tok, bsz, cache_len, out_dir):
    cfg = model.config
    nh = cfg.num_attention_heads
    nkvh = getattr(cfg, "num_key_value_heads", nh)
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
    p = os.path.join(out_dir, f"decode_B{bsz}_L{cache_len}.pt2")
    torch.export.save(ep, p)
    _maybe_write_dot(ep, "decode", os.path.join(out_dir, f"decode_B{bsz}_L{cache_len}.dot"))

if __name__ == "__main__":
    out_dir = os.environ.get("REMOTE_OUT_DIR", "/workspace/out_local")
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
    # Stage model snapshot to a local, non-volume path without symlinks
    repo_id = MODEL_ID
    local_root = os.environ.get("GRAPH_LOCAL_ROOT", "/tmp/hf_local")
    LOCAL_DIR = _ensure_local_snapshot(repo_id, local_root)

    # Offline, local-dir load to avoid remote checks and networked mmaps
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print("STEP 1: loading tokenizer from local dir")
    tok = AutoTokenizer.from_pretrained(LOCAL_DIR, use_fast=True, trust_remote_code=True)
    print("STEP 2: resolving config from local dir")
    cfg = AutoConfig.from_pretrained(LOCAL_DIR, trust_remote_code=True)
    print(f"CONFIG: loaded; layers={getattr(cfg,'num_hidden_layers',None)} heads={getattr(cfg,'num_attention_heads',None)} hidden={getattr(cfg,'hidden_size',None)}")
    print("STEP 3: loading model weights on CPU from local dir (single GPU move after)")
    t_load = time.time()
    device_map = _parse_device_map(os.environ.get("GRAPH_DEVICE_MAP"))
    low_cpu_mem = _env_flag("GRAPH_LOW_CPU_MEM_USAGE", True)
    model = AutoModelForCausalLM.from_pretrained(
        LOCAL_DIR,
        torch_dtype="auto",
        low_cpu_mem_usage=low_cpu_mem,
        trust_remote_code=True,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    print(f"LOAD: from_pretrained completed in {time.time() - t_load:.2f}s (CPU)")
    should_move_to_cuda = device_map is None and torch.cuda.is_available()
    if should_move_to_cuda:
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"CUDA: before move free={free_b} total={total_b}")
        print("STEP 4: moving model to CUDA (single bulk transfer)")
        t_mv = time.time()
        model.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        free_b2, total_b2 = torch.cuda.mem_get_info()
        print(f"CUDA: after move free={free_b2} total={total_b2} (move {time.time() - t_mv:.2f}s)")
    elif device_map is not None:
        print(f"STEP 4: device_map={device_map} handled placement; skipping manual cuda move")
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
    export_prefill(model, tok, 1, 512, out_dir)
    export_decode(model, tok, 1, 512, out_dir)
