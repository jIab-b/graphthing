import os
import time
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import masking_utils as hf_masking_utils
from transformers.cache_utils import DynamicCache
from transformers.utils import logging as hf_logging
try:
    import safetensors.torch as st
except Exception:
    st = None

MODEL_ID = os.environ.get("GRAPH_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

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
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "debug")
    hf_logging.set_verbosity_debug()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()
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
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True, trust_remote_code=True)
    AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("STEP 1: resolving config")
    cfg = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True)
    print(f"CONFIG: loaded; layers={getattr(cfg,'num_hidden_layers',None)} heads={getattr(cfg,'num_attention_heads',None)} hidden={getattr(cfg,'hidden_size',None)}")
    print("STEP 2: loading model weights on CPU (no device_map)")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_safetensors=True,
        trust_remote_code=True,
        low_cpu_mem_usage=False,
    )
    print("LOAD: from_pretrained completed")
    print("LOAD: weights materialized on CPU")
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"CUDA: before move free={free_b} total={total_b}")
    print("STEP 3: moving model to CUDA")
    model.to("cuda")
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        free_b2, total_b2 = torch.cuda.mem_get_info()
        print(f"CUDA: after move free={free_b2} total={total_b2}")
    print("STEP 4: finalizing attention implementation")





    print("STEP 5: model ready")
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

