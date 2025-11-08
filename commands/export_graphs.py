import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.fx.passes.graph_drawer import FxGraphDrawer
from transformers import masking_utils as hf_masking_utils
from transformers.cache_utils import DynamicCache


MODEL_ID = os.environ.get("GRAPH_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")


def _simple_causal_mask(
    config,
    input_embeds,
    attention_mask,
    cache_position,
    past_key_values,
    position_ids=None,
    **_,
):
    """Basic lower-triangular mask that avoids ProxyTorch custom ops."""
    batch_size, q_len = input_embeds.shape[:2]
    device = input_embeds.device
    dtype = input_embeds.dtype
    if past_key_values is None:
        past_len = 0
    elif hasattr(past_key_values, "get_seq_length"):
        past_len = past_key_values.get_seq_length()
    elif isinstance(past_key_values, tuple) and past_key_values:
        first_entry = past_key_values[0]
        if isinstance(first_entry, (tuple, list)) and first_entry:
            past_len = first_entry[0].shape[2]
        else:
            past_len = int(cache_position[0].item())
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


def _reset_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _safe_export(model, args, export_kwargs):
    """Run torch.export.export and retry without attention_mask if ProxyTorch breaks."""
    try:
        return torch.export.export(model, args, kwargs=export_kwargs)
    except AssertionError as exc:  # pragma: no cover - modal runtime only
        if "ProxyTorchDispatchMode" not in str(exc) or "attention_mask" not in export_kwargs:
            raise
        print("ProxyTorchDispatchMode missing during export; retrying without attention_mask.")
        retry_kwargs = {k: v for k, v in export_kwargs.items() if k != "attention_mask"}
        return torch.export.export(model, args, kwargs=retry_kwargs)


def _maybe_write_dot(ep, label, path):
    try:
        drawer = FxGraphDrawer(ep.graph_module, label)
        dot_graph = drawer.get_main_dot_graph()
    except RuntimeError as exc:
        # Raised when pydot/graphviz is missing from the environment.
        if "pydot" not in str(exc):
            raise
        print(f"Skipping {label} graph export because pydot is missing.")
        return

    base, ext = os.path.splitext(path)
    if not ext:
        ext = ".dot"
        path = base + ext
    writer_name = f"write_{ext.lstrip('.').lower()}"
    writer = getattr(dot_graph, writer_name, None)
    if writer is not None:
        writer(path)
        return
    # Fallback: emit DOT text.
    try:
        dot_text = dot_graph.to_string()
    except Exception:
        dot_text = dot_graph.create_dot()
    with open(path, "w", encoding="utf-8") as f:
        f.write(dot_text)

def export_prefill(model, tok, bsz, tok_len, out_dir):
    x = torch.randint(0, tok.vocab_size, (bsz, tok_len), device="cuda", dtype=torch.long)
    m = torch.ones(bsz, tok_len, device="cuda", dtype=torch.long)
    ep = _safe_export(
        model,
        (x,),
        {
            "attention_mask": m,
            "use_cache": False,
            "return_dict": False,
        },
    )
    torch.export.save(ep, os.path.join(out_dir, f"prefill_B{bsz}_T{tok_len}.pt2"))
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
        layer_k = torch.randn(bsz, nkvh, cache_len, hd, device="cuda", dtype=model.dtype)
        layer_v = torch.randn(bsz, nkvh, cache_len, hd, device="cuda", dtype=model.dtype)
        pkv.append((layer_k, layer_v))
    pkv = tuple(pkv)

    class DecodeWrapper(torch.nn.Module):
        def __init__(self, wrapped_model):
            super().__init__()
            self.wrapped_model = wrapped_model

        def forward(self, input_ids, attn_mask, cached_kv, position_ids):
            cache = DynamicCache(config=self.wrapped_model.config)
            for layer_idx, (layer_k, layer_v) in enumerate(cached_kv):
                cache.update(layer_k, layer_v, layer_idx)
            outputs = self.wrapped_model(
                input_ids,
                attention_mask=attn_mask,
                past_key_values=cache,
                use_cache=True,
                position_ids=position_ids,
                return_dict=False,
            )
            return outputs[0]

    decode_module = DecodeWrapper(model)
    ep = _safe_export(
        decode_module,
        (x, mask, pkv, pos),
        {},
    )
    torch.export.save(ep, os.path.join(out_dir, f"decode_B{bsz}_L{cache_len}.pt2"))
    _maybe_write_dot(ep, "decode", os.path.join(out_dir, f"decode_B{bsz}_L{cache_len}.dot"))

if __name__ == "__main__":
    _reset_directory("out_local")
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model.config._attn_implementation = "eager"
    if getattr(model, "generation_config", None) is not None:
        model.generation_config._attn_implementation = "eager"
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    attn_impl = model.config._attn_implementation
    print(f"Attn implementation: {attn_impl}")
    model.to("cuda")
    model.eval()
    export_prefill(model, tok, 1, 512, "out_local")
    export_decode(model, tok, 1, 2048, "out_local")
