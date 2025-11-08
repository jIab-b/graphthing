import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.fx.passes.graph_drawer import FxGraphDrawer

def export_prefill(model, tok, bsz, tok_len, out_dir):
    x = torch.randint(0, tok.vocab_size, (bsz, tok_len), device="cuda", dtype=torch.long)
    m = torch.ones(bsz, tok_len, device="cuda", dtype=torch.long)
    ep = torch.export.export(model, (x,), kwargs={"attention_mask": m, "use_cache": False, "return_dict": False})
    ep.save(os.path.join(out_dir, f"prefill_B{bsz}_T{tok_len}.pt2"))
    FxGraphDrawer(ep.graph_module, "prefill").to_dot_file(os.path.join(out_dir, f"prefill_B{bsz}_T{tok_len}.dot"))

def export_decode(model, tok, bsz, cache_len, out_dir):
    cfg = model.config
    nh = cfg.num_attention_heads
    nkvh = getattr(cfg, "num_key_value_heads", nh)
    hd = cfg.hidden_size // nh
    x = torch.randint(0, tok.vocab_size, (bsz, 1), device="cuda", dtype=torch.long)
    k = torch.randn(bsz, nkvh, cache_len, hd, device="cuda", dtype=model.dtype)
    v = torch.randn(bsz, nkvh, cache_len, hd, device="cuda", dtype=model.dtype)
    mask = torch.ones(bsz, cache_len + 1, device="cuda", dtype=torch.long)
    pos = torch.arange(cache_len, cache_len + 1, device="cuda", dtype=torch.long).unsqueeze(0).expand(bsz, 1)
    pkv = tuple((k, v) for _ in range(cfg.num_hidden_layers))
    ep = torch.export.export(model, (x,), kwargs={"attention_mask": mask, "past_key_values": pkv, "use_cache": True, "position_ids": pos, "return_dict": False})
    ep.save(os.path.join(out_dir, f"decode_B{bsz}_L{cache_len}.pt2"))
    FxGraphDrawer(ep.graph_module, "decode").to_dot_file(os.path.join(out_dir, f"decode_B{bsz}_L{cache_len}.dot"))

if __name__ == "__main__":
    os.makedirs("out_local", exist_ok=True)
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)
    model.to("cuda")
    model.eval()
    export_prefill(model, tok, 1, 512, "out_local")
    export_decode(model, tok, 1, 2048, "out_local")


