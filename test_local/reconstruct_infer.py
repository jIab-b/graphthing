import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer

from kv_utils import truncate_kv, kv_sequence_length


def _load_json(path: Path) -> Dict:
    with path.open() as f:
        return json.load(f)


def _scenario_map(manifest: Dict) -> Dict[str, Dict]:
    return {sc["id"]: sc for sc in manifest.get("scenarios", [])}


def _select_bundle(plan: Dict, bundle_id: str | None) -> Dict:
    bundles = plan.get("bundles", [])
    if not bundles:
        raise SystemExit("full_inference_plan has no bundles")
    if bundle_id is None:
        return bundles[0]
    for bundle in bundles:
        if bundle["id"] == bundle_id:
            return bundle
    raise SystemExit(f"bundle_id {bundle_id} not found")


def _device_from_arg(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise SystemExit("requested cuda but torch.cuda.is_available() is False")
    return name


def _prepare_prefill_inputs(tokenizer, prompt: str, batch: int, seq_len: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<pad>"
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    tokens = encoded["input_ids"][0]
    if tokens.numel() >= seq_len:
        trimmed = tokens[:seq_len]
        attn = torch.ones(seq_len, dtype=torch.long)
    else:
        pad_len = seq_len - tokens.numel()
        pad_val = tokenizer.pad_token_id or 0
        pad = torch.full((pad_len,), pad_val, dtype=torch.long)
        trimmed = torch.cat([tokens, pad], dim=0)
        attn = torch.cat([torch.ones(tokens.numel(), dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)], dim=0)
    input_ids = trimmed.unsqueeze(0).repeat(batch, 1).to(device)
    attention_mask = attn.unsqueeze(0).repeat(batch, 1).to(device)
    return input_ids, attention_mask


def _load_program(path: str, device: str):
    prog = torch.export.load(path)
    mod = prog.module()
    mod.to(device)
    return mod


def _sample_next(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)


def _decode_tokens(tokenizer, prompt_ids: torch.Tensor, generated: List[List[int]]) -> List[str]:
    outputs = []
    for batch_idx, gen in enumerate(generated):
        seq = torch.cat([prompt_ids[batch_idx], torch.tensor(gen, dtype=torch.long)], dim=0)
        outputs.append(tokenizer.decode(seq, skip_special_tokens=True))
    return outputs


def _ensure_pair(prefill, decode):
    if prefill["batch_size"] != decode["batch_size"]:
        raise SystemExit("Prefill/decode batch sizes differ")
    if prefill["sequence_length"] != decode.get("cache_length"):
        raise SystemExit("Prefill sequence length must match decode cache length for reconstruction")


def run(args):
    manifest_path = Path(args.manifest)
    manifest = _load_json(manifest_path)
    plan_path = Path(args.plan or manifest.get("full_inference_plan", manifest_path.parent / "full_inference_plan.json"))
    plan = _load_json(plan_path)
    bundle = _select_bundle(plan, args.bundle_id)
    scenarios = _scenario_map(manifest)
    pf_id = args.prefill_id or bundle["kv_cache_contract"]["producer_scenario"]
    dec_id = args.decode_id or bundle["kv_cache_contract"]["consumer_scenario"]
    prefill = scenarios[pf_id]
    decode = scenarios[dec_id]
    _ensure_pair(prefill, decode)

    device = _device_from_arg(args.device)
    tokenizer = AutoTokenizer.from_pretrained(manifest["model_id"], use_fast=True, trust_remote_code=True)
    input_ids, attn_mask = _prepare_prefill_inputs(tokenizer, args.prompt, prefill["batch_size"], prefill["sequence_length"], device)

    prefill_mod = _load_program(prefill["artifacts"]["torch_export"], device)
    decode_mod = _load_program(decode["artifacts"]["torch_export"], device)

    with torch.inference_mode():
        logits, kv_flat = prefill_mod(input_ids, attn_mask)

    kv_flat = truncate_kv(kv_flat, decode.get("cache_length"))
    seq_len = kv_sequence_length(kv_flat)
    current_token = input_ids[:, -1:].clone()
    position = torch.tensor(seq_len, device=device, dtype=torch.long)

    generated: List[List[int]] = [[] for _ in range(prefill["batch_size"])]
    for _ in range(args.decode_steps):
        attn = torch.ones(decode["batch_size"], decode["cache_length"] + 1, dtype=torch.long, device=device)
        pos_ids = position.view(1, 1).repeat(decode["batch_size"], 1)
        with torch.inference_mode():
            logits, kv_flat = decode_mod(current_token, attn, kv_flat, pos_ids)
        next_ids = _sample_next(logits[:, -1, :])
        for b in range(prefill["batch_size"]):
            generated[b].append(int(next_ids[b]))
        current_token = next_ids.view(prefill["batch_size"], 1)
        kv_flat = truncate_kv(kv_flat, decode["cache_length"])
        position += 1

    outputs = _decode_tokens(tokenizer, input_ids.cpu(), generated)
    for idx, text in enumerate(outputs):
        print(f"[sample {idx}] {text}")


def parse_args():
    parser = argparse.ArgumentParser(description="Replay inference using exported TorchExport graphs")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("out_local") / "export_manifest.json"))
    parser.add_argument("--plan", default=None, help="Optional override for full_inference_plan path")
    parser.add_argument("--bundle-id", help="Bundle id inside full_inference_plan", default=None)
    parser.add_argument("--prefill-id", help="Prefill scenario id override", default=None)
    parser.add_argument("--decode-id", help="Decode scenario id override", default=None)
    parser.add_argument("--prompt", default="Hello world", help="Prompt text to feed the tokenizer")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
