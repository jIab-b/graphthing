import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopPLogitsWarper,
)

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


def _build_prompt_tokens(tokenizer, prompt: str, system: str | None, use_chat_template: bool) -> torch.Tensor:
    if use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        token_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return torch.tensor(token_ids, dtype=torch.long)
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    return encoded["input_ids"][0]


def _prepare_prefill_inputs(
    tokenizer,
    prompt: str,
    batch: int,
    seq_len: int,
    device: str,
    system: str | None,
    use_chat_template: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Sequence[torch.Tensor]]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token or "<pad>"
    prompt_tokens = _build_prompt_tokens(tokenizer, prompt, system, use_chat_template)
    prompt_tokens = prompt_tokens[:seq_len]
    prompt_len = prompt_tokens.numel()
    pad_len = max(seq_len - prompt_len, 0)
    pad_val = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    if pad_len:
        pad = torch.full((pad_len,), pad_val, dtype=torch.long)
        full = torch.cat([prompt_tokens, pad], dim=0)
    else:
        full = prompt_tokens
    attn = torch.cat([torch.ones(prompt_len, dtype=torch.long), torch.zeros(pad_len, dtype=torch.long)], dim=0)
    input_ids = full.unsqueeze(0).repeat(batch, 1).to(device)
    attention_mask = attn.unsqueeze(0).repeat(batch, 1).to(device)
    prompt_repeated = [prompt_tokens.clone() for _ in range(batch)]
    return input_ids, attention_mask, prompt_repeated


def _load_program(path: str, device: str):
    prog = torch.export.load(path)
    mod = prog.module()
    mod.to(device)
    return mod


def _build_processors(temperature: float, top_p: float, repetition_penalty: float) -> Tuple[LogitsProcessorList, LogitsProcessorList]:
    base = LogitsProcessorList()
    if repetition_penalty and repetition_penalty != 1.0:
        base.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    sampling = LogitsProcessorList()
    if temperature and temperature != 1.0:
        sampling.append(TemperatureLogitsWarper(temperature))
    if top_p and top_p < 1.0:
        sampling.append(TopPLogitsWarper(top_p))
    return base, sampling


def _decode_text(tokenizer, prompt_tokens: torch.Tensor, generated: List[int]) -> str:
    seq = torch.cat([prompt_tokens, torch.tensor(generated, dtype=torch.long)], dim=0)
    return tokenizer.decode(seq, skip_special_tokens=True)


def _select_next_token(
    scores: torch.Tensor,
    history: torch.Tensor,
    base_processors: LogitsProcessorList,
    sampling_processors: LogitsProcessorList,
    temperature: float,
) -> torch.Tensor:
    processed = base_processors(history, scores) if base_processors else scores
    if temperature == 0.0:
        return processed.argmax(dim=-1, keepdim=True)
    warped = sampling_processors(history, processed) if sampling_processors else processed
    probs = F.softmax(warped, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _maybe_finalize_sample(
    tokenizer,
    prompt_tokens,
    generated_lists,
    sample_idx: int,
    eos_id: int | None,
    stop_sequences: Sequence[str],
    finished: torch.Tensor,
    final_texts: List[str | None],
):
    if finished[sample_idx]:
        return
    text = _decode_text(tokenizer, prompt_tokens[sample_idx], generated_lists[sample_idx]).rstrip()
    if eos_id is not None and generated_lists[sample_idx] and generated_lists[sample_idx][-1] == eos_id:
        generated_lists[sample_idx].pop()
        final_texts[sample_idx] = _decode_text(tokenizer, prompt_tokens[sample_idx], generated_lists[sample_idx]).strip()
        finished[sample_idx] = True
        return
    for stop in stop_sequences:
        idx = text.find(stop)
        if idx != -1:
            final_texts[sample_idx] = text[:idx].strip()
            finished[sample_idx] = True
            return


def _mask_finished(next_ids: torch.Tensor, finished: torch.Tensor, pad_token_id: int | None) -> torch.Tensor:
    if not torch.any(finished):
        return next_ids
    pad_val = pad_token_id if pad_token_id is not None else 0
    pad = torch.full_like(next_ids, pad_val)
    return torch.where(finished.view(-1, 1), pad, next_ids)


def _ensure_pair(prefill, decode):
    if prefill["batch_size"] != decode["batch_size"]:
        raise SystemExit("Prefill/decode batch sizes differ")
    if decode.get("cache_length") is None:
        raise SystemExit("Decode scenario missing cache_length")


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
    input_ids, attn_mask, prompt_tokens = _prepare_prefill_inputs(
        tokenizer,
        args.prompt,
        prefill["batch_size"],
        prefill["sequence_length"],
        device,
        args.system,
        not args.disable_chat_template,
    )

    prefill_mod = _load_program(prefill["artifacts"]["torch_export"], device)
    decode_mod = _load_program(decode["artifacts"]["torch_export"], device)

    if args.decode_steps <= 0:
        for idx in range(prefill["batch_size"]):
            base_text = _decode_text(tokenizer, prompt_tokens[idx], [])
            print(f"[sample {idx}] {base_text.strip()}")
        return

    with torch.inference_mode():
        prefill_logits, kv_flat = prefill_mod(input_ids, attn_mask)

    kv_flat = truncate_kv(kv_flat, decode.get("cache_length"))
    seq_len = kv_sequence_length(kv_flat)
    position = torch.full((prefill["batch_size"],), seq_len, device=device, dtype=torch.long)
    generated_tensor = torch.empty(prefill["batch_size"], 0, dtype=torch.long, device=device)
    generated_lists: List[List[int]] = [[] for _ in range(prefill["batch_size"])]
    finished = torch.zeros(prefill["batch_size"], dtype=torch.bool, device=device)
    final_texts: List[str | None] = [None] * prefill["batch_size"]
    eos_id = tokenizer.eos_token_id
    stop_sequences = args.stop or []
    base_processors, sampling_processors = _build_processors(args.temperature, args.top_p, args.repetition_penalty)

    history = input_ids
    first_scores = prefill_logits[:, -1, :]
    next_ids = _select_next_token(first_scores, history, base_processors, sampling_processors, args.temperature)
    generated_tensor = torch.cat([generated_tensor, next_ids], dim=1)
    for b in range(prefill["batch_size"]):
        generated_lists[b].append(int(next_ids[b].item()))
        _maybe_finalize_sample(
            tokenizer, prompt_tokens, generated_lists, b, eos_id, stop_sequences, finished, final_texts
        )
    next_ids = _mask_finished(next_ids, finished, tokenizer.pad_token_id)
    current_token = next_ids

    for _ in range(max(args.decode_steps - 1, 0)):
        if torch.all(finished):
            break
        attn = torch.ones(
            decode["batch_size"], decode["cache_length"] + 1, dtype=torch.long, device=device
        )
        pos_ids = position.view(-1, 1)
        with torch.inference_mode():
            logits, kv_flat = decode_mod(current_token, attn, kv_flat, pos_ids)
        scores = logits[:, -1, :]
        history = torch.cat([input_ids, generated_tensor], dim=1)
        next_ids = _select_next_token(scores, history, base_processors, sampling_processors, args.temperature)
        generated_tensor = torch.cat([generated_tensor, next_ids], dim=1)
        kv_flat = truncate_kv(kv_flat, decode["cache_length"])
        position = position + 1
        current_token = next_ids

        for b in range(prefill["batch_size"]):
            if finished[b]:
                continue
            generated_lists[b].append(int(next_ids[b].item()))
            _maybe_finalize_sample(
                tokenizer, prompt_tokens, generated_lists, b, eos_id, stop_sequences, finished, final_texts
            )
        next_ids = _mask_finished(next_ids, finished, tokenizer.pad_token_id)
        current_token = next_ids

    for idx in range(prefill["batch_size"]):
        if final_texts[idx] is None:
            final_texts[idx] = _decode_text(tokenizer, prompt_tokens[idx], generated_lists[idx]).strip()
        prompt_text = _decode_text(tokenizer, prompt_tokens[idx], []).strip()
        completion = final_texts[idx]
        if completion.startswith(prompt_text):
            completion = completion[len(prompt_text):].lstrip()
        print(f"[sample {idx}] {completion}")


def parse_args():
    parser = argparse.ArgumentParser(description="Replay inference using exported TorchExport graphs")
    parser.add_argument("--manifest", default=str(Path(__file__).with_name("out_local") / "export_manifest.json"))
    parser.add_argument("--plan", default=None, help="Optional override for full_inference_plan path")
    parser.add_argument("--bundle-id", help="Bundle id inside full_inference_plan", default=None)
    parser.add_argument("--prefill-id", help="Prefill scenario id override", default=None)
    parser.add_argument("--decode-id", help="Decode scenario id override", default=None)
    parser.add_argument("--prompt", default="Hello world", help="Prompt text to feed the tokenizer")
    parser.add_argument("--system", default=None, help="Optional system prompt when chat template is available")
    parser.add_argument("--disable-chat-template", action="store_true", help="Bypass tokenizer chat template")
    parser.add_argument("--decode-steps", type=int, default=32, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--stop", action="append", default=[], help="Optional stop strings; repeatable")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
