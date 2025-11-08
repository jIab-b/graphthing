import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
import sglang as sgl
import torch

MODEL_ID = os.environ.get("GRAPH_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

def _reset_directory(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

@sgl.function
def gen_json(s, text, max_tokens, temperature):
    s += text
    s += sgl.gen("out", max_tokens=max_tokens, temperature=temperature, regex=r"^\{.*\}$")

def text_of_len(n):
    return " ".join(["word"] * n)

def run_scenario(prompt_len, concurrency, max_tokens):
    texts = [text_of_len(prompt_len) for _ in range(concurrency)]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(lambda t: gen_json.run(t, max_tokens, 0.6), texts))

if __name__ == "__main__":
    _reset_directory("out_local")
    try:
        runtime = sgl.Runtime(model_path=MODEL_ID)
    except Exception as exc:  # pragma: no cover - runtime optional
        print(f"Skipping profile_sequence: failed to start runtime ({exc}).")
        sys.exit(0)
    sgl.set_default_backend(runtime)
    torch.cuda.nvtx.range_push("warmup")
    gen_json.run(text_of_len(64), 8, 0.6)
    torch.cuda.nvtx.range_pop()
    scenarios = [(64, 8, 32), (256, 8, 64), (1024, 4, 64), (2048, 2, 32)]
    torch.cuda.nvtx.range_push("profile_scenarios")
    for prompt_len, concurrency, max_tokens in scenarios:
        torch.cuda.nvtx.range_push(f"prefill_decode_L{prompt_len}_B{concurrency}_T{max_tokens}")
        run_scenario(prompt_len, concurrency, max_tokens)
        torch.cuda.nvtx.range_pop()
    torch.cuda.nvtx.range_pop()
