import os
from concurrent.futures import ThreadPoolExecutor
import sglang as sgl
import torch

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
    os.makedirs("out_local", exist_ok=True)
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    sgl.set_default_backend(sgl.Runtime(model_path=model_path))
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


