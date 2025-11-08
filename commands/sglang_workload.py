import os
import sglang as sgl
import torch

MODEL_ID = os.environ.get("GRAPH_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")

@sgl.function
def gen_text(s, text, tokens, temperature):
    s += text
    s += sgl.gen("out", max_tokens=tokens, temperature=temperature)

def text_of_len(n):
    return " ".join(["word"] * n)

if __name__ == "__main__":
    print(f"WORKLOAD: starting with model={MODEL_ID}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"CUDA: device={torch.cuda.get_device_name(0)} cc={props.major}.{props.minor} total_mem={props.total_memory}")
    else:
        print("CUDA: not available")
    print("RUNTIME: initializing sglang Runtime (this may take a while)")
    runtime = sgl.Runtime(model_path=MODEL_ID)
    print("RUNTIME: initialized")
    sgl.set_default_backend(runtime)
    print("WARMUP: starting")
    torch.cuda.nvtx.range_push("warmup")
    gen_text.run(text_of_len(64), 8, 0.0)
    torch.cuda.nvtx.range_pop()
    print("WARMUP: done")
    print("PREFILL: starting B=1 T=1024 tokens=8")
    torch.cuda.nvtx.range_push("prefill_B1_T1024")
    gen_text.run(text_of_len(1024), 8, 0.0)
    torch.cuda.nvtx.range_pop()
    print("PREFILL: done")
    print("DECODE: starting B=1 tokens=128")
    torch.cuda.nvtx.range_push("decode_B1_T128")
    gen_text.run(text_of_len(32), 128, 0.0)
    torch.cuda.nvtx.range_pop()
    print("DECODE: done")


