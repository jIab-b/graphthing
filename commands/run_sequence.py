import subprocess
import time
import socket
import openai

def wait_for_port(port, timeout=60):
    start = time.time()
    while time.time() - start < timeout:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect(('127.0.0.1', port))
            sock.close()
            return True
        except:
            sock.close()
            time.sleep(1)
    return False

# Start sglang server asynchronously
server_cmd = [
    "python", "-m", "sglang.launch_server",
    "--model-path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "--reasoning-parser", "deepseek-r1",
    "--port", "30000"
]

p = subprocess.Popen(server_cmd)

# Run client to make request and write response (client will wait for server)
client = openai.OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY"
)

# Wait for server to be ready before making request
if not wait_for_port(30000):
    p.terminate()
    raise RuntimeError("Server failed to start within timeout")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 1+3?"}],
    temperature=0.6,
    max_tokens=1024,
    extra_body={"separate_reasoning": True}
)

with open("./out_local/response.txt", "w") as f:
    f.write(f"Reasoning:\n{response.choices[0].message.reasoning_content}\n\n")
    f.write(f"Answer:\n{response.choices[0].message.content}\n")

# Terminate server after output is completed
p.terminate()
p.wait()
