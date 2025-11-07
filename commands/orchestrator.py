import subprocess
import time
import socket

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

server_cmd = [
    "python", "-m", "sglang.launch_server",
    "--model-path", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "--reasoning-parser", "deepseek-r1",
    "--port", "30000"
]

p = subprocess.Popen(server_cmd)

if not wait_for_port(30000):
    p.terminate()
    raise RuntimeError("Server failed to start within timeout")

subprocess.run(["python", "commands/client.py"])

p.terminate()
p.wait()
