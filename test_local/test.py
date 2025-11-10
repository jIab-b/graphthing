import os
import subprocess

TEST_PROMPT = "write a medieval poem"
DECODE_STEPS = 8


def run_test():
    cmd = [
        os.path.join("test_venv", "bin", "python"),
        os.path.join("test_local", "reconstruct_infer.py"),
        "--prompt",
        TEST_PROMPT,
        "--decode-steps",
        str(DECODE_STEPS),
        "--temperature",
        "0",
        "--top-p",
        "1.0",
        "--repetition-penalty",
        "1.0",
        "--disable-chat-template",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(proc.stdout)


def __main__():
    run_test()

if __name__ == "__main__":
    __main__()