import sglang as sgl

@sgl.function
def generate_user_profile(s, name):
    s += f"generate a user profile for {name} in json format:\n"
    s += sgl.gen("profile", max_tokens=200,
            regex=r"^\{.*\}$", temperature=0.6)

if __name__ == '__main__':
    # Initialize sglang backend with the DeepSeek model
    model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    sgl.set_default_backend(sgl.Runtime(model_path=model_path))

    result = generate_user_profile.run("Bob")
    print(result["profile"])