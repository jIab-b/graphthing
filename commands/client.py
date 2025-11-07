import openai

client = openai.OpenAI(
    base_url="http://127.0.0.1:30000/v1",
    api_key="EMPTY"
)

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "What is 1+3?"}],
    temperature=0.6,
    max_tokens=1024,
    extra_body={"separate_reasoning": True}
)

with open("/workspace/out_local/response.txt", "w") as f:
    f.write(f"Reasoning:\n{response.choices[0].message.reasoning_content}\n\n")
    f.write(f"Answer:\n{response.choices[0].message.content}\n")
