from openai import OpenAI 

client = OpenAI(
    api_key="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",
    base_url="https://openrouter.ai/api/v1"
)

MODEL = "deepseek/deepseek-chat-v3-0324"

def ask_llm(prompt: str) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

print(ask_llm("What is the capital of France?"))