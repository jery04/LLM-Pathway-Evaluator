from google import genai

client = genai.Client(
    api_key="AQ.Ab8RN6LByPrhqnpqPPFaMiKRcAKJoO_D1CIh2IowDdYZk96p2g"
)

MODEL = "gemini-2.5-flash"

def ask_llm(prompt: str) -> str:
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    return response.text.strip()

print(ask_llm("What is the capital of France?"))