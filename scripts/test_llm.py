# uv run python -m scripts.test_llm
from agent_lab.llm import LocalLLMClient


client = LocalLLMClient()

response = client.chat(
    messages=[
        {
            "role": "system",
            "content": "Answer concisely.",
        },
        {
            "role": "user",
            "content": "What is an AI agent?",
        },
    ]
)

print(response)
