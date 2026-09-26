import json

import httpx


'''
Architecture:

LLMProposer
    ↓
LocalLLMClient
    ↓
HTTP
    ↓
llama.cpp
    ↓
Qwen
'''

class LocalLLMClient:
    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8080",
        temperature: float = 0.0
    ) -> None:
        self.base_url = base_url
        self.temperature = temperature

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> str:

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 128,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
            },
            timeout=30.0,
        )

        response.raise_for_status()

        data = response.json()

        return data["choices"][0]["message"]["content"]


    def chat_json(
        self,
        messages: list[dict[str, str]],
        schema: dict,
    ) -> dict:
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": 128,
                "chat_template_kwargs": {
                    "enable_thinking": False,
                },
                "response_format": {
                    "type": "json_object",
                    "schema": schema,
                },
            },
            timeout=30.0,
        )

        response.raise_for_status()

        data = response.json()

        content = data["choices"][0]["message"]["content"]

        return json.loads(content)
