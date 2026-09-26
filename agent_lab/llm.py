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
    ) -> None:
        self.base_url = base_url

    def chat(
        self,
        messages: list[dict[str, str]],
    ) -> str:
        '''
        POST /v1/chat/completions

        request:
            messages
            temperature = 0
            max_tokens = 128

        response:
            choices[0].message.content
        '''

        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={
                "messages": messages,
                "temperature": 0,
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
                "temperature": 0,
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
