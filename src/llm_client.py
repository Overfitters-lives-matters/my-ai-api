from openai import OpenAI


class LLMClient:
    def __init__(self, api_key):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key
        )

    def stream_response(self, messages):
        stream = self.client.chat.completions.create(
            model="openrouter/auto",
            messages=messages,
            stream=True
        )

        response = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content or ""
            response += token

        return response