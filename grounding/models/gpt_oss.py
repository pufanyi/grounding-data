from openai import OpenAI

from .model import Model


class GPTOSS(Model):
    def __init__(self, max_retries: int = 5):
        self.openai_api_key = "EMPTY"
        self.openai_api_base = "http://localhost:8000/v1"
        self.client = OpenAI(
            api_key=self.openai_api_key, base_url=self.openai_api_base, max_retries=5
        )
        models = self.client.models.list()
        self.model = models.data[0].id

    def generate(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]

        extra_body = {"chat_template_kwargs": {"thinking": False}}

        response = self.client.chat.completions.create(
            model=self.model, messages=messages, extra_body=extra_body
        )

        content = response.choices[0].message.content

        return content
