from collections.abc import Iterator

from openai import OpenAI

from agentzero import Delta, Message

from .base import LLM


class OllamaLLM(LLM):
    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434/v1",
        **kwargs,
    ):
        self.model = model
        self.client = OpenAI(api_key="ollama", base_url=base_url, **kwargs)

    def complete(self, messages: list[Message], **kwargs) -> Message:
        tools = kwargs.get("tools")
        from .openai import _to_dict

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            tools=tools,
        )
        msg = response.choices[0].message
        return Message(role=msg.role or "assistant", content=msg.content or "")

    def stream(self, messages: list[Message], **kwargs) -> Iterator[Delta]:
        tools = kwargs.get("tools")
        from .openai import _to_dict

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            tools=tools,
            stream=True,
        )
        for chunk in response:
            if delta := chunk.choices[0].delta.content:
                yield Delta(delta)
