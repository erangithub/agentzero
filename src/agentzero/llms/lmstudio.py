from collections.abc import AsyncIterator, Iterator

from openai import AsyncOpenAI, OpenAI

from agentzero import Delta, Message

from .base import LLM


class LMStudioLLM(LLM):
    def __init__(
        self,
        model: str = "local-model",
        base_url: str = "http://localhost:1234/v1",
        **kwargs,
    ):
        self.model = model
        self.client = OpenAI(api_key="lm-studio", base_url=base_url, **kwargs)
        self.async_client = AsyncOpenAI(api_key="lm-studio", base_url=base_url, **kwargs)

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

    async def acomplete(self, messages: list[Message], **kwargs) -> Message:
        tools = kwargs.get("tools")
        from .openai import _to_dict

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            tools=tools,
        )
        msg = response.choices[0].message
        return Message(role=msg.role or "assistant", content=msg.content or "")

    async def astream(self, messages: list[Message], **kwargs) -> AsyncIterator[Delta]:
        tools = kwargs.get("tools")
        from .openai import _to_dict

        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            tools=tools,
            stream=True,
        )
        async for chunk in response:
            if delta := chunk.choices[0].delta.content:
                yield Delta(delta)
