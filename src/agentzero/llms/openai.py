from collections.abc import AsyncIterator, Iterator

from openai import AsyncOpenAI, OpenAI

from agentzero import Delta, Message, ToolCall

from .base import LLM


def _to_dict(msg: Message) -> dict:
    result: dict = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        result["tool_calls"] = [
            {"id": tc.id, "function": {"name": tc.name, "arguments": tc.arguments}}
            for tc in msg.tool_calls
        ]
    if msg.tool_call_id:
        result["tool_call_id"] = msg.tool_call_id
    return result


class OpenAILLM(LLM):
    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url, **kwargs)

    def complete(self, messages: list[Message], **kwargs) -> Message:
        tools = kwargs.get("tools")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            tools=tools,
        )
        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls: tuple[ToolCall, ...] = ()
        if msg.tool_calls:
            tool_calls = tuple(
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in msg.tool_calls
            )
        return Message(role=msg.role or "assistant", content=content, tool_calls=tool_calls)

    def stream(self, messages: list[Message], **kwargs) -> Iterator[Delta]:
        tools = kwargs.get("tools")
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
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            tools=tools,
        )
        choice = response.choices[0]
        msg = choice.message
        content = msg.content or ""
        tool_calls: tuple[ToolCall, ...] = ()
        if msg.tool_calls:
            tool_calls = tuple(
                ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
                for tc in msg.tool_calls
            )
        return Message(role=msg.role or "assistant", content=content, tool_calls=tool_calls)

    async def astream(self, messages: list[Message], **kwargs) -> AsyncIterator[Delta]:
        tools = kwargs.get("tools")
        response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=[_to_dict(m) for m in messages],
            tools=tools,
            stream=True,
        )
        async for chunk in response:
            if delta := chunk.choices[0].delta.content:
                yield Delta(delta)
