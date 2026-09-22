import os
from typing import Iterator, Callable
from google import genai
from google.genai import types
from agentzero import Message, Delta, ToolCall
from .base import LLM


def _convert_message(msg: Message) -> types.Content:
    parts = []
    if msg.content:
        parts.append(types.Part(text=msg.content))
    if msg.tool_calls:
        for tc in msg.tool_calls:
            parts.append(
                types.Part(
                    function_call=types.FunctionCall(
                        name=tc.name,
                        id=tc.id,
                        args=tc.arguments,
                    )
                )
            )
    if msg.tool_call_id:
        parts.append(
            types.Part(
                function_response=types.FunctionResponse(
                    id=msg.tool_call_id,
                    result={"result": msg.content},
                )
            )
        )
    role = msg.role
    if role == "system":
        role = "user"
        if parts:
            parts[0].text = f"System: {parts[0].text}"
        else:
            parts.append(types.Part(text="System message"))
    elif role == "assistant":
        role = "model"
    elif role == "tool":
        role = "function"
    return types.Content(role=role, parts=parts)


class GeminiLLM(LLM):
    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: str | None = None,
        **kwargs,
    ):
        self.model = model
        self.client = genai.Client(
            api_key=api_key or os.environ.get("GEMINI_API_KEY"),
            **kwargs,
        )

    def complete(self, messages: list[Message], **kwargs) -> Message:
        tools = kwargs.get("tools")
        tool_fns = kwargs.get("tool_fns")
        contents = [_convert_message(m) for m in messages]
        
        config = {}
        if tool_fns:
            config["tools"] = tool_fns
        elif tools:
            gemini_tools = []
            for tool in tools:
                if isinstance(tool, dict):
                    if "function" in tool:
                        gemini_tools.append(tool["function"])
                    else:
                        gemini_tools.append(tool)
                else:
                    gemini_tools.append(tool)
            config["tools"] = gemini_tools

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config) if config else None,
        )

        candidates = response.candidates
        if not candidates:
            return Message(role="assistant", content="")

        candidate = candidates[0]
        content = candidate.content
        if not content or not content.parts:
            return Message(role="assistant", content="")

        first_part = content.parts[0]
        
        if hasattr(first_part, "function_call") and first_part.function_call:
            fc = first_part.function_call
            tool_calls = (
                ToolCall(
                    id=fc.id or "call_001",
                    name=fc.name,
                    arguments=fc.args,
                ),
            )
            return Message(role="assistant", content=None, tool_calls=tool_calls)
        
        text = "".join(p.text for p in content.parts if hasattr(p, "text") and p.text)
        return Message(role="assistant", content=text)

    def stream(self, messages: list[Message], **kwargs) -> Iterator[Delta]:
        tools = kwargs.get("tools")
        tool_fns = kwargs.get("tool_fns")
        contents = [_convert_message(m) for m in messages]
        
        config = {"stream": True}
        if tool_fns:
            config["tools"] = tool_fns
        elif tools:
            gemini_tools = []
            for tool in tools:
                if isinstance(tool, dict):
                    if "function" in tool:
                        gemini_tools.append(tool["function"])
                    else:
                        gemini_tools.append(tool)
                else:
                    gemini_tools.append(tool)
            config["tools"] = gemini_tools

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=types.GenerateContentConfig(**config),
        )

        for chunk in response:
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        yield Delta(content=part.text)
