"""
agentzero.mock — development and testing utilities

These are not production components. They are useful for
unit tests and examples without any external dependencies.
"""
import json
import time
from typing import Iterator
from agentzero import Message, Delta, ToolCall
from agentzero.llms import LLM


class ToolCallLLM(LLM):
    """
    Simulates a two-turn tool use interaction:
      turn 1 — returns a tool call
      turn 2 — returns a final answer once a tool result is in context

    Useful for testing tool use loops without a real LLM.
    """

    def __init__(self, tool_name: str, arguments: dict, final_answer: str):
        self.tool_name    = tool_name
        self.arguments    = arguments
        self.final_answer = final_answer
        self._call_id     = "call_001"

    def complete(self, messages: list[Message], **kwargs) -> Message:
        if any(m.role == "tool" for m in messages):
            return Message("assistant", self.final_answer)
        return Message(
            role="assistant",
            content=None,
            tool_calls=(
                ToolCall(
                    id=self._call_id,
                    name=self.tool_name,
                    arguments=json.dumps(self.arguments),
                ),
            ),
        )

    def stream(self, messages: list[Message], **kwargs) -> Iterator[Delta]:
        response = self.complete(messages, **kwargs)
        if response.content:
            for word in response.content.split():
                time.sleep(0.05)
                yield Delta(content=word + " ")


def get_weather(location: str) -> str:
    """Returns a fixed weather string. Stand-in for a real weather API."""
    return f"25c and sunny in {location}"


tool_schemas = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": get_weather.__doc__,
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                },
                "required": ["location"],
            },
        },
    }
]