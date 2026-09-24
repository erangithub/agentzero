from collections.abc import Iterator
from time import sleep

from agentzero import Delta, Message

from .base import LLM


class EchoLLM(LLM):
    def complete(self, messages: list[Message], **kwargs) -> Message:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        return Message(role="assistant", content=f"echo: {last_user.content if last_user else ''}")

    def stream(self, messages: list[Message], **kwargs) -> Iterator[Delta]:
        last_user = next((m for m in reversed(messages) if m.role == "user"), None)
        for word in f"echo: {last_user.content if last_user else ''}".split():
            sleep(0.1)
            yield Delta(content=word + " ")
