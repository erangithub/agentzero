from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from agentzero import Delta, Message


class LLM(ABC):
    @abstractmethod
    def complete(self, messages: list[Message], **kwargs) -> Message:
        raise NotImplementedError

    def stream(self, messages: list[Message], **kwargs) -> Iterator[Delta]:
        raise NotImplementedError("Streaming not supported")

    async def acomplete(self, messages: list[Message], **kwargs) -> Message:
        return self.complete(messages, **kwargs)

    async def astream(self, messages: list[Message], **kwargs) -> AsyncIterator[Delta]:
        for delta in self.stream(messages, **kwargs):
            yield delta
