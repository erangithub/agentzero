from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Delta:
    content: str


@dataclass(frozen=True)
class Message:
    role: str
    content: str | None = None
    tool_calls: tuple = ()
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class MessageEvent:
    message: Message
    timestamp: float | None = None
    duration_ms: float | None = None

    def mismatch(self, expect: Message | None) -> str | None:
        """Describe how this recorded event differs from what the caller meant.

        ``expect`` is whatever the caller already holds at the moment replay
        intercepts the call. For a message that is the Message itself, built
        before the event runs. ``None`` means the caller could not state its
        intent -- the value only exists once the work has run, as with an LLM
        call or a prompt -- so there is nothing to disagree with and the
        recorded event stands.

        Returning the reason rather than a bool keeps the wording next to the
        comparison that knows how to describe both sides, so the caller of this
        needs no knowledge of event kinds at all.
        """
        if expect is None:
            return None
        if (expect.role, expect.content) != (self.message.role, self.message.content):
            return (
                f"the next recorded message is {self.message.role}: "
                f"{self.message.content!r}, but {expect.role}: {expect.content!r} was given"
            )
        return None


@dataclass(frozen=True)
class CallEvent:
    fn_name: str
    result: str
    timestamp: float | None = None
    duration_ms: float | None = None
    # TBD: never populated. Nothing records a call's arguments, so replay can
    # only check fn_name -- see mismatch() below and the todo.md entry.
    args: str | None = None

    def mismatch(self, expect: str | None) -> str | None:
        """Describe how this recorded call differs from the call being replayed.

        ``expect`` is the function name, which the caller always knows before
        the call runs. This can only compare the name: ``args`` is part of the
        schema but nothing records it yet, so a call replayed with different
        arguments is indistinguishable from a faithful one.
        """
        if expect is None:
            return None
        if self.fn_name != expect:
            return f"the next recorded call is {self.fn_name}, but {expect} was called"
        return None


# This event is not recorded
@dataclass(frozen=True)
class TransientEvent:
    value: str


Event = MessageEvent | CallEvent | TransientEvent


@dataclass(frozen=True)
class EventNode:
    id: str
    event: Event
    parent: EventNode | None
    depth: int = 0

    def is_message(self, role: str | None = None):
        return isinstance(self.event, MessageEvent) and (
            (role is None) or (role == self.event.message.role)
        )

    def find(self, predicate=None) -> EventNode | None:
        node: EventNode | None = self
        while node:
            if predicate is None or predicate(node):
                return node
            node = node.parent
        return None


class Sequence:
    after_node: EventNode | None
    to_node: EventNode | None

    def __init__(self, after_node: EventNode | None, to_node: EventNode | None):
        self.after_node = after_node
        self.to_node = to_node

    def __bool__(self):
        return self.to_node is not None and self.after_node != self.to_node

    def iter_messages(self) -> Iterator[Message]:
        for node in self.iter_nodes():
            if isinstance(node.event, MessageEvent):
                yield node.event.message

    def iter_nodes(self) -> Iterator[EventNode]:
        stack: list[EventNode] = []
        node: EventNode | None = self.to_node
        while node and (node != self.after_node):
            stack.append(node)
            node = node.parent
        while stack:
            yield stack.pop()


class WriteHead:
    def __init__(self, prev: EventNode | None = None):
        self.prev = prev

    def append(self, event: Event):
        if isinstance(event, TransientEvent):
            return
        node_id = str(uuid.uuid4())
        depth = (self.prev.depth + 1) if self.prev else 0
        self.prev = EventNode(id=node_id, event=event, parent=self.prev, depth=depth)


class ReadHead:
    def __init__(self, s: Sequence):
        self.iter: Iterator[EventNode] = s.iter_nodes()
        self.next: EventNode | None = next(self.iter, None)
        self.prev: EventNode | None = self.next.parent if self.next else None

    def step(self):
        if self.next:
            self.prev = self.next
            self.next = next(self.iter, None)

    def __bool__(self):
        return (self.prev is not None) or (self.next is not None)


def build_context(
    sequence: Sequence,
    system: str | None = None,
    injections: list[str] | None = None,
) -> list[Message]:
    context: list[Message] = []
    if system:
        context.append(Message(role="system", content=system))
    for text in injections or []:
        context.append(Message(role="system", content=text))
    for msg in sequence.iter_messages():
        context.append(msg)
    return context
