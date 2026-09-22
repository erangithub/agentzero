from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterator, Any
import json
import uuid


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


class ControlKind(Enum):
    branch_start = "branch_start"
    branch_cancelled = "branch_cancelled"


@dataclass
class ControlEvent:
    control: ControlKind


@dataclass
class MessageEvent:
    message: Message
    timestamp: float | None = None
    duration_ms: float | None = None


@dataclass
class CallEvent:
    fn_name: str
    result: str
    timestamp: float | None = None
    duration_ms: float | None = None
    args: str | None = None

# This event is not recorded
@dataclass
class TransientEvent:
    value: str

Event = MessageEvent | CallEvent | ControlEvent | TransientEvent


@dataclass(frozen=True)
class EventNode:
    id: str
    event: Event
    parent: EventNode | None
    depth: int = 0
    
    def is_message(self, role : str | None =None):
        return isinstance(self.event, MessageEvent) and ((role is None) or (role == self.event.message.role))

    def find(self, predicate=None) -> EventNode | None:
        node = self
        while node:
            if predicate is None or predicate(node):
                return node
            node = node.parent
        return None


class Sequence:
    after_node: EventNode | None
    to_node: EventNode

    def __init__(self, after_node: EventNode | None, to_node: EventNode):
        self.after_node = after_node
        self.to_node = to_node
        assert(self.to_node is not None)

    def __bool__(self):
        return self.after_node != self.to_node
    
    def iter_messages(self) -> Iterator[Message]:
        for node in self.iter_nodes():
            if isinstance(node.event, MessageEvent):
                yield node.event.message

    def iter_nodes(self) -> Iterator[EventNode]:
        stack = []
        node = self.to_node
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

    def fork(self) -> EventNode:
        node_id = str(uuid.uuid4())
        depth = (self.prev.depth + 1) if self.prev else 0
        return EventNode(
            id=node_id,
            event=ControlEvent(control=ControlKind.branch_start),
            parent=self.prev,
            depth=depth,
        )

    @staticmethod
    def start() -> WriteHead:
        return WriteHead()


class ReadHead:
    def __init__(self, s: Sequence):
        self.iter = s.iter_nodes() if s else None
        self.next = next(self.iter) if s else None
        self.prev = self.next.parent if self.next else None

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
    for text in (injections or []):
        context.append(Message(role="system", content=text))
    for msg in sequence.iter_messages():
        context.append(msg)
    return context