from __future__ import annotations

import functools
import json
from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import Any, TypeVar

from agentzero.eventlog import (
    CallEvent,
    ControlEvent,
    Event,
    EventNode,
    Message,
    MessageEvent,
    ReadHead,
    Sequence,
    ToolCall,
    TransientEvent,
    WriteHead,
)
from agentzero.llms.base import LLM

T = TypeVar("T")


@dataclass
class Tool:
    name: str
    fn: Callable[..., str]
    schema: dict


def transient(value: Any):
    return TransientEvent(value=value)


class Environment:
    def __init__(
        self,
        llm: LLM | None = None,
        input_fn: Callable[[], str] | None = None,
        continue_live: bool = False,
        origin_node: EventNode | None = None,
        registered_fns: dict[str, Callable] | None = None,
    ):
        self.registered_fns: dict[str, Callable] = {}
        for name, fn in (registered_fns or {}).items():
            setattr(self, name, MethodType(fn, self))
            self.registered_fns[name] = fn

        if origin_node is None:
            origin_node = WriteHead.start().fork()
        self.origin_node = origin_node
        self.write_head = WriteHead(prev=origin_node)
        self.forks: dict[str, list[Environment]] = {}

        if llm is not None:
            self.register_llm_fn(llm.complete)
            if hasattr(llm, "acomplete"):
                self.register_llm_afn(llm.acomplete)
            if hasattr(llm, "stream"):
                self.register_llm_stream_fn(llm.stream)
        if input_fn is not None:
            self.register_input_fn(input_fn)

        self.rewind(continue_live)
        self.replay_stop_predicate: Callable[[EventNode], bool] | None = None

    @property
    def is_replay(self) -> bool:
        nxt = self.read_head.next
        return nxt is not None and not self._is_replay_stop_on(nxt)

    def _is_replay_stop_on(self, node: EventNode) -> bool:
        if self.replay_stop_predicate is None:
            return False
        return bool(self.replay_stop_predicate(node))

    def _write(self, event: Event):
        if isinstance(event, TransientEvent):
            return

        if self.is_replay:
            raise RuntimeError("Cannot write in replay mode")
        self.write_head.append(event)
        self.child_index = 0

    def _read(self) -> MessageEvent | CallEvent | None:
        assert self.is_replay
        self.child_index = 0
        node = self.read_head.next
        while node is not None:
            if isinstance(node.event, ControlEvent):
                self.read_head.step()
                if not self.is_replay:
                    return None
                node = self.read_head.next
                continue
            self.read_head.step()
            prev = self.read_head.prev
            assert prev is not None
            assert isinstance(prev.event, (MessageEvent, CallEvent))
            return prev.event
        return None

    @property
    def prev_node(self) -> EventNode:
        node = self.read_head.prev or self.write_head.prev
        assert node is not None
        return node

    @property
    def current_depth(self):
        return self.prev_node.depth + 1

    def history(self) -> Sequence:
        return Sequence(after_node=None, to_node=self.prev_node)

    def full_history(self) -> Sequence:
        assert self.write_head.prev is not None
        return Sequence(after_node=None, to_node=self.write_head.prev)

    def fork_history(self) -> Sequence:
        return Sequence(after_node=self.origin_node, to_node=self.prev_node)

    def full_fork_history(self) -> Sequence:
        assert self.write_head.prev is not None
        return Sequence(after_node=self.origin_node, to_node=self.write_head.prev)

    def fork(self) -> Environment:
        forknode = self.prev_node
        if forknode.id not in self.forks:
            self.forks[forknode.id] = []
        child_envs = self.forks[forknode.id]
        if self.child_index < len(child_envs):
            forked_env = child_envs[self.child_index]
            forked_env.rewind()
        else:
            forked_env = Environment(
                continue_live=self.continue_live,
                origin_node=self.write_head.fork(),
                registered_fns=self.registered_fns,
            )
            child_envs.append(forked_env)

        self.child_index += 1
        return forked_env

    # --- core invoke primitives ---

    def _message_event(self, fn: Callable[[], Message]) -> Message:
        """Invoke a function that produces a Message. Writes a MessageEvent.

        If the function returns TransientEvent, it's returned without recording.
        """
        if self.is_replay:
            event = self._read()
            assert event is not None
            if not isinstance(event, MessageEvent):
                raise RuntimeError(f"Expected MessageEvent, got {type(event)}")
            return event.message
        elif not self.continue_live:
            raise RuntimeError("Replay exhausted")
        result = fn()
        if isinstance(result, TransientEvent):
            return result
        if not isinstance(result, Message):
            raise RuntimeError(f"Expected Message, got {type(result)}")

        # Sync write_head to read_head when going live with a real message
        if self.read_head.prev:
            self.write_head.prev = self.read_head.prev
            self.read_head.prev = self.read_head.next = None

        self._write(MessageEvent(message=result))
        self.replay_stop_predicate = None

        return result

    async def _amessage_event(self, fn):
        """Invoke an async function that produces a Message. Writes a MessageEvent."""
        if self.is_replay:
            event = self._read()
            if event is not None:
                if not isinstance(event, MessageEvent):
                    raise RuntimeError(f"Expected MessageEvent, got {type(event)}")
                return event.message
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        result = await fn()
        if not isinstance(result, Message):
            raise RuntimeError(f"Expected Message, got {type(result)}")
        self._write(MessageEvent(message=result))
        return result

    def _call_event(self, fn_name: str, fn: Callable[[], T]) -> T:
        """Invoke a non-deterministic function. Writes a CallEvent."""
        if self.is_replay:
            event = self._read()
            if event is not None:
                if not isinstance(event, CallEvent):
                    raise RuntimeError(f"Expected CallEvent, got {type(event)}")
                if event.fn_name != fn_name:
                    raise RuntimeError(f"Expected {fn_name}, got {event.fn_name}")
                return json.loads(event.result)
            if not self.continue_live:
                raise RuntimeError("Replay exhausted")
        result = fn()
        self._write(CallEvent(fn_name=fn_name, result=json.dumps(result)))
        return result

    # --- built-in env methods ---

    def call_tool(self, tool_call: ToolCall) -> str:
        tool_fn_name = f"tool.{tool_call.name}"
        tool_method = getattr(self, tool_fn_name, None)
        if tool_method is None:
            raise RuntimeError(f"Tool {tool_call.name} not registered.")
        args = json.loads(tool_call.arguments)
        msg = tool_method(**args)
        return msg.content

    def add_message(self, role: str, content: str | None = None, **kwargs) -> Message:
        msg = Message(role=role, content=content, **kwargs)
        return self._message_event(fn=lambda: msg)

    def add_user_message(self, text: str) -> str | None:
        return self.add_message(role="user", content=text).content

    # --- nondet / register ---

    def nondet(self, fn: Callable[..., T]) -> Callable[..., T]:
        fn_name = fn.__name__

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            return self._call_event(fn_name=fn_name, fn=lambda: fn(*args, **kwargs))

        return wrapper

    def _register_method(self, name: str, wrapper: Callable):
        self.registered_fns[name] = wrapper
        setattr(self, name, MethodType(wrapper, self))

    def register_llm_fn(self, fn: Callable[..., Message], name: str = "llm_complete"):
        def call_llm(obj, *args, **kwargs):
            return obj._message_event(
                fn=lambda: fn(*args, **kwargs),
            )

        self._register_method(name, call_llm)

    def register_llm_afn(self, fn, name: str = "llm_acomplete"):
        async def call_llm(obj, *args, **kwargs):
            return await obj._amessage_event(
                fn=lambda: fn(*args, **kwargs),
            )

        self._register_method(name, call_llm)

    def register_llm_stream_fn(self, fn: Callable, name: str = "llm_stream"):
        def method(obj, messages, **kwargs):
            content_parts = []
            for delta in fn(messages, **kwargs):
                content_parts.append(delta.content)
                yield delta

            obj._message_event(fn=lambda: Message(role="assistant", content="".join(content_parts)))

        self._register_method(name, method)

    def register_llm_astream_fn(self, fn, name: str = "llm_stream"):
        async def method(obj, messages, **kwargs):
            content_parts = []
            async for delta in fn(messages, **kwargs):
                content_parts.append(delta.content)
                yield delta

            await obj._amessage_event(
                fn=lambda: Message(role="assistant", content="".join(content_parts))
            )

        self._register_method(name, method)

    def register_nondet(self, fn: Callable[..., T], name: str | None = None):
        name = name or fn.__name__
        self._register_method(
            name,
            lambda obj, *args, **kwargs: obj._call_event(
                fn_name=name,
                fn=lambda: fn(*args, **kwargs),
            ),
        )

    def register_input_fn(self, fn: Callable[..., str], name: str = "input"):
        def wrapper(*args, **kwargs):
            result = fn(*args, **kwargs)
            if isinstance(result, TransientEvent):
                return result
            return Message(role="user", content=result)

        def method(obj, *args, **kwargs):
            result = obj._message_event(fn=lambda: wrapper(*args, **kwargs))
            if isinstance(result, TransientEvent):
                return result.value
            return result.content

        self._register_method(name, method)

    def register_tool_fns(self, tools: list[Tool]):
        for tool in tools:
            fn_name = f"tool.{tool.name}"

            def make_method(f, n):
                return lambda obj, **kwargs: obj._message_event(
                    fn=lambda: Message(role="tool", content=f(**kwargs)),
                )

            self._register_method(fn_name, make_method(tool.fn, fn_name))

    def rewind(self, continue_live: bool | None = None):
        if continue_live is not None:
            self.continue_live = continue_live
        self.child_index = 0
        self.read_head = ReadHead(self.full_fork_history())

    def go_live(self):
        """Switch to live mode from the current node.
        If the read_head was in the middle of replay, previous nodes will be orphaned.
        """
        self.replay_stop_predicate = lambda node: True

    def replay_until(self, fn: Callable[[EventNode], bool]):
        self.replay_stop_predicate = fn

    def print_tree(self):
        """Print the event log as a tree."""
        history = self.full_fork_history()
        self._print_tree_recursive(list(history.iter_nodes()), "", True)

    def _print_tree_recursive(self, nodes: list, prefix: str, is_last: bool):
        for i, node in enumerate(nodes):
            node_is_last = i == len(nodes) - 1
            connector = "└── " if node_is_last else "├── "
            new_prefix = prefix + ("    " if node_is_last else "│   ")

            if isinstance(node.event, MessageEvent):
                msg = node.event.message
                if msg.content and len(msg.content) > 50:
                    content: str | None = msg.content[:50] + "..."
                else:
                    content = msg.content
                print(f"{prefix}{connector}[{msg.role}] {content}")

            child_forks = self.forks.get(node.id, [])
            for j, fork_env in enumerate(child_forks):
                fork_is_last = j == len(child_forks) - 1
                fork_connector = "└── " if fork_is_last else "├── "
                print(f"{new_prefix}{fork_connector}Fork {j + 1}")
                fork_prefix = new_prefix + ("    " if fork_is_last else "│   ")
                fork_nodes = list(fork_env.full_fork_history().iter_nodes())
                fork_env._print_tree_recursive(fork_nodes, fork_prefix, fork_is_last)
