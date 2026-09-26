from __future__ import annotations

import functools
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from types import MethodType
from typing import Any, TypeVar

from agentzero.eventlog import (
    CallEvent,
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
from agentzero.session import Session

T = TypeVar("T")


@dataclass
class Tool:
    name: str
    fn: Callable[..., str]
    schema: dict

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


def transient(value: Any):
    return TransientEvent(value=value)


class Environment:
    """A cursor over the shared session event log: a read head + write head.

    An environment holds no branch marker of its own. It is positioned by
    ``fork_point`` (the log node it branched from, ``None`` for the root) and
    moves by advancing its two heads. ``fork()`` places another cursor at the
    current write-head node without writing anything, so the log stays a single
    shared DAG in which the trunk is not structurally special.

    Environments are never constructed directly by users. They are created by a
    ``Session`` (``session.root``), by ``Session.from_json``, or by another
    environment's ``fork()``. The session is the single entry point.
    """

    def __init__(
        self,
        session: Session,
        continue_live: bool = False,
        fork_point: EventNode | None = None,
        registered_fns: dict[str, Callable] | None = None,
        env_id: str | None = None,
        parent_id: str | None = None,
    ):
        self.registered_fns: dict[str, Callable] = {}
        for name, fn in (registered_fns or {}).items():
            setattr(self, name, MethodType(fn, self))
            self.registered_fns[name] = fn

        self.id = env_id or str(uuid.uuid4())
        self.parent_id = parent_id
        self.fork_point = fork_point
        self.write_head = WriteHead(prev=fork_point)
        self.forks: dict[str | None, list[Environment]] = {}
        self.session = session
        session.register(self)

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
        node = self.write_head.prev
        if node is not None:
            self.session.record_node(node)
        self.child_index = 0

    def _read(self) -> MessageEvent | CallEvent | None:
        assert self.is_replay
        self.child_index = 0
        if self.read_head.next is None:
            return None
        self.read_head.step()
        prev = self.read_head.prev
        assert prev is not None
        assert isinstance(prev.event, (MessageEvent, CallEvent))
        return prev.event

    def _anchor_write_head(self) -> None:
        """Re-anchor the write head at the read head before a live write.

        If the read head is positioned anywhere — including at the very start of
        the log, where ``prev`` is None but ``next`` is set — move the write head
        to the read position and clear the read head, so the node is appended
        there and any remaining replay tail is orphaned. Then drop the replay
        stop predicate, since we are now genuinely live.

        Every primitive that appends a recorded event must call this first, so
        that sync/async message writes and nondeterministic call writes all
        resume at the same point and orphan the discarded tail consistently.
        """
        if self.read_head.prev is not None or self.read_head.next is not None:
            self.write_head.prev = self.read_head.prev
            self.read_head.prev = self.read_head.next = None
        self.replay_stop_predicate = None

    def _read_head_positioned(self) -> bool:
        """True once the read head has a place to sit in the log.

        A rewound env starts *before* the first node, where ``prev`` is None but
        ``next`` is not -- that is a real position, not an absent one. A live
        env never advances its read head at all, so both are None and the write
        head is the position.
        """
        return self.read_head.prev is not None or self.read_head.next is not None

    @property
    def prev_node(self) -> EventNode | None:
        """The node just before this env's cursor.

        None means the cursor sits at the very start of the log -- or that the
        env has no history at all, which is the same thing.
        """
        if self._read_head_positioned():
            return self.read_head.prev
        return self.write_head.prev

    @property
    def next_node(self) -> EventNode | None:
        """The node the cursor will read next, or None if not positioned.

        Together with ``prev_node`` this pins the cursor *between* two nodes.
        """
        return self.read_head.next

    @property
    def current_depth(self):
        node = self.prev_node
        return (node.depth + 1) if node is not None else 0

    def history(self) -> Sequence:
        return Sequence(after_node=None, to_node=self.prev_node)

    def full_history(self) -> Sequence:
        return Sequence(after_node=None, to_node=self.write_head.prev)

    def fork_history(self) -> Sequence:
        return Sequence(after_node=self.fork_point, to_node=self.prev_node)

    def full_fork_history(self) -> Sequence:
        return Sequence(after_node=self.fork_point, to_node=self.write_head.prev)

    def fork(self) -> Environment:
        """Place a new cursor on the shared log at the current write-head node.

        No event node is written: the child simply starts appending to the same
        node the parent is at, so the two cursors produce co-equal children of it.
        Forking the same node again returns the next existing sibling
        (deterministic reuse), creating one only when needed.
        """
        fork_point = self.write_head.prev
        key = fork_point.id if fork_point is not None else None
        child_envs = self.forks.setdefault(key, [])
        if self.child_index < len(child_envs):
            forked_env = child_envs[self.child_index]
            forked_env.rewind()
        else:
            forked_env = Environment(
                self.session,
                continue_live=self.continue_live,
                fork_point=fork_point,
                registered_fns=self.registered_fns,
                parent_id=self.id,
            )
            child_envs.append(forked_env)

        self.child_index += 1
        return forked_env

    # --- core invoke primitives ---

    def _message_event(self, fn: Callable[[], Message], expect: Message | None = None) -> Message:
        """Invoke a function that produces a Message. Writes a MessageEvent.

        If the function returns TransientEvent, it's returned without recording.

        ``expect`` is the message the caller means to append, when the caller
        built it up front. Replay never calls ``fn`` -- that is the whole point,
        since ``fn`` is usually the side-effecting work -- so ``expect`` is what
        lets us tell a faithful replay from a caller that has gone off-script.
        A caller that rewound, never called ``go_live``, and then tries to add
        something new is not replaying, it is writing into a replay; if the
        message it holds disagrees with the log we say so instead of quietly
        dropping it on the floor.
        """
        if self.is_replay:
            event = self._read()
            assert event is not None
            if not isinstance(event, MessageEvent):
                raise RuntimeError(f"Expected MessageEvent, got {type(event)}")
            recorded = event.message
            if expect is not None and (expect.role, expect.content) != (
                recorded.role,
                recorded.content,
            ):
                raise RuntimeError(
                    f"Not live: this env is replaying, so the next recorded message is "
                    f"{recorded.role}: {recorded.content!r}, but add_message was given "
                    f"{expect.role}: {expect.content!r}. Nothing was written. Call go_live() "
                    f"to branch from here, or let the replay run."
                )
            return recorded
        elif self._read_head_positioned() and not self.continue_live:
            raise RuntimeError("Replay exhausted")
        result = fn()
        if isinstance(result, TransientEvent):
            return result
        if not isinstance(result, Message):
            raise RuntimeError(f"Expected Message, got {type(result)}")

        # Sync write_head to read_head when going live with a real message, so
        # writes resume at the go-live node and the replayed tail is orphaned.
        self._anchor_write_head()

        self._write(MessageEvent(message=result))

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
        elif self._read_head_positioned() and not self.continue_live:
            raise RuntimeError("Replay exhausted")
        result = await fn()
        if not isinstance(result, Message):
            raise RuntimeError(f"Expected Message, got {type(result)}")
        self._anchor_write_head()
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
        elif self._read_head_positioned() and not self.continue_live:
            raise RuntimeError("Replay exhausted")
        result = fn()
        self._anchor_write_head()
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
        return self._message_event(fn=lambda: msg, expect=msg)

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

    def register_tool_fns(self, tools: list[Tool | Callable[..., Any]]):
        for item in tools:
            if isinstance(item, Tool):
                tool = item
            else:
                from agentzero.schema import schema_from_fn

                tool = Tool(name=item.__name__, fn=item, schema=schema_from_fn(item))

            fn_name = f"tool.{tool.name}"

            def make_method(f):
                return lambda obj, **kwargs: obj._message_event(
                    fn=lambda: Message(role="tool", content=f(**kwargs)),
                )

            self._register_method(fn_name, make_method(tool.fn))

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
