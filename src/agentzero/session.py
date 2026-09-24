from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from agentzero.eventlog import (
    CallEvent,
    ControlEvent,
    ControlKind,
    Event,
    EventNode,
    Message,
    MessageEvent,
    ToolCall,
)

if TYPE_CHECKING:
    from agentzero.environment import Environment


class Session:
    """Shared container for every ``Environment`` of a run.

    Single source of truth for the parent->child topology (which an environment's
    heads only capture implicitly): each environment is registered by its origin
    (``branch_start``) node id. Registration is thread-safe; the rest of the
    framework assumes at most one thread per ``Environment``.

    A ``Session`` is the single way to create an ``Environment``: it owns a fresh
    root environment (``session.root``), and branches are created via ``fork()``.
    """

    def __init__(
        self,
        llm=None,
        input_fn=None,
        continue_live: bool = False,
        _make_root: bool = True,
    ):
        from agentzero.environment import Environment

        self._lock = threading.Lock()
        self._envs: dict[str, Environment] = {}
        self.root = None
        if _make_root:
            root = Environment(self, continue_live=continue_live)
            if llm is not None:
                root.register_llm_fn(llm.complete)
                if hasattr(llm, "acomplete"):
                    root.register_llm_afn(llm.acomplete)
                if hasattr(llm, "stream"):
                    root.register_llm_stream_fn(llm.stream)
            if input_fn is not None:
                root.register_input_fn(input_fn)
            self.root = root

    def register(self, env: Environment) -> None:
        with self._lock:
            self._envs[env.origin_node.id] = env

    def __getitem__(self, origin_id: str) -> Environment:
        with self._lock:
            return self._envs[origin_id]

    def envs(self) -> list[Environment]:
        with self._lock:
            return list(self._envs.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._envs)

    # --- JSON serialization (language-agnostic JSONL; DB persistence comes later) ---

    def to_json(self) -> str:
        envs = self.envs()

        nodes: dict[str, EventNode] = {}
        for env in envs:
            node = env.write_head.prev
            assert node is not None
            while node is not None:
                nodes[node.id] = node
                if node is env.origin_node:
                    break
                assert node.parent is not None or node is env.origin_node
                node = node.parent

        lines = [
            json.dumps(
                {
                    "typ": "header",
                    "version": 1,
                    "root": self.root.origin_node.id if self.root is not None else None,
                }
            )
        ]
        for node in sorted(nodes.values(), key=lambda n: (n.depth, n.id)):
            lines.append(json.dumps(_node_record(node)))
        for env in envs:
            tail = env.write_head.prev
            assert tail is not None
            lines.append(
                json.dumps(
                    {
                        "typ": "env",
                        "origin": env.origin_node.id,
                        "tail": tail.id,
                    }
                )
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_json(cls, text: str, continue_live: bool = True) -> Session:
        header: dict | None = None
        node_records: list[dict] = []
        env_records: list[dict] = []
        for line in text.splitlines():
            if not line.strip():
                continue
            record: dict = json.loads(line)
            typ = record["typ"]
            if typ == "header":
                header = record
            elif typ == "node":
                node_records.append(record)
            elif typ == "env":
                env_records.append(record)
            else:
                raise ValueError(f"Unknown record type {typ!r}")

        nodes: dict[str, EventNode] = {}
        for record in sorted(node_records, key=lambda r: (r["depth"], r["id"])):
            parent_id = record["parent"]
            nodes[record["id"]] = EventNode(
                id=record["id"],
                event=_event_from_record(record["event"]),
                parent=nodes[parent_id] if parent_id else None,
                depth=record["depth"],
            )

        from agentzero.environment import Environment

        session = cls(continue_live=continue_live, _make_root=False)
        envs: list[Environment] = []
        for record in env_records:
            env = Environment(
                session,
                origin_node=nodes[record["origin"]],
                continue_live=continue_live,
            )
            env.write_head.prev = nodes[record["tail"]]
            env.rewind()
            envs.append(env)

        roots = [env for env in envs if env.origin_node.parent is None]
        if header is not None and header.get("root") is not None:
            roots = [env for env in envs if env.origin_node.id == header["root"]]
        if roots:
            session.root = roots[0]

        chain_ids: dict[Environment, set[str]] = {
            env: {node.id for node in _chain(env)} for env in envs
        }
        for env in envs:
            fork_point = env.origin_node.parent
            if fork_point is None:
                continue
            owner = max(
                (candidate for candidate in envs if fork_point.id in chain_ids[candidate]),
                key=lambda candidate: candidate.origin_node.depth,
            )
            owner.forks.setdefault(fork_point.id, []).append(env)

        return session


def _chain(env: Environment) -> list[EventNode]:
    node = env.write_head.prev
    assert node is not None
    chain: list[EventNode] = []
    while node is not None:
        chain.append(node)
        if node is env.origin_node:
            break
        assert node.parent is not None
        node = node.parent
    return chain


def _node_record(node: EventNode) -> dict:
    return {
        "typ": "node",
        "id": node.id,
        "parent": node.parent.id if node.parent else None,
        "depth": node.depth,
        "event": _event_record(node.event),
    }


def _event_record(event: Event) -> dict:
    if isinstance(event, MessageEvent):
        message = event.message
        return {
            "kind": "message",
            "role": message.role,
            "content": message.content,
            "tool_calls": [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in message.tool_calls
            ],
            "tool_call_id": message.tool_call_id,
            "ts": event.timestamp,
            "ms": event.duration_ms,
        }
    if isinstance(event, CallEvent):
        return {
            "kind": "call",
            "fn_name": event.fn_name,
            "args": event.args,
            "result": event.result,
            "ts": event.timestamp,
            "ms": event.duration_ms,
        }
    if isinstance(event, ControlEvent):
        return {"kind": "control", "control": event.control.value}
    raise TypeError(f"Cannot serialize event {type(event).__name__}")


def _event_from_record(record: dict) -> Event:
    kind = record["kind"]
    if kind == "message":
        tool_calls = tuple(ToolCall(**tc) for tc in record["tool_calls"])
        return MessageEvent(
            message=Message(
                role=record["role"],
                content=record["content"],
                tool_calls=tool_calls,
                tool_call_id=record["tool_call_id"],
            ),
            timestamp=record["ts"],
            duration_ms=record["ms"],
        )
    if kind == "call":
        return CallEvent(
            fn_name=record["fn_name"],
            args=record["args"],
            result=record["result"],
            timestamp=record["ts"],
            duration_ms=record["ms"],
        )
    if kind == "control":
        return ControlEvent(control=ControlKind(record["control"]))
    raise ValueError(f"Unknown event kind {kind!r}")
