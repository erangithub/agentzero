from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

from agentzero.eventlog import (
    CallEvent,
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

    Owns the single event DAG that all environments are cursors over, and the
    parent->child cursor topology (which the log itself no longer records).
    Each environment is registered by its own ``id``, since cursors are no
    longer identified by a unique branch node — several may share a fork point.
    Registration is thread-safe; the rest of the framework assumes at most one
    thread per ``Environment``.

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
        # Every node ever created in this session, including nodes orphaned by a
        # rewind+go_live. Kept so the full branch structure stays inspectable;
        # to_json still persists only nodes reachable from an env's heads.
        self._nodes: dict[str, EventNode] = {}
        self.root = None
        if _make_root:
            root = Environment(self, continue_live=continue_live)
            if llm is not None:
                _register_llm(root, llm)
            if input_fn is not None:
                root.register_input_fn(input_fn)
            self.root = root

    def register(self, env: Environment) -> None:
        with self._lock:
            self._envs[env.id] = env

    def record_node(self, node: EventNode) -> None:
        with self._lock:
            self._nodes[node.id] = node

    def __getitem__(self, env_id: str) -> Environment:
        with self._lock:
            return self._envs[env_id]

    def envs(self) -> list[Environment]:
        with self._lock:
            return list(self._envs.values())

    def __len__(self) -> int:
        with self._lock:
            return len(self._envs)

    def print_tree(self):
        """Print the whole event log as a forest of branches.

        This is the session-wide view: it walks the shared DAG from every root,
        so branches orphaned by ``rewind()`` + ``go_live()`` are shown too. A
        branch with no live cursor on it was abandoned — its nodes are still in
        the log but no environment's heads point at them, so they no longer
        affect replay. That is the visible side effect of undo/branching in a
        live session, and the reason this lives on the session rather than an
        environment: an environment can only reach its own live history and the
        forks cut from it, never the abandoned branches.

        Rendered as a single tree, oldest event first: a non-last sibling opens a
        new column with ``├──``, while the last sibling stays in its parent's
        column with no connector.
        """
        with self._lock:
            nodes = list(self._nodes.values())
        if not nodes:
            print("(empty log)")
            return

        # Build parent -> children so we can walk the DAG (each node's event
        # links to the next event written after it). Walking up from any node
        # reaches a root, and every root is drawn into the one tree.
        children: dict[str | None, list[EventNode]] = {}
        for node in nodes:
            key = node.parent.id if node.parent is not None else None
            children.setdefault(key, []).append(node)
        for kids in children.values():
            kids.sort(key=lambda n: (n.depth, n.id))

        # Mark where each env's cursor is. The cursor belongs on the line of the
        # event it will produce next: for a rewound env that is the next node it
        # will read, so the cursor prefixes that node's row. Only once an env has
        # caught up with its own tail is there no next event, and the cursor
        # stands alone on the empty line after the last one.
        on_row: set[str] = set()
        after: set[str] = set()
        for env in self.envs():
            nxt = env.next_node
            if nxt is not None:
                on_row.add(nxt.id)
            else:
                tail = env.prev_node
                if tail is not None:
                    after.add(tail.id)

        for line in _tree_lines(children.get(None, []), children, on_row, after):
            print(line)

    # --- JSON serialization (language-agnostic JSONL; DB persistence comes later) ---

    def to_json(self) -> str:
        envs = self.envs()

        # Walk each cursor's tail up to the head of the shared DAG. The union is
        # the whole log; shared ancestors are collected once.
        nodes: dict[str, EventNode] = {}
        for env in envs:
            node = env.write_head.prev
            while node is not None:
                nodes[node.id] = node
                node = node.parent

        lines = [
            json.dumps(
                {
                    "typ": "header",
                    "version": 1,
                    "root": self.root.id if self.root is not None else None,
                }
            )
        ]
        for node in sorted(nodes.values(), key=lambda n: (n.depth, n.id)):
            lines.append(json.dumps(_node_record(node)))
        for env in envs:
            tail = env.write_head.prev
            fork_point = env.fork_point
            lines.append(
                json.dumps(
                    {
                        "typ": "env",
                        "id": env.id,
                        "fork_point": fork_point.id if fork_point is not None else None,
                        "tail": tail.id if tail is not None else None,
                        "parent": env.parent_id,
                    }
                )
            )
        return "\n".join(lines) + "\n"

    @classmethod
    def from_json(
        cls,
        text: str,
        continue_live: bool = True,
        llm=None,
        input_fn=None,
    ) -> Session:
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
        for node in nodes.values():
            session.record_node(node)
        envs: list[Environment] = []
        for record in env_records:
            fork_point_id = record["fork_point"]
            tail_id = record["tail"]
            env = Environment(
                session,
                continue_live=continue_live,
                fork_point=nodes[fork_point_id] if fork_point_id else None,
                env_id=record["id"],
                parent_id=record.get("parent"),
            )
            if tail_id:
                env.write_head.prev = nodes[tail_id]
            env.rewind()
            envs.append(env)

        if header is not None and header.get("root") is not None:
            session.root = session[header["root"]]
        else:
            roots = [env for env in envs if env.parent_id is None]
            if roots:
                session.root = roots[0]

        for env in envs:
            if llm is not None:
                _register_llm(env, llm)
            if input_fn is not None:
                env.register_input_fn(input_fn)

        # Rebuild the cursor topology from the recorded parent links; the log
        # itself carries no branch structure.
        for env in envs:
            if env.parent_id is None:
                continue
            key = env.fork_point.id if env.fork_point is not None else None
            session[env.parent_id].forks.setdefault(key, []).append(env)

        return session


def _register_llm(env, llm) -> None:
    env.register_llm_fn(llm.complete)
    if hasattr(llm, "acomplete"):
        env.register_llm_afn(llm.acomplete)
    if hasattr(llm, "stream"):
        env.register_llm_stream_fn(llm.stream)


def _tree_lines(roots, children, on_row, after) -> list[str]:
    """Lay the log out as one tree, oldest event first.

    A non-last sibling opens a new column and is drawn with ``├──``; the *last*
    sibling never opens a new column and gets no connector at all, so it lands
    back in its parent's column. A node only hands a ``│  `` rail down to its
    children when it drew ``├──`` -- which is why linear runs and last-sibling
    subtrees both stay in the column they started in. There is no ``└──``: a
    branch that ends simply stops.

    Roots are the top of the tree, so a log with a single root opens on a bare
    ``*``. Rewinding to the start and writing orphans the old log, and a session
    can then genuinely have more than one root; those are drawn as extra
    top-level branches of the same tree rather than as separate ones. Their
    ``parent`` is still ``None`` in the DAG -- they are only *drawn* side by
    side, never re-parented.

    Every row carries a two-character left margin. ``>`` marks the row of the
    event an env will produce next; when an env has caught up with its own tail
    there is no next event, so the cursor is drawn alone on the empty line just
    after the node it follows.
    """
    out: list[str] = []

    def walk(node, base: str, connector: str) -> None:
        kids = children.get(node.id, [])
        margin = ">" if node.id in on_row else " "
        out.append(f"{margin} {base}{connector}* {_node_label(node)}")

        child_base = base + ("│  " if connector == "├──" else "")
        if node.id in after:
            out.append(f"> {child_base}*")
        last = len(kids) - 1
        for i, kid in enumerate(kids):
            walk(kid, child_base, "" if i == last else "├──")

    last = len(roots) - 1
    for i, root in enumerate(roots):
        walk(root, "", "" if i == last else "├──")
    return out


def _node_label(node: EventNode) -> str:
    if isinstance(node.event, MessageEvent):
        msg = node.event.message
        content = msg.content
        if content and len(content) > 50:
            content = content[:50] + "..."
        return f"[{msg.role}] {content or ''}".rstrip()
    if isinstance(node.event, CallEvent):
        return f"[call:{node.event.fn_name}]"
    return f"[{type(node.event).__name__}]"


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
    raise ValueError(f"Unknown event kind {kind!r}")
