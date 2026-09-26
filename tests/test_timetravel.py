import asyncio

from agentzero import Message, Session, build_context
from agentzero.environment import transient
from agentzero.llms import EchoLLM


def prev_user_depth(env) -> int | None:
    node = env.prev_node.find(lambda n: n.is_message("user"))
    return node.depth if node else None


def run_session(env, commands):
    """Mirrors examples/timetravel/delorean.py main loop with scripted input.

    Slash commands ('/b', '/n', '/quit', '/reset') are returned as ``transient``
    so they never enter the log — matching the example's ``get_input``.
    """
    inputs = list(commands)

    def get_input(depth=None):
        if not inputs:
            return transient("/quit")
        value = inputs.pop(0)
        if value.startswith("/"):
            return transient(value)
        return value

    env.register_input_fn(get_input)

    while True:
        user_input = env.input(env.current_depth)

        if user_input == "/quit":
            break

        if user_input == "/b":
            target_depth = prev_user_depth(env)
            env.rewind()
            env.replay_until(lambda n, d=target_depth: n.depth >= d)
            continue

        if user_input == "/n":
            current_depth = env.current_depth
            env.replay_until(lambda n, d=current_depth: n.is_message("user") and n.depth > d)
            continue

        env.llm_complete(build_context(env.history()))


def messages(env):
    return [(m.role, m.content) for m in env.history().iter_messages()]


def test_go_back_discards_rolled_back_exchange():
    """'b' rolls back the last exchange; the re-answer replaces it in the log.

    'n' (next-exchange) and 'quit' are also exercised here — command inputs are
    transient and never enter the log.
    """
    env = Session(continue_live=True).root
    env.register_llm_fn(EchoLLM().complete)
    run_session(env, ["apple", "/b", "banana", "/n", "/quit"])

    # Only the kept exchange survives — the apple exchange is discarded.
    assert messages(env) == [("user", "banana"), ("assistant", "echo: banana")]


def _two_exchanges():
    """A live env holding u1/a1/u2/a2."""
    env = Session(continue_live=True).root
    env.register_llm_fn(EchoLLM().complete)
    env.add_user_message("u1")
    env.llm_complete(build_context(env.history()))
    env.add_user_message("u2")
    env.llm_complete(build_context(env.history()))
    return env


def _rewind_parked_on_u2(env):
    """Rewind and replay forward so the read head is parked on u2 (go-live point)."""
    env.rewind()
    env.replay_until(lambda n: n.is_message("user") and n.event.message.content == "u2")
    env.add_user_message("u1")  # replays u1
    env.llm_complete(build_context(env.history()))  # replays a1
    assert not env.is_replay  # parked on u2, so the next write goes live here
    return env


def tail_messages(env):
    """Tail-based: reflects the true write position regardless of read-head state."""
    return [(m.role, m.content) for m in env.full_history().iter_messages()]


def test_nondet_call_going_live_anchors_at_read_head():
    """A nondeterministic write after going live must resume at the read head,
    orphaning the discarded tail — not append after it."""
    env = _two_exchanges()
    env.register_nondet(lambda: {"v": 1}, name="tick")
    _rewind_parked_on_u2(env)

    assert env.tick() == {"v": 1}

    # The tick node (a CallEvent) hangs off a1; u2/a2 are orphaned.
    assert tail_messages(env) == [("user", "u1"), ("assistant", "echo: u1")]


def test_async_message_going_live_anchors_at_read_head():
    """Same anchoring for an async message write."""
    env = _two_exchanges()
    _rewind_parked_on_u2(env)

    async def live_async(_context):
        return Message(role="assistant", content="async-live")

    env.register_llm_afn(live_async, name="llm_acomplete")
    asyncio.run(env.llm_acomplete(build_context(env.history(), system="s")))

    # u2/a2 are orphaned; the async message hangs off a1.
    assert tail_messages(env) == [
        ("user", "u1"),
        ("assistant", "echo: u1"),
        ("assistant", "async-live"),
    ]


def test_session_tree_shows_orphaned_branches(capsys):
    """The session tree must show branches abandoned by go_live, not just the
    live cursor's history — that's the visible side effect of an undo."""
    env = _two_exchanges()  # u1/a1/u2/a2
    env.rewind()
    # Park on u1, then go live: this writes a fresh root and orphans a1/u2/a2.
    env.replay_until(lambda n: n.is_message("user") and n.event.message.content == "u1")
    env.add_user_message("u3")
    env.llm_complete(build_context(env.history()))

    # The abandoned branch is no longer reachable from the env's write head...
    assert tail_messages(env) == [("user", "u3"), ("assistant", "echo: u3")]

    # ...but the session still holds it, and labels it orphaned.
    env.session.print_tree()
    out = capsys.readouterr().out
    assert "u1" in out and "u2" in out  # the discarded branch is present
    assert "(orphaned)" in out
    assert "> " in out  # the live cursor is marked in the left margin
    assert "u3" in out


def test_replay_of_kept_log_is_deterministic():
    env = Session(continue_live=True).root
    env.register_llm_fn(EchoLLM().complete)
    run_session(env, ["apple", "/b", "banana", "/quit"])

    kept = messages(env)
    assert kept == [("user", "banana"), ("assistant", "echo: banana")]

    # Replaying the kept log re-executes each exchange with no live input.
    env.rewind()
    replayed = []
    while True:
        user_input = env.input()
        if user_input == "/quit":
            break
        assert user_input == "banana"
        response = env.llm_complete(build_context(env.history()))
        assert response.content == "echo: banana"
        replayed.append((response.role, response.content))

    assert replayed == [("assistant", "echo: banana")]
    # Replay is read-only: the log is unchanged.
    assert messages(env) == kept
