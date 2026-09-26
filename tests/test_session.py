from concurrent.futures import ThreadPoolExecutor

from agentzero import Session, build_context
from agentzero.environment import transient
from agentzero.llms import EchoLLM


def _make_env():
    llm = EchoLLM()
    env = Session(continue_live=True).root
    env.register_llm_fn(llm.complete)
    env.register_input_fn(lambda: "")
    return env


def _flow(env):
    """Mirrors examples/swarm/branch_summarize.py flow() with a summary fork."""
    env.add_user_message("long conversation here...")
    sub = env.fork()
    response = sub.llm_complete(build_context(sub.history(), system="Summarize."))
    context = build_context(env.history(), injections=[f"Summary: {response.content}"])
    final = env.llm_complete(context)
    return final.content


def _messages(env):
    # Tail-based: matches what the log actually holds regardless of read-head state.
    return [(m.role, m.content) for m in env.full_history().iter_messages()]


def test_json_roundtrip_restores_messages_and_forks():
    env = _make_env()
    result = _flow(env)
    assert result == "echo: long conversation here..."

    session = env.session
    payload = session.to_json()
    loaded = Session.from_json(payload)

    root = loaded.root
    assert root is not None
    assert len(loaded) == len(session)
    assert _messages(root) == _messages(env)


def test_loaded_root_replays_deterministically():
    env = _make_env()
    first = _flow(env)
    payload = env.session.to_json()

    for _ in range(2):
        loaded = Session.from_json(payload)
        for loaded_env in loaded.envs():
            loaded_env.register_llm_fn(EchoLLM().complete)
            loaded_env.register_input_fn(lambda: "")
        loaded_root = loaded.root
        assert loaded_root is not None
        second = _flow(loaded_root)
        assert second == first


def test_fork_is_reused_after_load_not_recreated():
    env = _make_env()
    env.add_user_message("long conversation here...")
    original = env.fork()
    original.llm_complete(build_context(original.history(), system="Summarize."))
    fork_point = original.fork_point
    assert fork_point is not None

    loaded = Session.from_json(env.session.to_json())
    loaded_root = loaded.root
    assert fork_point.id in loaded_root.forks
    reused = loaded_root.forks[fork_point.id][0]

    assert reused.id == original.id
    assert reused.fork_point.id == original.fork_point.id
    assert loaded_root.fork() is reused


def test_orphaned_nodes_are_not_restored():
    """Rolled-back exchanges (delorean 'b') stay gone after load."""
    env = Session(continue_live=True).root
    env.register_llm_fn(EchoLLM().complete)
    inputs = ["apple", "b", "banana", "quit"]

    def get_input(depth=None):
        if not inputs:
            return transient("quit")
        value = inputs.pop(0)
        return transient(value) if value in ("b", "quit") else value

    env.register_input_fn(get_input)
    while True:
        user_input = env.input(env.current_depth)
        if user_input == "quit":
            break
        if user_input == "b":
            target = None
            node = env.prev_node.find(lambda n: n.is_message("user"))
            if node:
                target = node.depth
            assert target is not None
            env.rewind()
            env.replay_until(lambda n, d=target: n.depth >= d)
            continue
        env.llm_complete(build_context(env.history()))

    assert _messages(env) == [("user", "banana"), ("assistant", "echo: banana")]

    loaded_root = Session.from_json(env.session.to_json()).root
    assert _messages(loaded_root) == [("user", "banana"), ("assistant", "echo: banana")]


def _run_isolated_session() -> list[tuple[str, str | None]]:
    """Each thread owns its own session and env — one thread per env."""
    env = Session(continue_live=True).root
    env.register_llm_fn(EchoLLM().complete)
    env.register_input_fn(lambda: "hello")
    env.input()
    env.llm_complete(build_context(env.history()))
    return [(m.role, m.content) for m in env.full_history().iter_messages()]


def test_isolated_sessions_are_thread_safe():
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _run_isolated_session(), range(32)))

    expected = [("user", "hello"), ("assistant", "echo: hello")]
    assert all(messages == expected for messages in results)
