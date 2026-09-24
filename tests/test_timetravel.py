from agentzero import build_context
from agentzero.environment import Environment, transient
from agentzero.llms import EchoLLM


def prev_user_depth(env) -> int | None:
    node = env.prev_node.find(lambda n: n.is_message("user"))
    return node.depth if node else None


def run_session(env, commands):
    """Mirrors examples/timetravel/delorean.py main loop with scripted input.

    Commands ('b', 'n', 'quit', 'reset') are returned as ``transient`` so they
    never enter the log — matching the example's ``get_input``.
    """
    inputs = list(commands)

    def get_input(depth=None):
        if not inputs:
            return transient("quit")
        value = inputs.pop(0)
        if value in ("b", "n", "quit", "reset"):
            return transient(value)
        return value

    env.register_input_fn(get_input)

    while True:
        user_input = env.input(env.current_depth)

        if user_input == "quit":
            break

        if user_input == "b":
            target_depth = prev_user_depth(env)
            env.rewind()
            env.replay_until(lambda n, d=target_depth: n.depth >= d)
            continue

        if user_input == "n":
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
    env = Environment(llm=EchoLLM(), continue_live=True)
    run_session(env, ["apple", "b", "banana", "n", "quit"])

    # Only the kept exchange survives — the apple exchange is discarded.
    assert messages(env) == [("user", "banana"), ("assistant", "echo: banana")]


def test_replay_of_kept_log_is_deterministic():
    env = Environment(llm=EchoLLM(), continue_live=True)
    run_session(env, ["apple", "b", "banana", "quit"])

    kept = messages(env)
    assert kept == [("user", "banana"), ("assistant", "echo: banana")]

    # Replaying the kept log re-executes each exchange with no live input.
    env.rewind()
    replayed = []
    while True:
        user_input = env.input()
        if user_input == "quit":
            break
        assert user_input == "banana"
        response = env.llm_complete(build_context(env.history()))
        assert response.content == "echo: banana"
        replayed.append((response.role, response.content))

    assert replayed == [("assistant", "echo: banana")]
    # Replay is read-only: the log is unchanged.
    assert messages(env) == kept
