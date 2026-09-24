from agentzero import Session, build_context
from agentzero.llms import EchoLLM


def _make_env():
    llm = EchoLLM()
    env = Session(continue_live=True).root
    env.register_llm_fn(llm.complete)
    env.register_input_fn(lambda: "")
    return env


def _run_flow(env):
    """Mirrors examples/swarm/branch_summarize.py flow()."""
    env.add_user_message("long conversation here...")
    summary = _summarize(env)
    context = build_context(env.history(), injections=[summary])
    response = env.llm_complete(context)
    return summary, response.content


def _summarize(env):
    sub = env.fork()
    context = build_context(sub.history(), system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return f"Summary: {response.content}"


def test_fork_does_not_touch_parent_log():
    env = _make_env()
    env.add_user_message("long conversation here...")
    parent_len_before = len(list(env.history().iter_messages()))

    summary = _summarize(env)

    assert summary == "Summary: echo: long conversation here..."
    assert len(list(env.history().iter_messages())) == parent_len_before


def test_build_context_injections_and_system():
    env = _make_env()
    env.add_user_message("long conversation here...")

    summary = _summarize(env)
    context = build_context(env.history(), injections=[summary])

    assert context[0].role == "system"
    assert context[0].content == summary
    assert context[1].role == "user"
    assert context[1].content == "long conversation here..."

    final = env.llm_complete(context)
    assert final.content == "echo: long conversation here..."


def test_fork_flow_replays_identically():
    env = _make_env()

    first = _run_flow(env)

    env.rewind()
    second = _run_flow(env)

    assert first == second
    assert first == (
        "Summary: echo: long conversation here...",
        "echo: long conversation here...",
    )
