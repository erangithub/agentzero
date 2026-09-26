import asyncio

from agentzero import Session, build_context
from agentzero.llms import EchoLLM


def test_async_parallel_forks():
    """Mirrors examples/swarm/swarm.py — parallel fork + reduce back to trunk."""

    async def main():
        env = Session(continue_live=True).root
        env.register_llm_fn(EchoLLM().complete)
        env.register_llm_afn(EchoLLM().acomplete)
        env.add_message("user", "Analyze the impact of remote work on urban planning.")

        # Map: create branches sequentially to lock in the log order
        fork_econ = env.fork()
        fork_social = env.fork()
        fork_infra = env.fork()

        async def get_perspective(fork, system_msg):
            ctx = build_context(env.history(), system=system_msg)
            return await fork.llm_acomplete(ctx)

        results = await asyncio.gather(
            get_perspective(fork_econ, "Focus on economic shifts and tax revenue."),
            get_perspective(fork_social, "Focus on social isolation and community building."),
            get_perspective(fork_infra, "Focus on public transit and office space conversion."),
        )

        expected = "echo: Analyze the impact of remote work on urban planning."
        assert [r.content for r in results] == [expected, expected, expected]

        # Each branch recorded its own assistant response in its own log
        for fork in (fork_econ, fork_social, fork_infra):
            msgs = list(fork.history().iter_messages())
            assert msgs[-1].role == "assistant"
            assert msgs[-1].content == expected

        # Reduce: join branch outputs back in the trunk
        join_content = "Synthesize these three perspectives into a 3-point strategy:\n\n"
        for label, text in (
            ("Economics", results[0].content),
            ("Social", results[1].content),
            ("Infrastructure", results[2].content),
        ):
            join_content += f"### {label} Analysis\n{text}\n\n"

        final_ctx = build_context(env.history(), injections=[join_content])
        final_report = await env.llm_acomplete(final_ctx)
        assert final_report.content == expected

        # The trunk log only ever contained user + its own assistant reply
        parent_roles = [m.role for m in env.history().iter_messages()]
        assert parent_roles == ["user", "assistant"]

    asyncio.run(main())


def test_print_tree_runs_on_forked_log():
    """Mirrors the 'Inspect the Tree' step of examples/swarm/swarm.py."""
    env = Session(continue_live=True).root
    env.register_llm_fn(EchoLLM().complete)
    env.add_message("user", "hello")
    fork_a = env.fork()
    fork_a.llm_complete(build_context(env.history(), system="Branch A"))

    env.session.print_tree()  # should not raise on a log with forks
