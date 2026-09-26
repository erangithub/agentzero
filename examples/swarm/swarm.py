import asyncio

from agentzero import Session, build_context
from agentzero.llms import EchoLLM


async def main():
    # 1. Setup the shared session (its root env is where the story starts)
    #    Forks inherit the registered llm functions automatically
    env = Session(llm=EchoLLM(), continue_live=True).root

    # 2. Add the Initial "Context" (The Trunk)
    env.add_message("user", "Analyze the impact of remote work on urban planning.")

    # 3. The "Map" Phase: Create explicit branches
    # We create them sequentially to lock in the log order
    fork_econ = env.fork()
    fork_social = env.fork()
    fork_infra = env.fork()

    print("\n--- Running specialized analyses in parallel ---")

    async def get_perspective(fork, system_msg):
        ctx = build_context(env.history(), system=system_msg)
        return await fork.llm_acomplete(ctx)

    # Launch parallel tasks
    results = await asyncio.gather(
        get_perspective(fork_econ, "Focus on economic shifts and tax revenue."),
        get_perspective(fork_social, "Focus on social isolation and community building."),
        get_perspective(fork_infra, "Focus on public transit and office space conversion."),
    )

    # 4. Organize Results (The Dictionary Pattern)
    perspectives = {
        "Economics": results[0].content,
        "Social": results[1].content,
        "Infrastructure": results[2].content,
    }

    # 5. The "Reduce" Phase: Join them back in the Trunk
    print("\n--- Joining Results for Synthesis ---")

    # Build a prompt that references the specific branch outputs
    join_content = "Synthesize these three perspectives into a 3-point strategy:\n\n"
    for label, text in perspectives.items():
        join_content += f"### {label} Analysis\n{text}\n\n"

    # We inject this into the main environment's context
    final_ctx = build_context(env.history(), injections=[join_content])
    final_report = await env.llm_acomplete(final_ctx)

    print("\n=== Final Synthesis ===")
    print(final_report.content)

    # 6. Inspect the Tree
    # This shows the one-to-many relationship clearly (session-wide view,
    # git-log style: flat for linear runs, connectors where the log branches)
    print("\n=== Event Log Structure ===")
    env.session.print_tree()


if __name__ == "__main__":
    asyncio.run(main())
