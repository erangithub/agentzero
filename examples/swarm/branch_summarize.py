from agentzero import Session, build_context
from agentzero.llms import EchoLLM


def summarize(env) -> str:
    """Run summarization in a forked branch. Fork shares parent's store."""
    sub = env.fork()
    context = build_context(sub.history(), system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return f"Summary: {response.content}"


def flow(env):
    user_input = env.add_user_message("long conversation here...")
    print(f"User: {user_input}")

    summary = summarize(env)
    print(f"Flow: {summary}")

    context = build_context(env.history(), injections=[summary])
    response = env.llm_complete(context)
    print(f"Assistant: {response}")

    print("-" * 20)


def main():
    llm = EchoLLM()
    env = Session(llm=llm, input_fn=input, continue_live=True).root

    print("=== First run: replay + continue live ===")
    flow(env)

    print("=== Second run: replay from saved store ===")
    env.rewind()
    flow(env)


if __name__ == "__main__":
    main()
