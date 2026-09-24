import os

from agentzero import Session, build_context
from agentzero.llms import GeminiLLM


def get_weather(location: str) -> str:
    """Returns a fixed weather string. Stand-in for a real weather API."""
    return f"25c and sunny in {location}"


def main():
    llm = GeminiLLM(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    env = Session(
        llm=llm,
        input_fn=lambda: "Use the get_weather tool to tell me what the weather is in london.",
        continue_live=True,
    ).root
    env.register_tool_fns([get_weather])
    env.input()

    context = build_context(env.history())
    # Gemini handles tool execution internally when tool_fns is passed.
    # AgentZero logs the result — no manual call_tool loop needed.
    env.llm_complete(context, tool_fns=[get_weather])

    for msg in env.history().iter_messages():
        print(f"[{msg.role}] {msg.content or msg.tool_calls}")


if __name__ == "__main__":
    main()
