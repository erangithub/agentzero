import os
from agentzero import build_context
from agentzero.llms import GeminiLLM
from agentzero.environment import Environment, Tool
from agentzero.mock import tool_schemas


def get_weather(location: str) -> str:
    """Returns a fixed weather string. Stand-in for a real weather API."""
    return f"25c and sunny in {location}"


def main():
    llm = GeminiLLM(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    tools = [
        Tool(name="get_weather", fn=get_weather, schema=tool_schemas[0])
    ]

    env = Environment(llm=llm, continue_live=True)
    env.register_tool_fns(tools)
    env.register_input_fn(lambda: "Use the get_weather tool to tell me what the weather is in london.")
    env.input()

    context = build_context(env.history())
    # Gemini handles tool execution internally when tool_fns is passed.
    # AgentZero logs the result — no manual call_tool loop needed.
    env.llm_complete(context, tool_fns=[get_weather])

    for msg in env.history().iter_messages():
        print(f"[{msg.role}] {msg.content or msg.tool_calls}")


if __name__ == "__main__":
    main()
