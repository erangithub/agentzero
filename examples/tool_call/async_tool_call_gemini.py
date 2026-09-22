import asyncio
import os
from agentzero import build_context
from agentzero.llms import GeminiLLM
from agentzero.environment import Environment, Tool


def get_weather(location: str) -> dict:
    """Returns weather info as JSON."""
    return {"temperature": "25c", "conditions": "sunny", "location": location}


async def main():
    llm = GeminiLLM(
        model="gemini-2.5-flash",
        api_key=os.environ.get("GEMINI_API_KEY"),
    )

    tools = [
        Tool(name="get_weather", fn=get_weather, schema={"type": "function", "function": {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}})
    ]

    env = Environment(continue_live=True)
    env.register_llm_afn(llm.acomplete)
    env.register_tool_fns(tools)
    env.register_input_fn(lambda: "What's the weather in London?")
    env.input()

    context = build_context(env.history())
    # Gemini handles tool execution internally when tool_fns is passed.
    # AgentZero logs the result — no manual call_tool loop needed.
    response = await env.llm_acomplete(context, tool_fns=[get_weather])

    print(f"Response: {response}")

    print("\n=== Full history ===")
    for msg in env.history().iter_messages():
        print(f"[{msg.role}] {msg.content or msg.tool_calls}")


if __name__ == "__main__":
    asyncio.run(main())
