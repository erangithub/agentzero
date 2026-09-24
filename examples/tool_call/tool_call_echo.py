from agentzero import ToolCall, build_context
from agentzero.environment import Environment, Tool
from agentzero.mock import ToolCallLLM, tool_schemas


def main():
    def get_weather_tool(location: str) -> str:
        return f"25c and sunny in {location}"

    tools = [Tool(name="get_weather", fn=get_weather_tool, schema=tool_schemas[0])]

    llm = ToolCallLLM(
        tool_name="get_weather",
        arguments={"location": "london"},
        final_answer="The weather in London is 25c and sunny.",
    )

    env = Environment(continue_live=True)
    env.register_llm_fn(llm.complete)
    env.register_input_fn(input)
    env.register_tool_fns(tools)
    env.add_user_message("what's the weather in london?")
    msg = env.add_message(
        role="assistant",
        content=None,
        tool_calls=(ToolCall(id="1", name="get_weather", arguments='{"location": "london"}'),),
    )
    for tc in msg.tool_calls:
        env.call_tool(tc)
    env.add_message(role="assistant", content="The weather in London is 25c and sunny.")
    env.rewind()

    env.input()  # replays user message
    while True:
        context = build_context(env.history())
        response = env.llm_complete(context)

        if response.tool_calls:
            for tool_call in response.tool_calls:
                env.call_tool(tool_call)
        else:
            break

    for msg in env.history().iter_messages():
        print(f"[{msg.role}] {msg.content or msg.tool_calls}")


if __name__ == "__main__":
    main()
