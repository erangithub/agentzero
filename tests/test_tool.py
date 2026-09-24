from agentzero import Session, ToolCall, build_context, tool
from agentzero.environment import Tool
from agentzero.llms import EchoLLM


@tool
def get_weather(location: str, units: str = "c") -> str:
    """Returns the weather for a location."""
    return f"25c and sunny in {location}"


def test_tool_schema_inference():
    schema = get_weather.schema["function"]
    assert schema["name"] == "get_weather"
    assert schema["description"] == "Returns the weather for a location."

    params = schema["parameters"]
    assert params["properties"]["location"]["type"] == "string"
    assert params["properties"]["units"]["type"] == "string"
    assert params["properties"]["units"]["default"] == "c"
    assert params["required"] == ["location"]


def test_tool_schema_override():
    custom = {"type": "function", "function": {"name": "f"}}

    @tool(schema=custom)
    def f():
        return 1

    assert f.schema is custom
    assert f.name == "f"


def test_tool_name_override():
    @tool(name="renamed")
    def f(x: int) -> int:
        return x

    assert f.name == "renamed"
    assert f.schema["function"]["name"] == "renamed"


def test_tool_remains_callable():
    assert isinstance(get_weather, Tool)
    assert get_weather("london") == "25c and sunny in london"
    assert get_weather("london", units="f") == "25c and sunny in london"


def test_register_plain_function_without_decorator():
    def get_news(topic: str) -> str:
        """Returns the latest news for a topic."""
        return f"headline: {topic}"

    llm = EchoLLM()
    env = Session(continue_live=True).root
    env.register_llm_fn(llm.complete)
    env.register_input_fn(lambda: "")
    env.register_tool_fns([get_news])

    tc = ToolCall(id="1", name="get_news", arguments='{"topic": "tech"}')
    assert env.call_tool(tc) == "headline: tech"

    env.rewind()
    env.input()
    response = env.llm_complete(build_context(env.history()))
    for replayed_tc in response.tool_calls:
        assert env.call_tool(replayed_tc) == "headline: tech"


def test_tool_replay_through_env():
    llm = EchoLLM()
    env = Session(continue_live=True).root
    env.register_llm_fn(llm.complete)
    env.register_input_fn(lambda: "")
    env.register_tool_fns([get_weather])

    tc = ToolCall(id="1", name="get_weather", arguments='{"location": "london"}')
    env.add_user_message("what's the weather in london?")
    env.add_message(
        role="assistant",
        content=None,
        tool_calls=(tc,),
    )
    result = env.call_tool(tc)
    assert result == "25c and sunny in london"
    env.add_message(role="assistant", content="It is 25c and sunny.")
    env.rewind()

    env.input()
    response = env.llm_complete(build_context(env.history()))
    assert response.tool_calls
    for replayed_tc in response.tool_calls:
        assert env.call_tool(replayed_tc) == "25c and sunny in london"
