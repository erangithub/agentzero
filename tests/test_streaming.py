from agentzero import build_context
from agentzero.environment import Environment
from agentzero.llms import EchoLLM


def stream_flow(env):
    env.input()
    full_content = ""
    count_deltas = 0
    for delta in env.llm_stream(build_context(env.history())):
        full_content += delta.content
        count_deltas += 1
        print(delta.content, end="")

    print()
    messages = list(env.history().iter_messages())
    assert len(messages) == 2
    expected_user = "one two three four five"
    assert messages[0].content == expected_user
    expected_assistant = "echo: " + expected_user
    assert (messages[1].content or "").strip() == expected_assistant
    assert full_content.strip() == expected_assistant
    assert count_deltas == 6


def test_streaming():
    llm = EchoLLM()
    env = Environment(continue_live=True)
    env.register_llm_stream_fn(llm.stream, name="llm_stream")  # event name: llm.complete
    env.register_input_fn(lambda: "one two three four five")

    stream_flow(env)
    env.rewind()
    stream_flow(env)


if __name__ == "__main__":
    test_streaming()
