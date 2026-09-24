from agentzero import Session, build_context
from agentzero.llms import EchoLLM


def test_hello_world_replay():
    llm = EchoLLM()
    env = Session(continue_live=True).root
    env.register_llm_fn(llm.complete)
    env.register_input_fn(input)
    env.add_message("user", "hello")
    env.add_message("assistant", "echo: hello")
    env.rewind()

    env.input()  # replays user message
    response = env.llm_complete(build_context(env.history()))

    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_full_replay passed")


def test_hello_world_replay_then_live():
    llm = EchoLLM()
    env = Session(continue_live=True).root
    env.register_llm_fn(llm.complete)
    env.register_input_fn(input)
    env.add_message("user", "hello")
    env.rewind()

    env.input()  # replays user message
    response = env.llm_complete(build_context(env.history()))

    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_replay_then_live passed")


def test_hello_world_live():
    llm = EchoLLM()
    env = Session(continue_live=True).root
    env.register_llm_fn(llm.complete)
    env.register_input_fn(lambda: "hello")

    env.input()  # reads user message
    response = env.llm_complete(build_context(env.history()))

    assert response.role == "assistant"
    assert response.content == "echo: hello"
    print("test_hello_world_live passed")


if __name__ == "__main__":
    test_hello_world_replay()
    test_hello_world_replay_then_live()
    test_hello_world_live()
