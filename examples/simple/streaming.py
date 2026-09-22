from agentzero import Message, build_context
from agentzero.llms import EchoLLM
from agentzero.environment import Environment


def main():
    llm = EchoLLM()
    env = Environment(continue_live=True)
    env.register_llm_stream_fn(llm.stream, name="llm_stream")
    env.register_input_fn(input)
    env.add_user_message("And I think to myself, what a wonderful world.")

    env.rewind(continue_live=True)

    print(env.input())
    full_content = ""
    for delta in env.llm_stream(build_context(env.history())):
        print(delta.content, end="", flush=True)
        full_content += delta.content

    print(f"\nLogged {len(list(env.history().iter_messages()))} messages")


if __name__ == "__main__":
    main()
