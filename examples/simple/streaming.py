from agentzero import Session, build_context
from agentzero.llms import EchoLLM


def main():
    llm = EchoLLM()
    env = Session(llm=llm, input_fn=input, continue_live=True).root
    env.register_llm_stream_fn(llm.stream, name="llm_stream")
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
