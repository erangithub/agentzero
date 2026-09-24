from agentzero import Session, build_context
from agentzero.llms import EchoLLM


def main():
    llm = EchoLLM()
    env = Session(llm=llm, input_fn=input, continue_live=True).root
    env.add_user_message("hello")
    response = env.llm_complete(build_context(env.history()))
    print(response)


if __name__ == "__main__":
    main()
