from agentzero import build_context
from agentzero.environment import Environment
from agentzero.llms import EchoLLM


def main():
    llm = EchoLLM()
    env = Environment(continue_live=True)
    env.register_llm_fn(llm.complete)
    env.register_input_fn(input)
    env.add_user_message("hello")
    response = env.llm_complete(build_context(env.history()))
    print(response)


if __name__ == "__main__":
    main()
