from agentzero import build_context
from agentzero.environment import Environment
from agentzero.llms import LMStudioLLM  # or OllamaLLM


def main():
    llm = LMStudioLLM()
    env = Environment(continue_live=True)
    env.register_llm_fn(llm.complete)
    env.register_input_fn(lambda: "7")
    env.add_message("user", "what is 2 plus 5?")
    env.add_message("assistant", "7")

    env.rewind()
    env.input()  # replays user message
    response = env.llm_complete(build_context(env.history()))
    print(response)


if __name__ == "__main__":
    main()
