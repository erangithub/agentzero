from agentzero import Session, build_context
from agentzero.llms import LMStudioLLM  # or OllamaLLM


def main():
    env = Session(llm=LMStudioLLM(), input_fn=lambda: "7", continue_live=True).root
    env.add_message("user", "what is 2 plus 5?")
    env.add_message("assistant", "7")

    env.rewind()
    env.input()  # replays user message
    response = env.llm_complete(build_context(env.history()))
    print(response)


if __name__ == "__main__":
    main()
