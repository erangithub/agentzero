from agentzero import Session, build_context
from agentzero.llms import LMStudioLLM


def main():
    def get_input():
        return input("You: ")

    env = Session(llm=LMStudioLLM(), input_fn=get_input).root

    while True:
        user_input = env.input()
        if user_input.lower() in ("quit", "exit"):
            break
        response = env.llm_complete(build_context(env.history()))
        print(f"Assistant: {response.content}")


if __name__ == "__main__":
    main()
