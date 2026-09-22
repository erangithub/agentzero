from agentzero import build_context
from agentzero.llms import EchoLLM # LMStudioLLM  # or OllamaLLM
from agentzero.environment import Environment, transient
from agentzero.eventlog import MessageEvent


def prev_user_depth(env) -> int | None:
    node = env.prev_node.find(lambda n: n.is_message("user"))
    return node.depth if node else None

# Set debug_input to prefill user inputs for debugging
debug_input = [] # ["apple", "banana", "b", "b", "n", "citrus"]

def get_input(depth):
    global debug_input
    if debug_input:
        user_input = debug_input.pop(0)
    else:
        user_input = input(f"{depth} > ").strip()
    if user_input.lower() in ["quit", "b", "n", "reset"]:
        return transient(user_input.lower())
    return user_input

def fresh_env():
    env = Environment(llm=EchoLLM(), continue_live=True)
    env.register_input_fn(get_input)
    return env

def main():
    env = fresh_env()

    print("Chat with your LLM. Commands:")
    print("  'b' — roll back the last exchange")
    print("  'n' - move to the next exchange")
    print("  'reset' - reset environment")
    print("  'quit' — exit")
    print()

    while True:
        user_input = env.input(env.current_depth)

        if user_input == "reset":
            print("Environment resetted")
            env = fresh_env()
            continue
        
        if user_input == "quit":
            break

        if user_input == "b":
            target_depth = prev_user_depth(env)
            if target_depth is None:
                print('No previous user input step')
                continue
            env.rewind()
            env.replay_until(lambda n, d=target_depth: n.depth >= d)
            print(f"Going back to ({target_depth})")
            continue

        if user_input == "n":
            current_depth = env.current_depth
            env.replay_until(lambda n, d=current_depth: n.is_message("user") and n.depth > d)
            if not env.is_replay:
                print('No more steps')
            else:
                print(f"Going to next user input")
            continue

        print(f"{env.prev_node.depth} | User: {user_input}")
        response = env.llm_complete(build_context(env.history()))
        print(f"{env.prev_node.depth} | Assistant: {response.content}")
        print()


    # The final log — only the messages the user kept
    print("\n=== Conversation Log ===")
    for msg in env.history().iter_messages():
        label = "You" if msg.role == "user" else "Assistant"
        print(f"{label}: {msg.content}")


if __name__ == "__main__":
    main()