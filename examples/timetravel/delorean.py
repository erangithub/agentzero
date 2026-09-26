from agentzero import Session, build_context
from agentzero.environment import transient
from agentzero.llms import EchoLLM  # LMStudioLLM  # or OllamaLLM


def prev_user_depth(env) -> int | None:
    node = env.prev_node.find(lambda n: n.is_message("user"))
    return node.depth if node else None


# Set debug_input to prefill user inputs for debugging
debug_input = []  # ["apple", "banana", "/b", "/b", "/n", "citrus"]


def get_input(depth):
    global debug_input
    if debug_input:
        user_input = debug_input.pop(0)
    else:
        user_input = input(f"{depth} > ").strip()
    if user_input.startswith("/"):
        return transient(user_input)
    return user_input


def fresh_session():
    return Session(llm=EchoLLM(), input_fn=get_input, continue_live=True)


def main():
    session = fresh_session()
    env = session.root

    print("Chat with your LLM. Commands:")
    print("  /b — roll back the last exchange")
    print("  /n - move to the next exchange")
    print("  /tree - print every branch of the log, including abandoned ones")
    print("  /reset - start a new session")
    print("  /quit — exit")
    print()

    while True:
        user_input = env.input(env.current_depth)

        if user_input.startswith("/"):
            command = user_input[1:]

            if command == "quit":
                break

            if command == "tree":
                session.print_tree()
                continue

            if command == "reset":
                print("Starting a new session")
                session = fresh_session()
                env = session.root
                continue

            if command == "b":
                target_depth = prev_user_depth(env)
                if target_depth is None:
                    print("No previous user input step")
                    continue
                env.rewind()
                env.replay_until(lambda n, d=target_depth: n.depth >= d)
                print(f"Going back to ({target_depth})")
                continue

            if command == "n":
                current_depth = env.current_depth
                env.replay_until(lambda n, d=current_depth: n.is_message("user") and n.depth > d)
                if not env.is_replay:
                    print("No more steps")
                else:
                    print("Going to next user input")
                continue

            print(f"Unknown command: /{command}")
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
