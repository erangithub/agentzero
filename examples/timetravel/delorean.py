from agentzero import Sequence, Session, build_context
from agentzero.environment import transient
from agentzero.llms import EchoLLM  # LMStudioLLM  # or OllamaLLM

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
                current = env.prev_node
                if current is None or current.depth == 0:
                    print("Already at the start of the log")
                    continue
                # Stop just before the preceding user turn, so that turn gets
                # replayed into the next prompt rather than skipped. `find`
                # includes the current node, hence the depth test.
                target = current.find(
                    lambda n, c=current: n.depth < c.depth and n.is_message("user")
                )
                if target is None:
                    print("Already at the start of the log")
                    continue
                env.rewind()
                env.replay_until(lambda n, d=target.depth: n.depth >= d)
                print(f"Going back to ({target.depth})")
                continue

            if command == "n":
                # "Is anything ahead?" is next_node, not is_replay: straight
                # after a /b the stop predicate blocks on the very first node,
                # so is_replay is False even though the log is still unread.
                users = [
                    n
                    for n in Sequence(
                        after_node=env.prev_node, to_node=env.write_head.prev
                    ).iter_nodes()
                    if n.is_message("user")
                ]
                if len(users) >= 2:
                    env.replay_until(lambda n, d=users[1].depth: n.depth >= d)
                    print("Going to next user input")
                elif users:
                    # One exchange left: replay it, then let the next read go
                    # live. A predicate that never fires walks to the end and
                    # then stops on its own, so the tail is not orphaned.
                    env.replay_until(lambda n: False)
                    print("Replayed to the end of the log")
                else:
                    print("No more steps")
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
