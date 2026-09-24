"""Converse with LM Studio, saving the session to a local .sessions/*.jsonl file.

On resume the saved log is replayed from origin (deterministic — no LLM calls),
then the conversation continues live, exactly like a fresh run.

Usage:
    python examples/simple/persist.py                  # start a new conversation
    python examples/simple/persist.py <.sessions/x.jsonl>   # resume a saved one
"""

import sys
import time
from pathlib import Path

from agentzero import Session, build_context
from agentzero.environment import transient
from agentzero.llms import LMStudioLLM


def get_input():
    text = input().strip()
    if text.lower() in ("quit", "exit"):
        return transient(text.lower())
    return text


def main():
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            sys.exit(f"Session file not found: {path}")
        session = Session.from_json(path.read_text(), llm=LMStudioLLM(), input_fn=get_input)
        env = session.root
        print(f"Resumed from {path}")
    else:
        path = Path(".sessions") / f"chat-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
        path.parent.mkdir(exist_ok=True)
        session = Session(llm=LMStudioLLM(), input_fn=get_input)
        env = session.root
        print("New session. Type 'quit' to save and exit.")

    def save():
        path.parent.mkdir(exist_ok=True)
        path.write_text(session.to_json())
        print(f"(saved to {path})")

    while True:
        replaying = env.is_replay
        user_input = env.input()
        if not replaying and user_input in ("quit", "exit"):
            save()
            break
        print(f"You: {user_input}")
        response = env.llm_complete(build_context(env.history()))
        print(f"Assistant: {response.content}")
        if not replaying:
            save()


if __name__ == "__main__":
    main()
