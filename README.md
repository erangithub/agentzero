# AgentZero

> *Deus ex machina* — the god from the machine. Ancient playwrights lowered a god onto the stage with a crane when they'd lost control of the plot. Aristotle called it a cheat. The crane was the problem, not the god.

Modern AI frameworks are cranes. LangGraph, AutoGen, CrewAI — elaborate machinery built to lower the LLM onto the stage in a controlled way. Graphs of nodes. Edge conditions. Orchestration layers. DSLs inside Python.

**You don't need any of that.**

Python is already a universal language. Any workflow you can express as a graph, you can express as a function — and the function is simpler, more debuggable, and more composable. The graph adds nothing computationally. It adds ceremony.

The LLM is a non-deterministic function call. Treat it like one.

The only thing Python *can't* give you natively is a disciplined boundary between your deterministic code and the non-deterministic outside world — user input, LLM completions, tool results. That boundary is what AgentZero provides. Nothing more.

---

## The idea

Separate your code into two things:

**Your logic** — deterministic, explicit, testable Python. Loops, conditionals, functions. No DSL. No framework-specific abstractions. Just code.

**The environment** — the source of all non-determinism. Every call to the outside world goes through `env`. Every result is written to an immutable log.

```python
from agentzero import build_context
from agentzero.llms import OpenAILLM
from agentzero.environment import Environment

env = Environment(llm=OpenAILLM(), input_fn=input)

while True:
    request  = env.input()
    while True:
        context  = build_context(env.history())
        response = env.llm_complete(context)
        if not response.tool_calls: break
        for tc in response.tool_calls:
            env.call_tool(tc)
```

That's the entire framework. Everything else — memory, checkpointing, branching, multi-agent, streaming — is a variation of this loop, written in Python.

---

## Why the log matters

You don't need the LLM to be deterministic. You need the log to be honest.

Every call to `env` appends to an immutable log. This one discipline unlocks three things that other frameworks solve with infrastructure:

**Replay.** Rerun a conversation exactly — no LLM calls, no user input, just the log. Useful for debugging, testing, and understanding what actually happened.

**Branching.** Fork the log at any point and continue down a different path. The parent is never touched. No copies, no overhead — forks share history.

**Auditability.** The log is the ground truth. Not the LLM's memory. Not a database. The log.

This is [event sourcing](https://martinfowler.com/eaaDev/EventSourcing.html) — a well-understood pattern — applied to LLM conversations.

| Event Sourcing | AgentZero |
|---|---|
| Event Store | Immutable node chain — O(1) branching |
| Event | `Message` |
| Projection | `build_context(env.history())` |
| Command | `env.input()` / `env.llm_complete()` / `env.call_tool()` |

---

## Replay

The same code runs in three modes. No mocks. No test flags. No special cases.

```python
from agentzero import Message, build_context
from agentzero.llms import EchoLLM
from agentzero.environment import Environment

# Record
env = Environment(llm=EchoLLM(), input_fn=input)
env.add_message(Message("user", "hello"))
env.add_message(Message("assistant", "echo: hello"))

# Replay — reads from log, no LLM called
env.rewind()
env.input()
response = env.llm_complete(build_context(env.history()))

assert response.content == "echo: hello"
```

`rewind()` resets the environment. `env.input()` reads from the log instead of prompting. `env.llm_complete()` returns the recorded response instead of calling the API. Your loop doesn't change.

You can also replay up to a point, then continue live:

```python
env = Environment(llm=OpenAILLM(), input_fn=input, continue_live=True)
env.add_message(Message("user", "hello"))

env.rewind()
env.input()                                        # replayed from log
response = env.llm_complete(build_context(...))    # live LLM call from here
```

Pre-fill a conversation. Replay the setup. Continue from any point. Useful for development, evaluation, and regression testing.

---

## Forking

`env.fork()` creates a child environment that shares the parent's history but has its own write head. The parent log is never touched.

The practical use: running a sub-task without contaminating the main conversation.

```python
def summarize(env) -> str:
    sub = env.fork()
    context = build_context(sub.history(), system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return response.content

def flow(env):
    env.add_message(Message("user", "long conversation..."))

    summary = summarize(env)   # fork — parent log unchanged

    context = build_context(env.history(), injections=[summary])
    return env.llm_complete(context)
```

The fork replays automatically on `rewind()`. `flow()` is identical in live and replay modes.

Forking also enables parallelism — standard Python concurrency, no swarm orchestrator needed:

```python
import asyncio

async def run_parallel(env):
    branch_a = env.fork()
    branch_b = env.fork()

    result_a, result_b = await asyncio.gather(
        run_branch(branch_a),
        run_branch(branch_b),
    )
```

---

## Tools

```python
from agentzero.environment import Environment, Tool

tools = [Tool(name="get_weather", fn=get_weather, schema=weather_schema)]
env = Environment(llm=OpenAILLM(), tools=tools, input_fn=input)

while True:
    request  = env.input()
    while True:
        context  = build_context(env.history())
        response = env.llm_complete(context)
        if not response.tool_calls: break
        for tc in response.tool_calls:
            env.call_tool(tc)
```

Schema generation is handled by whatever you're already using — OpenAI SDK, Pydantic, FastMCP. AgentZero doesn't provide `@tool`. That would conflict with what you already have.

---

## LLM providers

```python
from agentzero.llms import OpenAILLM   # OpenAI
from agentzero.llms import GroqLLM     # Groq (free tier)
from agentzero.llms import OllamaLLM   # Ollama (local)
from agentzero.llms import LMStudioLLM # LM Studio (local)
from agentzero.llms import EchoLLM     # testing
```

---

## Requirements

Python 3.10+

```
pip install -e .
```

---

## Examples

- `examples/hello.py` — basic LLM call
- `examples/summarize.py` — forking for summarization
- `examples/tools.py` — tool use loop
- `examples/replay.py` — all three replay modes

---

*The playwright who reaches for the crane because they've lost control of the plot has a different relationship to their story than the one who never needed it. AgentZero is for the second kind of engineer.*
