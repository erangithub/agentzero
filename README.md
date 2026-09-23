# agentzero
 
**The non-agentic, agentic framework.**
 
## Message history is not state
 
Frameworks like LangGraph make you conflate the two. A conversation branches — one path summarizes, another retries, a third runs a sub-task — and now you need a schema for how to merge them back: `Annotated[list, operator.add]`, custom reducers, `InvalidUpdateError` when two branches write the same key, sub-agents spun up just to keep one branch's context from bleeding into another's. The graph, the reducers, and the sub-agents all exist to solve a problem that conflating history with state created in the first place.
 
In agentzero they're separate by construction. Message history lives on `Environment`: immutable, forkable, replayable. State is just computation: local variables, function calls, return values, whatever data structure or control structure the moment calls for. Forking a conversation to run three parallel analyses doesn't require declaring how their outputs merge: you write `results = await asyncio.gather(...)` and combine them however Python already lets you combine things.
 
**LangGraph** — merging three parallel branches requires declaring a reducer up front, in the schema, before you've written any logic:
 
```python
from typing import Annotated, TypedDict
import operator
 
class State(TypedDict):
    perspectives: Annotated[list[str], operator.add]  # merge strategy, declared ahead of time
 
graph.add_edge("start", "econ_node")
graph.add_edge("start", "social_node")
graph.add_edge("start", "infra_node")
# all three feed "synthesize", which reads state["perspectives"]
# want a dict keyed by label instead of a flat list? write a custom reducer.
```
 
**agentzero** — merging is just the next line of Python:
 
```python
fork_econ, fork_social, fork_infra = env.fork(), env.fork(), env.fork()
 
results = await asyncio.gather(
    get_perspective(fork_econ, "Focus on economic shifts and tax revenue."),
    get_perspective(fork_social, "Focus on social isolation and community building."),
    get_perspective(fork_infra, "Focus on public transit and office space conversion."),
)
 
perspectives = {
    "Economics": results[0].content,
    "Social": results[1].content,
    "Infrastructure": results[2].content,
}
```
 
No schema. No reducer. No `InvalidUpdateError` to debug. Each fork sees its own slice of history; the dict on the last line is the entire "merge strategy."
 
---
 
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
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            env.call_tool(tc)
```
 
That's the entire framework. Memory, checkpointing, branching, multi-agent, streaming — all of it is a variation on this loop, written in plain Python.

---

## The idea

Split your code into two things.

**Your logic.** Deterministic, explicit, testable Python — loops, conditionals, functions. No DSL, no framework-specific control flow.

**The environment.** The source of all non-determinism. Every call to the outside world — LLM completions, user input, tool results — goes through `env` and gets written to an immutable log.

You don't need the LLM to be deterministic. You need the log to be honest. That one discipline gets you three things other frameworks solve with infrastructure:

- **Replay** — rerun a conversation exactly from the log, no LLM calls, no live input.
- **Branching** — fork the log at any point and continue down a different path. The parent is never touched.
- **Auditability** — the log is the ground truth, not the LLM's memory, not a side database.

This is [event sourcing](https://martinfowler.com/eaaDev/EventSourcing.html) applied to LLM conversations.

| Event sourcing | agentzero |
|---|---|
| Event store | Immutable node chain — O(1) branching |
| Event | `Message` / tool result |
| Projection | `build_context(env.history())` |
| Command | `env.input()` / `env.llm_complete()` / `env.call_tool()` |

---

## Replay

The same code runs in three modes — recording, replaying, and replay-then-live. No mocks, no test flags, no special cases.

```python
from agentzero import build_context
from agentzero.llms import EchoLLM
from agentzero.environment import Environment

# Record
env = Environment(llm=EchoLLM(), input_fn=input)
env.add_message(role="user", content="hello")
env.add_message(role="assistant", content="echo: hello")

# Replay — reads from the log, no LLM called
env.rewind()
env.input()
response = env.llm_complete(build_context(env.history()))

assert response.content == "echo: hello"
```

`rewind()` resets the environment's read head to the start of the log. `env.input()` reads from the log instead of prompting. `env.llm_complete()` returns the recorded response instead of calling the API. Your loop doesn't change.

You can also replay up to a point, then continue live:

```python
env = Environment(llm=OpenAILLM(), input_fn=input, continue_live=True)
env.add_message(role="user", content="hello")

env.rewind()
env.input()                                       # replayed from log
response = env.llm_complete(build_context(env.history()))  # live from here
```

Pre-fill a conversation, replay the setup, continue live from any point. Useful for development, evals, and regression testing.

---

## Forking

`env.fork()` creates a child environment that shares the parent's history up to the fork point, with its own read/write head. The parent log is never touched — no merges, no conflicts. Think of it like a function call: you never merge the call stack back into the caller, but you can return a value.

```python
def summarize(env) -> str:
    sub = env.fork()
    context = build_context(sub.history(), system="Summarize the conversation.")
    response = sub.llm_complete(context)
    return response.content

def flow(env):
    env.add_message(role="user", content="long conversation...")

    summary = summarize(env)   # forked — parent log unchanged

    context = build_context(env.history(), injections=[summary])
    return env.llm_complete(context)
```

The fork replays automatically on `rewind()`. `flow()` is identical in live and replay modes.

Forking also gets you parallelism for free — standard Python concurrency, no orchestrator required:

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

History and state are separate by design. A fork isolates *history* — what the branch sees when it builds context — but forks can share a *store* for cross-cutting application state (counters, DB handles, cost tracking) if you choose to. A branch can also look past its own history and read from the root if that's what the logic calls for; `build_context` decides what a given call actually sees, `Environment` just provides the log to read from.

---

## Tools

```python
from agentzero.environment import Environment, Tool

tools = [Tool(name="get_weather", fn=get_weather, schema=weather_schema)]
env = Environment(llm=OpenAILLM(), input_fn=input)
env.register_tool_fns(tools)

while True:
    request  = env.input()
    while True:
        context  = build_context(env.history())
        response = env.llm_complete(context)
        if not response.tool_calls:
            break
        for tc in response.tool_calls:
            env.call_tool(tc)
```

Schema generation is handled by whatever you already use — the OpenAI SDK, Pydantic, FastMCP. agentzero doesn't provide a `@tool` decorator; that would conflict with what you already have.

Tool calls registered this way go through the same replay-or-execute mechanism as everything else: replayed from the log if recorded, executed live otherwise. If you'd rather a tool call always run live regardless of replay state, call it directly outside `env` — that's a deliberate escape hatch, not an oversight.

---

## Any non-deterministic function

LLM calls and tool calls are just the built-in cases. Anything non-deterministic can go through the same mechanism:

```python
env = Environment()

get_price = env.nondet(get_price_fn)   # wrapped, local reference
env.register_nondet(get_price_fn)      # or attached to env by name

price = get_price("AAPL")              # executes live, records the result
env.rewind()
price = get_price("AAPL")              # replayed from the log, not re-executed
```

`env.llm_complete`, `env.input()`, and `env.add_message()` are all built on this same primitive.

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

- `examples/simple/hello.py` — basic LLM call
- `examples/simple/replay.py` — all three replay modes
- `examples/simple/interactive.py` — interactive chat loop
- `examples/simple/streaming.py` — streaming completions
- `examples/nondet/example.py` — wrapping arbitrary non-deterministic functions
- `examples/tool_call/tool_call_echo.py` — tool use loop (no external dependencies)
- `examples/tool_call/tool_call_gemini.py` — tool use with Gemini
- `examples/tool_call/async_tool_call_gemini.py` — async tool use with Gemini
- `examples/swarm/swarm.py` — multi-branch tree of agents
- `examples/swarm/branch_summarize.py` — summarization in a forked branch
- `examples/timetravel/delorean.py` — debugging with replay / time travel
