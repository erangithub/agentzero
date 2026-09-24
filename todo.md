# TODO

## Naming & branding
- [ ] **Choose a name** (candidates so far: `retroscope`, `nahar`, `refold`, `recursor`, `agentfold`, `godsriver`; check PyPI/GitHub availability)
- [ ] Rename repo dir from `xmachina` and update `pyproject.toml` name / README

## High-priority features
- [x] **`Session`** — the single entry point (owns the root `Environment`; branches come from `fork()`/`from_json`); thread-safe registry of environments keyed by origin node; single source of truth for parent→child topology; direct `Environment` construction is undocumented (docstring: "only created by a Session")
- [x] **JSON persistence (in-memory JSONL round-trip)** — `Session.to_json()` / `Session.from_json()` with language-agnostic spec; DB/file I/O deferred (`tests/test_session.py`)
  - [ ] **Log persistence** — horizon: `stored()` / `load()` from disk wrapped around the JSON round-trip (the log currently lives only in memory)
  - [ ] Persistence format must be language-agnostic (JSON/JSONL spec) since a future **C++ port** will need to read/write the same logs
- [ ] **(future) C++ interop** — not a full parallel framework; language-agnostic log format + small native *consumer/continuation* libraries (C++, optionally TS) so other hosts can read-and-continue Python-written logs
- [ ] **Token/cost accounting** — record usage metadata per LLM event; `examples/token_saving/` is currently empty
- [x] **`@tool` decorator** — thin, Pydantic-backed schema inference (`create_model` → `model_json_schema()`) returning `Tool`, with `schema=`/`name=` overrides; decorated fns stay callable; verbatim `Tool` construction still supported for bring-your-own-schema users
- [ ] **Language-agnostic log format** — lock the on-disk spec (JSON/JSONL) once persistence lands, so future TS/C++ consumers can read-and-continue

## Tests (prove the thesis)
- [x] Fork semantics: parent log untouched, fork replay from its write head (`tests/test_fork.py`)
- [x] Tool-call replay vs. live execution through `env.call_tool` (`tests/test_tool.py`)
- [x] `build_context` regions / injections (`tests/test_fork.py`)
- [x] Replay determinism across the three modes (`tests/test_hello.py`, `tests/test_timetravel.py`)
- [x] Swarm-style async parallel forks + reduce (`tests/test_swarm.py`)
- [x] Time travel: 'b' rollback discards exchange, transient commands never logged (`tests/test_timetravel.py`)
- [x] Shared-store behavior across forks: registrations thread-safe, fork reuse after load, orphaned nodes not restored (`tests/test_session.py`)

## Quality / CI
- [x] GitHub Actions workflow (pytest on 3.10+, ruff check/format, mypy)
- [x] ruff + mypy config in `pyproject.toml` (markdown excluded from ruff format — README alignment is intentional)
- [x] `py.typed` marker for a typed framework
- [ ] Push and confirm CI is green end-to-end

## Packaging
- [x] `requires-python`, license, readme, classifiers in `pyproject.toml`
- [x] dev dependency group (`pytest`, `ruff`, `mypy`)
- [x] `[tool.pytest.ini_options]` config
- [ ] Publish to PyPI (confirm name on PyPI first — supersedes "choose a name")
- [ ] Changelog / version notes

## Docs
- [ ] API reference for `EventNode`, `Sequence`, `WriteHead`, `build_context`