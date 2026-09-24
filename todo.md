# TODO

## Naming & branding
- [ ] **Choose a name** (candidates so far: `retroscope`, `nahar`, `refold`, `recursor`, `agentfold`, `godsriver`; check PyPI/GitHub availability)
- [ ] Rename repo dir from `xmachina` and update `pyproject.toml` name / README

## High-priority features
- [ ] **Log persistence** — save/load the event node chain to disk so replay survives process restarts (the log currently lives only in memory)
  - [ ] Persistence format must be language-agnostic (e.g. JSON/JSONL or a defined binary schema), since a future **C++ port** will need to read/write the same logs
- [ ] **(future) C++ interop** — not a full parallel framework; language-agnostic log format + small native *consumer/continuation* libraries (C++, optionally TS) so other hosts can read-and-continue Python-written logs
- [ ] **Token/cost accounting** — record usage metadata per LLM event; `examples/token_saving/` is currently empty
- [ ] **`@tool` decorator** — thin, Pydantic-backed schema inference (`TypeAdapter.json_schema()`) returning `Tool`, with `schema=` override for bring-your-own-schema users (no LangChain dependency)

## Tests (prove the thesis)
- [ ] Fork semantics: parent log untouched, fork replay from its write head
- [ ] Tool-call replay vs. live execution through `env.call_tool`
- [ ] `build_context` regions / injections
- [ ] Replay determinism across the three modes
- [ ] Shared-store behavior across forks
- [ ] Verify `examples/` with actual assertions (currently none)

## Quality / CI
- [x] GitHub Actions workflow (pytest on 3.10+, ruff check/format, mypy)
- [x] ruff + mypy config in `pyproject.toml`
- [x] `py.typed` marker for a typed framework

## Packaging
- [x] `requires-python`, license, project URLs in `pyproject.toml`
- [x] dev dependency group (`pytest`, `ruff`, `mypy`)
- [x] `[tool.pytest.ini_options]` config
- [ ] Changelog / version notes

## Docs
- [ ] API reference for `EventNode`, `Sequence`, `WriteHead`, `build_context`