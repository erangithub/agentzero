# TODO

## Naming & branding
- [ ] **Choose a name** (candidates so far: `retroscope`, `nahar`, `refold`, `recursor`, `agentfold`, `godsriver`; check PyPI/GitHub availability)
- [ ] Rename repo dir from `xmachina` and update `pyproject.toml` name / README

## High-priority features
- [x] **`Session`** — the single entry point (owns the root `Environment`; branches come from `fork()`/`from_json`); thread-safe registry of environments keyed by origin node; single source of truth for parent→child topology; direct `Environment` construction is undocumented (docstring: "only created by a Session")
- [x] **JSON persistence** — `Session.to_json()` / `Session.from_json()` language-agnostic JSONL round-trip; a reloaded session replays from origin, then continues live; disk usage in `examples/simple/persist.py` (`.sessions/*.jsonl`) (`tests/test_session.py`)
  - [ ] **Log persistence API** — horizon: `store()` / `load()` from disk wrapped around the JSON round-trip
- [ ] **Postgres persistence** — same `store()`/`load()` surface backed by a log table (JSONL stays the wire format); not started
- [ ] Persistence format must be language-agnostic (JSON/JSONL spec) since a future **C++ port** will need to read/write the same logs
- [ ] **(future) C++ interop** — not a full parallel framework; language-agnostic log format + small native *consumer/continuation* libraries (C++, optionally TS) so other hosts can read-and-continue Python-written logs
- [ ] **Token/cost accounting** — record usage metadata per LLM event; `examples/token_saving/` is currently empty
- [ ] **Record `CallEvent.args`** — the field is in the schema but nothing populates it, so `CallEvent.mismatch()` can only compare `fn_name`: a call replayed with *different arguments* goes unnoticed and silently returns the old result. An LLM call is not a function of the log, so these args are part of the fold's recorded inputs — without them a "fold" that cannot be re-evaluated is a cached value wearing a fold's clothes. Needs capture at the call boundary (`register_nondet` receives `fn` as a closure, so the arguments are not reachable from the function object) plus serialization in `session.py`
- [x] **`@tool` decorator** — thin, Pydantic-backed schema inference (`create_model` → `model_json_schema()`) returning `Tool`, with `schema=`/`name=` overrides; decorated fns stay callable; verbatim `Tool` construction still supported for bring-your-own-schema users
- [ ] **Language-agnostic log format** — lock the on-disk spec (JSON/JSONL) once persistence lands, so future TS/C++ consumers can read-and-continue

## Log shape: env as a cursor, not a baked-in branch
- [x] **Drop `branch_start` fork nodes** — make an `Environment` a *cursor* on a shared event DAG (like a git branch = a movable ref) instead of a branch baked into the history. An env = `{write_head, read_head, fork_point}`; `fork()` just places a new cursor at the current node and writes **no** node. A fork point's children become co-equal events (mainline is not structurally special), and branch content attaches directly to the shared node instead of one hop removed
  - [x] Fork point points at a *real* event node (replaces `origin_node = branch_start`); replay still starts strictly after it, so `fork_history()` is unchanged
  - [x] Root env has `fork_point is None`; `from_json` identifies roots as `origin is None` (replaces the `origin.parent is None` check)
  - [x] Env records carry the parent env id, so `from_json` reconstructs parent→child directly instead of the `max(origin_node.depth)` chain-matching hack (`session.py:172-183`)
  - [x] Env identity is its own uuid (`env.id`), since cursors are no longer identified by a unique branch node — siblings share a fork point
  - [x] `to_json` walks each cursor to the DAG head and unions, so shared ancestors are stored once
  - [x] `ControlEvent`/`ControlKind` deleted outright (they existed only for branch markers); the marker-skip in `_read` and control serialization return with the transaction item
  - [x] Nullable `Sequence.to_node` / `prev_node` / `current_depth`, since the log's first node can now have no parent
  - [x] Fixed the go-live sync in `_message_event`: it was guarded on `read_head.prev` being truthy, which only held because of the old `branch_start` sentinel — rolling back to the *first* node failed to orphan the tail
  - [x] Fixed the same gap in `_call_event` / `_amessage_event`: all three primitives now call one shared `_anchor_write_head()` (move write head to the read position, clear the read head, drop the stop predicate), so a flow that goes live mid-replay then makes a nondet or async call resumes at the cursor instead of appending after the discarded tail. Covered by `test_nondet_call_going_live_anchors_at_read_head` and `test_async_message_going_live_anchors_at_read_head`

## Transactions — atomic regions of the log
- [ ] **Transaction markers** — `BEGIN`/`COMMIT` as recorded marker nodes (reuse the `ControlEvent`/`ControlKind` slot freed by dropping `branch_start`). Log-level atomicity: the run of events between BEGIN and COMMIT is one unit; aborting orphans the whole range via the existing rewind + go-live (orphan) machinery
  - [ ] API: `with env.transaction():` context manager (commit on clean exit, abort+rollback on exception) plus explicit `env.begin()` / `env.commit()` / `env.abort()`; optional txn name/id on BEGIN
  - [ ] Scope: **per-cursor** (a cursor's own writes). Each env tracks its open transaction and refuses a nested `BEGIN`
  - [ ] Crash detection: a `BEGIN` with no matching `COMMIT` at the tail = an interrupted session; surface it on reload (auto-rollback or an explicit flag)
  - [ ] Replay is free: markers are recorded events, so `begin()`/`commit()` re-read them during replay and the flow sees the same boundaries
  - [ ] Introduce `MarkerEvent`/`MarkerKind` for these (replacing the now-deleted `ControlEvent`/`ControlKind`), plus the marker-skip in `_read` and control serialization in `session.py`
  - [ ] Caveat: log-level atomicity only — does **not** undo external side effects already fired by tools/LLMs inside the transaction

## Marker / checkpoint nodes (forward-looking)
- [ ] **Checkpoint / cached-state markers** — "resume here from cached state" for long sessions (O(1) instead of replaying the flow) and to cache large derived state out-of-band. Honest caveat: replay already never re-calls the LLM, so the classic snapshot motivation is weak here; it earns its keep for *large/external* cached state, and unlike transactions it needs real new framework machinery (a `seek` replay entry point). Partly overlaps `nondet`, which already records small nondet results inline

## Replay timing & pacing (stashed — rebuild one item at a time)
- [ ] **Record per-event timing** — stamp every `MessageEvent`/`CallEvent` at write time with a wall-clock `timestamp` and a measured `duration_ms` (survives the JSONL round-trip)
- [ ] **`Sequence.iter_timed()`** — yield timed events with `delta_ms` (unfolded wall delta vs the previous timed event) and `gap_ms` (time consumed by neither event), skipping control/branch nodes
- [ ] **Async non-deterministic events** — `_acall_event` + async-aware `nondet`/`register_nondet`, so async flows record timers/tools like sync ones do
- [ ] **`replay_speed` pacing** — replay a loaded session against its recorded timeline: `0`/`None` = instant, `1.0` = real time, `2.0` = 2x. Settable at load (`Session.from_json(..., speed=)`) or changed any time on a live `env`; needs a sync and an async read path

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