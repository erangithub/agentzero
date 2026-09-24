"""@tool decorator — thin, Pydantic-backed tool schema inference.

Schemas are OpenAI-style JSON; the env's replay/logging path is unchanged.
Brings no schema DSL of its own — if you already produce a schema (OpenAI
SDK, FastMCP, hand-written), pass it via ``schema=``.

The decorator is optional: ``env.register_tool_fns`` infers schemas for plain
functions too. Decorating is only useful to pin a ``name`` or ``schema``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from agentzero.environment import Tool
from agentzero.schema import schema_from_fn


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    schema: dict[str, Any] | None = None,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Turn a function into an agentzero ``Tool``.

    Usage::

        @tool
        def get_weather(location: str, units: str = "c") -> str: ...

        env.register_tool_fns([get_weather])   # works with replay/logging

    ``get_weather`` remains directly callable — it's now a ``Tool`` whose
    ``__call__`` delegates to the original function.

    ``schema=`` overrides inference. ``name=`` overrides the function name.
    """

    def decorate(f: Callable[..., Any]) -> Tool:
        return Tool(
            name=name or f.__name__,
            fn=f,
            schema=schema or schema_from_fn(f, name),
        )

    if fn is not None:
        return decorate(fn)
    return decorate
