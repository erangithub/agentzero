"""Tool schema helpers. Kept import-free of ``agentzero.environment`` to avoid
circular imports (``environment`` and ``tools`` both depend on this module).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, cast, get_type_hints

from pydantic import create_model


def schema_from_fn(fn: Callable[..., Any], name: str | None = None) -> dict[str, Any]:
    """Infer an OpenAI-style function schema from a function's signature.

    Parameters map to properties; those without defaults land in ``required``.
    The docstring becomes the description. ``name`` overrides ``fn.__name__``.
    """
    try:
        hints = get_type_hints(fn)
    except Exception:
        hints = {}
    fields: dict[str, tuple[Any, Any]] = {}
    for param in inspect.signature(fn).parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        annotation = hints.get(param.name, Any)
        default = Ellipsis if param.default is param.empty else param.default
        fields[param.name] = (annotation, default)

    model = create_model(fn.__name__ + "_args", **cast(Any, fields))
    return {
        "type": "function",
        "function": {
            "name": name or fn.__name__,
            "description": inspect.getdoc(fn),
            "parameters": model.model_json_schema(),
        },
    }
