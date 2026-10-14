"""Typed tools: a Python function becomes a tool with a JSON schema for free.

The function's type hints build a Pydantic model, which both produces the schema
Claude sees and validates the arguments Claude sends back. Bad arguments come back
to the model as an error it can correct, instead of crashing the run.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model


class ToolError(Exception):
    """An expected tool failure; the message is shown to the model."""


@dataclass
class Tool:
    name: str
    description: str
    fn: Callable[..., Any]
    args_model: type[BaseModel]
    side_effect: bool = False  # True: changes something outside the agent, needs approval
    max_calls: int | None = None

    def schema(self) -> dict[str, Any]:
        schema = self.args_model.model_json_schema()
        schema.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}

    def run(self, arguments: dict[str, Any]) -> str:
        try:
            parsed = self.args_model.model_validate(arguments)
        except ValidationError as exc:
            problems = "; ".join(
                f"{'.'.join(map(str, e['loc'])) or 'input'}: {e['msg']}" for e in exc.errors()
            )
            raise ToolError(f"invalid arguments: {problems}") from exc
        result = self.fn(**parsed.model_dump())
        return result if isinstance(result, str) else str(result)


def tool(
    side_effect: bool = False, max_calls: int | None = None, name: str | None = None
) -> Callable[[Callable[..., Any]], Tool]:
    """Decorator: `@tool()` on a typed function with a docstring."""

    def wrap(fn: Callable[..., Any]) -> Tool:
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param.name] = (hints.get(param.name, Any), default)
        model = create_model(f"{fn.__name__}_args", **fields)
        description = inspect.getdoc(fn) or fn.__name__
        return Tool(name or fn.__name__, description, fn, model, side_effect, max_calls)

    return wrap


@dataclass
class ToolRegistry:
    tools: dict[str, Tool] = field(default_factory=dict)

    def add(self, *tools: Tool) -> ToolRegistry:
        for t in tools:
            self.tools[t.name] = t
        return self

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    def get(self, name: str) -> Tool | None:
        return self.tools.get(name)
