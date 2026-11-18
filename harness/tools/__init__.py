"""Tools: what the model can *do*, beyond producing text.

Write a plain typed Python function, decorate it with `@tool`, and you get a
`Tool` whose JSON schema is generated from the type hints (via pydantic) and
whose description comes from the docstring. A `ToolRegistry` holds the tools
for one agent, hands their specs to the model, and executes calls safely:
bad arguments or exceptions come back as error text the model can react to,
instead of crashing the run.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable
from typing import Any, get_type_hints

from pydantic import BaseModel, ValidationError, create_model


class Tool:
    def __init__(self, fn: Callable[..., Any], name: str | None = None) -> None:
        self.fn = fn
        self.name = name or fn.__name__
        self.description = inspect.getdoc(fn) or self.name
        hints = get_type_hints(fn, include_extras=True)
        fields: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param.name] = (hints.get(param.name, Any), default)
        self.input_model: type[BaseModel] = create_model(f"{self.name}_input", **fields)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)

    def spec(self) -> dict[str, Any]:
        """The provider-neutral spec handed to the model: name, description, JSON schema."""
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {"name": self.name, "description": self.description, "parameters": schema}


def tool(fn: Callable[..., Any] | None = None, *, name: str | None = None) -> Any:
    """Turn a typed function into a `Tool`. Use as `@tool` or `@tool(name="...")`."""
    if fn is None:
        return lambda f: Tool(f, name=name)
    return Tool(fn, name=name)


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools:
            self.register(t)

    def register(self, t: Tool) -> None:
        if t.name in self._tools:
            raise ValueError(f"Duplicate tool name: {t.name}")
        self._tools[t.name] = t

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict[str, Any]]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Run a tool and always return a string — errors included."""
        t = self._tools.get(name)
        if t is None:
            return f"Error: unknown tool {name!r}. Available: {', '.join(self._tools)}"
        try:
            args = t.input_model.model_validate(arguments)
        except ValidationError as exc:
            return f"Error: invalid arguments for {name}: {exc.errors(include_url=False)}"
        try:
            return str(t.fn(**dict(args)))
        except Exception as exc:  # the model should see the failure and adapt
            return f"Error: {name} failed: {type(exc).__name__}: {exc}"


def default_tools() -> ToolRegistry:
    """A registry with the three example tools."""
    from harness.tools.calculator import calculator
    from harness.tools.file_read import file_read
    from harness.tools.web_fetch import web_fetch

    return ToolRegistry([calculator, web_fetch, file_read])
