"""Tool registry: schema generation, safe execution, and the example tools."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import pytest
from pydantic import Field

from harness.tools import ToolRegistry, default_tools, tool
from harness.tools.calculator import calculator
from harness.tools.file_read import file_read


@tool
def greet(name: Annotated[str, Field(description="Who to greet")], times: int = 1) -> str:
    """Say hello."""
    return " ".join([f"hello {name}"] * times)


def test_schema_generated_from_type_hints() -> None:
    spec = greet.spec()
    assert spec["name"] == "greet"
    assert spec["description"] == "Say hello."
    props = spec["parameters"]["properties"]
    assert props["name"] == {"type": "string", "description": "Who to greet"}
    assert props["times"]["type"] == "integer"
    assert spec["parameters"]["required"] == ["name"]


def test_custom_name_and_direct_call() -> None:
    @tool(name="shout")
    def _shout(text: str) -> str:
        """Uppercase it."""
        return text.upper()

    assert _shout.name == "shout"
    assert _shout("hi") == "HI"


def test_execute_validates_and_reports_errors() -> None:
    reg = ToolRegistry([greet])
    assert reg.execute("greet", {"name": "Ada", "times": 2}) == "hello Ada hello Ada"
    assert reg.execute("greet", {"times": 2}).startswith("Error: invalid arguments")
    assert reg.execute("nope", {}).startswith("Error: unknown tool")


def test_tool_exceptions_become_error_text() -> None:
    @tool
    def boom() -> str:
        """Always fails."""
        raise RuntimeError("kaboom")

    assert "kaboom" in ToolRegistry([boom]).execute("boom", {})


def test_duplicate_names_rejected() -> None:
    with pytest.raises(ValueError):
        ToolRegistry([greet, greet])


def test_default_tools() -> None:
    assert default_tools().names() == ["calculator", "web_fetch", "file_read"]


def test_calculator() -> None:
    assert calculator("(17 + 25) * 3 - 6 / 2") == "123"
    assert calculator("2 ** 10") == "1024"
    reg = ToolRegistry([calculator])
    assert reg.execute("calculator", {"expression": "__import__('os')"}).startswith("Error")
    assert reg.execute("calculator", {"expression": "9 ** 99999"}).startswith("Error")


def test_file_read_stays_inside_working_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "notes.txt").write_text("hi there")
    monkeypatch.chdir(tmp_path)
    assert file_read("notes.txt") == "hi there"
    assert "outside" in ToolRegistry([file_read]).execute("file_read", {"path": "../../etc/passwd"})
