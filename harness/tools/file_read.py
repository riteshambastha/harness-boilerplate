"""Example tool: read a local text file.

Shows a guarded tool: the model can only read files under the current working
directory, and output is capped. Tools are where the harness enforces what
the model is allowed to touch — put the guardrails here, not in the prompt.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field

from harness.tools import tool


@tool
def file_read(
    path: Annotated[str, Field(description="File path, relative to the working directory")],
    max_chars: Annotated[int, Field(description="Truncate to this many characters", le=50_000)] = 8000,
) -> str:
    """Read a text file from the current working directory."""
    root = Path.cwd().resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"{path!r} is outside the working directory")
    text = target.read_text(encoding="utf-8", errors="replace")
    return text[:max_chars] + (" …[truncated]" if len(text) > max_chars else "")
