"""Tracing: a structured JSONL record of every run, saved to runs/<run_id>.jsonl.

You can't improve what you can't see. Each line is one event (run_start,
model_call, tool_call, model_error, run_end) with a timestamp, latency, and
token usage. Traces are plain JSONL so you can grep them, load them in a
notebook, or diff two runs. (This module is `harness.logging`; it does not
replace Python's standard `logging`.)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]


class Tracer:
    """Appends events for one run to its own JSONL file."""

    def __init__(self, runs_dir: Path | str = "runs", run_id: str | None = None) -> None:
        self.run_id = run_id or new_run_id()
        self.path = Path(runs_dir) / f"{self.run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, **data: Any) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "event": event,
            **data,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


def load_trace(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
