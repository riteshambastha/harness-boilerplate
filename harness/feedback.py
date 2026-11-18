"""Feedback: attach a human judgment (thumbs up/down + note) to a run.

Traces tell you what happened; feedback tells you whether it was good.
Entries go to runs/feedback.jsonl keyed by run ID, so you can join them with
traces later — e.g. turn every thumbs-down into a new eval case.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

Rating = Literal["up", "down"]


class Feedback(BaseModel):
    run_id: str
    rating: Rating
    note: str = ""
    ts: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))


def record_feedback(run_id: str, rating: Rating, note: str = "", runs_dir: Path | str = "runs") -> Feedback:
    entry = Feedback(run_id=run_id, rating=rating, note=note)
    path = Path(runs_dir) / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(entry.model_dump_json() + "\n")
    return entry


def load_feedback(runs_dir: Path | str = "runs") -> list[Feedback]:
    path = Path(runs_dir) / "feedback.jsonl"
    if not path.exists():
        return []
    return [Feedback.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
