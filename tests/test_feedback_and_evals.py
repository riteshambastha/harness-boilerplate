"""Feedback storage and eval scoring helpers."""

from __future__ import annotations

from pathlib import Path

from evals.run_evals import exact_match, run_evals
from harness.agent import Agent
from harness.feedback import load_feedback, record_feedback
from harness.model import ModelResponse, ScriptedClient
from harness.tools import ToolRegistry


def test_feedback_roundtrip(tmp_path: Path) -> None:
    record_feedback("run-1", "up", "great", runs_dir=tmp_path)
    record_feedback("run-2", "down", runs_dir=tmp_path)
    entries = load_feedback(tmp_path)
    assert [(e.run_id, e.rating, e.note) for e in entries] == [("run-1", "up", "great"), ("run-2", "down", "")]


def test_exact_match_normalizes() -> None:
    assert exact_match("7,006,652.", "7006652")
    assert exact_match("123.0", "123")
    assert exact_match(" Canberra. ", "canberra")
    assert not exact_match("Sydney", "Canberra")


def test_run_evals_with_fake_model(tmp_path: Path) -> None:
    model = ScriptedClient([ModelResponse(text="Canberra"), ModelResponse(text="8")])
    agent = Agent(model, ToolRegistry(), runs_dir=tmp_path)
    dataset = [
        {"id": "a", "question": "capital of Australia?", "expected": "Canberra"},
        {"id": "b", "question": "letters in harness?", "expected": "7"},
    ]
    results = run_evals(agent, dataset)
    assert [r["exact"] for r in results] == [True, False]
