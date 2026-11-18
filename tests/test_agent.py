"""Agent loop with a scripted fake model: no API calls."""

from __future__ import annotations

from pathlib import Path

from harness.agent import Agent
from harness.logging import load_trace
from harness.model import ModelResponse, ScriptedClient, ToolCall, Usage
from harness.tools import default_tools


def calc_call(expr: str, id: str = "c1") -> ModelResponse:
    return ModelResponse(
        tool_calls=[ToolCall(id=id, name="calculator", arguments={"expression": expr})],
        usage=Usage(input_tokens=10, output_tokens=5),
    )


def make_agent(responses: list[ModelResponse], tmp_path: Path, **kw: int) -> tuple[Agent, ScriptedClient]:
    model = ScriptedClient(responses)
    return Agent(model, default_tools(), runs_dir=tmp_path, **kw), model


def test_direct_answer(tmp_path: Path) -> None:
    agent, _ = make_agent([ModelResponse(text="Hi!")], tmp_path)
    result = agent.run("hello")
    assert (result.answer, result.stop_reason, result.steps) == ("Hi!", "final_answer", 1)


def test_tool_call_then_answer(tmp_path: Path) -> None:
    agent, model = make_agent(
        [calc_call("6 * 7"), ModelResponse(text="42", usage=Usage(input_tokens=20, output_tokens=1))], tmp_path
    )
    result = agent.run("What is 6 * 7?")

    assert result.answer == "42" and result.stop_reason == "final_answer" and result.steps == 2
    assert result.usage == Usage(input_tokens=30, output_tokens=6)
    # The model's second call saw the tool result, linked to its call id.
    tool_msg = model.calls[1][-1]
    assert (tool_msg.role, tool_msg.content, tool_msg.tool_call_id) == ("tool", "42", "c1")

    events = [e["event"] for e in load_trace(result.trace_path)]
    assert events == ["run_start", "model_call", "tool_call", "model_call", "run_end"]
    assert result.trace_path.parent == tmp_path


def test_stops_at_max_steps(tmp_path: Path) -> None:
    agent, _ = make_agent([calc_call("1+1", id=f"c{i}") for i in range(5)], tmp_path, max_steps=3)
    result = agent.run("loop forever")
    assert result.stop_reason == "max_steps" and result.steps == 3


def test_model_error_stops_cleanly(tmp_path: Path) -> None:
    agent, _ = make_agent([], tmp_path)  # ScriptedClient raises when empty
    result = agent.run("anything")
    assert result.stop_reason == "model_error"
    assert load_trace(result.trace_path)[-1]["stop_reason"] == "model_error"


def test_tool_errors_are_fed_back_not_raised(tmp_path: Path) -> None:
    bad = ModelResponse(tool_calls=[ToolCall(id="c1", name="does_not_exist", arguments={})])
    agent, model = make_agent([bad, ModelResponse(text="Sorry")], tmp_path)
    result = agent.run("try a missing tool")
    assert result.stop_reason == "final_answer"
    assert model.calls[1][-1].content.startswith("Error: unknown tool")
