"""Quickstart: ask one question, watch the agent use a tool, then read the trace.

    uv run python examples/quickstart.py          # uses the model from your .env
    uv run python examples/quickstart.py --fake   # offline: a scripted fake model
"""

from __future__ import annotations

import sys

from harness import Agent, ModelResponse, ScriptedClient, ToolCall, client_from_env, default_tools
from harness.logging import load_trace

QUESTION = "What is 1234 * 5678?"


def fake_client() -> ScriptedClient:
    """Pretends to be a model: first asks for the calculator, then answers."""
    return ScriptedClient(
        [
            ModelResponse(
                text="I'll use the calculator for that.",
                tool_calls=[ToolCall(id="call_1", name="calculator", arguments={"expression": "1234 * 5678"})],
            ),
            ModelResponse(text="1234 * 5678 = 7,006,652."),
        ]
    )


def main() -> None:
    model = fake_client() if "--fake" in sys.argv else client_from_env()
    agent = Agent(model, default_tools())

    print(f"Q: {QUESTION}  (model: {model.model})\n")
    result = agent.run(QUESTION)
    print(f"A: {result.answer}\n")

    print(f"Trace ({result.trace_path}):")
    for event in load_trace(result.trace_path):
        if event["event"] == "model_call":
            calls = ", ".join(c["name"] for c in event["tool_calls"]) or "none"
            print(f"  step {event['step']}: model_call  {event['latency_ms']}ms  tools requested: {calls}")
        elif event["event"] == "tool_call":
            print(f"  step {event['step']}: tool_call   {event['name']}({event['arguments']}) -> {event['output']}")
        else:
            print(f"  {event['event']}")
    print(f"\nRate it: uv run python -m harness feedback {result.run_id} up \"nice\"")


if __name__ == "__main__":
    main()
