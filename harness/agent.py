"""The agent loop: the heart of the harness.

    ask the model -> if it requested tools, run them and feed back the results -> repeat

The loop stops for exactly one of three reasons:
  - "final_answer": the model replied without requesting any tool
  - "max_steps":    the step budget ran out (protects against loops and cost blowups)
  - "model_error":  the model call raised (network, auth, rate limit, ...)

Tool failures do NOT stop the loop: the error text is returned to the model,
which can usually recover. Every step is written to the run's trace.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from harness.context import ContextManager
from harness.logging import Tracer
from harness.model import Message, ModelClient, Usage
from harness.tools import ToolRegistry

StopReason = Literal["final_answer", "max_steps", "model_error"]

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the available tools whenever they help you "
    "answer accurately (for example, use the calculator for arithmetic). "
    "When you have the answer, reply with it directly and concisely."
)


class AgentResult(BaseModel):
    run_id: str
    answer: str
    stop_reason: StopReason
    steps: int
    usage: Usage
    messages: list[Message]
    trace_path: Path


class Agent:
    def __init__(
        self,
        model: ModelClient,
        tools: ToolRegistry,
        *,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 8,
        context: ContextManager | None = None,
        runs_dir: Path | str = "runs",
    ) -> None:
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.context = context or ContextManager()
        self.runs_dir = Path(runs_dir)

    def run(self, question: str, history: Sequence[Message] = ()) -> AgentResult:
        tracer = Tracer(self.runs_dir)
        messages = [*history, Message(role="user", content=question)]
        tracer.log("run_start", question=question, model=self.model.model, tools=self.tools.names())

        usage = Usage()
        answer, stop_reason, step = "", "max_steps", 0
        for step in range(1, self.max_steps + 1):
            system, window = self.context.build(self.system_prompt, messages)
            started = time.perf_counter()
            try:
                response = self.model.complete(window, self.tools.specs(), system)
            except Exception as exc:
                tracer.log("model_error", step=step, error=f"{type(exc).__name__}: {exc}")
                answer, stop_reason = f"Model call failed: {exc}", "model_error"
                break
            usage += response.usage
            tracer.log(
                "model_call",
                step=step,
                latency_ms=round((time.perf_counter() - started) * 1000, 1),
                messages_sent=len(window),
                messages_total=len(messages),
                usage=response.usage.model_dump(),
                stop_reason=response.stop_reason,
                text=response.text,
                tool_calls=[c.model_dump() for c in response.tool_calls],
            )
            messages.append(Message(role="assistant", content=response.text, tool_calls=response.tool_calls))

            if not response.tool_calls:
                answer, stop_reason = response.text, "final_answer"
                break

            for call in response.tool_calls:
                started = time.perf_counter()
                output = self.tools.execute(call.name, call.arguments)
                tracer.log(
                    "tool_call",
                    step=step,
                    name=call.name,
                    arguments=call.arguments,
                    output=output,
                    latency_ms=round((time.perf_counter() - started) * 1000, 1),
                )
                messages.append(Message(role="tool", content=output, tool_call_id=call.id, name=call.name))

        if stop_reason == "max_steps":
            answer = f"Stopped after {self.max_steps} steps without a final answer."

        tracer.log("run_end", stop_reason=stop_reason, steps=step, answer=answer, usage=usage.model_dump())
        return AgentResult(
            run_id=tracer.run_id,
            answer=answer,
            stop_reason=stop_reason,
            steps=step,
            usage=usage,
            messages=messages,
            trace_path=tracer.path,
        )
