"""Context management: trimming, summarization, and retrieval injection."""

from __future__ import annotations

from harness.context import ContextManager, estimate_tokens, trim_messages
from harness.model import Message, ToolCall


def step(i: int, size: int = 400) -> list[Message]:
    """One agent step: an assistant tool call plus its result."""
    call = ToolCall(id=f"c{i}", name="calculator", arguments={"expression": "1+1"})
    return [
        Message(role="assistant", content=f"step {i}", tool_calls=[call]),
        Message(role="tool", content="x" * size, tool_call_id=f"c{i}", name="calculator"),
    ]


def conversation(n_steps: int) -> list[Message]:
    msgs = [Message(role="user", content="the task")]
    for i in range(n_steps):
        msgs += step(i)
    return msgs


def test_under_budget_is_unchanged() -> None:
    msgs = conversation(2)
    kept, dropped = trim_messages(msgs, budget=10_000)
    assert kept == msgs and dropped == []


def test_over_budget_drops_oldest_and_keeps_task() -> None:
    msgs = conversation(10)
    budget = estimate_tokens(msgs) // 2
    kept, dropped = trim_messages(msgs, budget)
    assert estimate_tokens(kept) <= budget
    assert kept[0].content == "the task"
    assert kept[-1] == msgs[-1]  # newest step survives
    assert dropped[0].content == "step 0"  # oldest goes first
    assert len(kept) + len(dropped) == len(msgs)


def test_tool_results_never_orphaned() -> None:
    kept, _ = trim_messages(conversation(10), budget=300)
    ids = {c.id for m in kept for c in m.tool_calls}
    assert all(m.tool_call_id in ids for m in kept if m.role == "tool")


def test_newest_step_kept_even_if_huge() -> None:
    msgs = [Message(role="user", content="task"), *step(0, size=100_000)]
    kept, _ = trim_messages(msgs, budget=100)
    assert kept == msgs


def test_window_starts_with_user_message() -> None:
    history = conversation(3) + [Message(role="assistant", content="done")]
    msgs = history + [Message(role="user", content="follow-up")] + step(99)
    kept, _ = trim_messages(msgs, budget=estimate_tokens(msgs) - 50)
    assert kept[0].role == "user"


def test_summarizer_called_once_for_same_history() -> None:
    seen: list[int] = []

    def summarizer(dropped: list[Message]) -> str:
        seen.append(len(dropped))
        return "SUMMARY"

    ctx = ContextManager(max_tokens=400, summarizer=summarizer)
    msgs = conversation(10)
    system, kept = ctx.build("base", msgs)
    ctx.build("base", msgs)
    assert "SUMMARY" in system and system.startswith("base")
    assert len(kept) < len(msgs)
    assert len(seen) == 1


def test_retriever_docs_injected() -> None:
    queries: list[str] = []

    def retriever(query: str) -> list[str]:
        queries.append(query)
        return ["Doc about llamas."]

    system, _ = ContextManager(retriever=retriever).build("base", conversation(1))
    assert queries == ["the task"]
    assert "<documents>" in system and "Doc about llamas." in system
