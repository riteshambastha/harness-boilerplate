"""Context management: decide what the model actually sees on each step.

The conversation grows every step (tool results can be large), but the model's
context window and your budget don't. Before each model call, `ContextManager`
1. injects retrieved documents into the system prompt (optional retriever hook),
2. trims the oldest turns so the messages fit a token budget, and
3. optionally replaces what was trimmed with a short LLM-written summary.

Token counts are a rough estimate (~4 characters per token). That's enough
for budgeting; use the provider's token counter if you need precision.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence

from harness.model import Message, ModelClient

Summarizer = Callable[[list[Message]], str]
Retriever = Callable[[str], list[str]]


def estimate_tokens(messages: Sequence[Message] | str) -> int:
    if isinstance(messages, str):
        return len(messages) // 4 + 1
    total = 0
    for m in messages:
        total += 4 + len(m.content) // 4
        total += sum(len(json.dumps(c.arguments)) // 4 + 8 for c in m.tool_calls)
    return total


def _group(messages: Sequence[Message]) -> list[list[Message]]:
    """Split into droppable units. A tool result always stays with the
    assistant message that requested it — providers reject orphaned results."""
    groups: list[list[Message]] = []
    for m in messages:
        if m.role == "tool" and groups:
            groups[-1].append(m)
        else:
            groups.append([m])
    return groups


def trim_messages(messages: Sequence[Message], budget: int) -> tuple[list[Message], list[Message]]:
    """Return (kept, dropped). Always keeps the latest user message (the task)
    and the newest group; fills the rest of the budget newest-first."""
    if estimate_tokens(messages) <= budget:
        return list(messages), []
    groups = _group(messages)
    users = [i for i, g in enumerate(groups) if g[0].role == "user"]
    keep = {users[-1] if users else 0, len(groups) - 1}
    used = sum(estimate_tokens(groups[i]) for i in keep)
    for i in range(len(groups) - 1, -1, -1):
        if i in keep:
            continue
        cost = estimate_tokens(groups[i])
        if used + cost > budget:
            break
        keep.add(i)
        used += cost
    # Providers expect the window to open with a user message.
    while len(keep) > 1 and groups[min(keep)][0].role != "user":
        keep.discard(min(keep))
    kept = [m for i, g in enumerate(groups) if i in keep for m in g]
    dropped = [m for i, g in enumerate(groups) if i not in keep for m in g]
    return kept, dropped


def llm_summarizer(model: ModelClient) -> Summarizer:
    """A summarizer that asks the model to compress trimmed history."""

    def summarize(messages: list[Message]) -> str:
        transcript = "\n".join(f"[{m.name or m.role}] {m.content}" for m in messages)
        prompt = (
            "Summarize this earlier part of a conversation in a few bullet points. "
            "Keep facts, numbers, and tool results the assistant may still need.\n\n" + transcript
        )
        return model.complete([Message(role="user", content=prompt)], tools=[]).text

    return summarize


class ContextManager:
    def __init__(
        self,
        max_tokens: int = 12_000,
        summarizer: Summarizer | None = None,
        retriever: Retriever | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.summarizer = summarizer
        self.retriever = retriever
        self._summaries: dict[str, str] = {}  # avoid re-summarizing the same history

    def build(self, system: str, messages: list[Message]) -> tuple[str, list[Message]]:
        """Return the (system prompt, messages) to send on this step."""
        if self.retriever:
            query = next((m.content for m in reversed(messages) if m.role == "user"), "")
            docs = self.retriever(query)
            if docs:
                body = "\n".join(f'<document index="{i}">\n{d}\n</document>' for i, d in enumerate(docs, 1))
                system += f"\n\nUse these documents if relevant:\n<documents>\n{body}\n</documents>"

        kept, dropped = trim_messages(messages, self.max_tokens - estimate_tokens(system))
        if dropped and self.summarizer:
            system += f"\n\n<earlier_conversation_summary>\n{self._summarize(dropped)}\n</earlier_conversation_summary>"
        return system, kept

    def _summarize(self, dropped: list[Message]) -> str:
        key = hashlib.sha256("".join(m.model_dump_json() for m in dropped).encode()).hexdigest()
        if key not in self._summaries:
            assert self.summarizer is not None
            self._summaries[key] = self.summarizer(dropped)
        return self._summaries[key]
