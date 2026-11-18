"""The model layer: one small interface, two adapters.

Everything else in the harness talks to a `ModelClient`. It takes neutral
`Message`s plus tool specs and returns a neutral `ModelResponse`. Adapters
translate to and from each provider's wire format, so swapping Anthropic for
OpenAI (or a scripted fake, in tests) never touches the agent loop.
"""

from __future__ import annotations

import json
import os
from typing import Any, Literal, Protocol

import anthropic
import openai
from dotenv import load_dotenv
from pydantic import BaseModel, Field

DEFAULT_MODELS = {"anthropic": "claude-sonnet-5-5", "openai": "gpt-5-mini"}


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """One turn in the conversation, in a provider-neutral shape."""

    role: Literal["user", "assistant", "tool"]
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)  # assistant only
    tool_call_id: str | None = None  # tool only: which call this answers
    name: str | None = None  # tool only: which tool produced it


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )


class ModelResponse(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    stop_reason: str | None = None


class ModelClient(Protocol):
    """Anything with a `complete` method like this can drive the agent."""

    model: str

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]], system: str = ""
    ) -> ModelResponse: ...


class AnthropicClient:
    def __init__(self, model: str, max_tokens: int = 4096) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]], system: str = ""
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": self._to_wire(messages),
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
                for t in tools
            ]
        resp = self._client.messages.create(**kwargs)
        return ModelResponse(
            text="".join(b.text for b in resp.content if b.type == "text"),
            tool_calls=[
                ToolCall(id=b.id, name=b.name, arguments=dict(b.input))
                for b in resp.content
                if b.type == "tool_use"
            ],
            usage=Usage(input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens),
            stop_reason=resp.stop_reason,
        )

    @staticmethod
    def _to_wire(messages: list[Message]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for m in messages:
            if m.role == "tool":
                # Anthropic wants tool results as blocks inside a *user* message;
                # consecutive results from one step share that message.
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                blocks: list[dict[str, Any]] = [{"type": "text", "text": m.content}] if m.content else []
                blocks += [
                    {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
                    for c in m.tool_calls
                ]
                out.append({"role": "assistant", "content": blocks or "(no content)"})
            else:
                out.append({"role": "user", "content": m.content})
        return out


class OpenAIClient:
    def __init__(self, model: str, max_tokens: int = 4096) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._client = openai.OpenAI()  # reads OPENAI_API_KEY

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]], system: str = ""
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_completion_tokens": self.max_tokens,
            "messages": self._to_wire(messages, system),
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        return ModelResponse(
            text=choice.message.content or "",
            tool_calls=[
                ToolCall(id=tc.id, name=tc.function.name, arguments=_parse_json(tc.function.arguments))
                for tc in choice.message.tool_calls or []
            ],
            usage=Usage(
                input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                output_tokens=resp.usage.completion_tokens if resp.usage else 0,
            ),
            stop_reason=choice.finish_reason,
        )

    @staticmethod
    def _to_wire(messages: list[Message], system: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m.role == "tool":
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
            elif m.role == "assistant":
                msg: dict[str, Any] = {"role": "assistant", "content": m.content or None}
                if m.tool_calls:
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                        }
                        for c in m.tool_calls
                    ]
                out.append(msg)
            else:
                out.append({"role": "user", "content": m.content})
        return out


def _parse_json(raw: str | None) -> dict[str, Any]:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {"_invalid_json": raw}  # let tool validation report it to the model
    return value if isinstance(value, dict) else {"_invalid_json": raw}


class ScriptedClient:
    """A fake model that replays canned responses. No network, no API key.

    Use it in tests and offline demos. `calls` records what the agent sent,
    so tests can assert on exactly what the model saw.
    """

    def __init__(self, responses: list[ModelResponse], model: str = "scripted") -> None:
        self.model = model
        self._responses = list(responses)
        self.calls: list[list[Message]] = []

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]], system: str = ""
    ) -> ModelResponse:
        self.calls.append(list(messages))
        if not self._responses:
            raise RuntimeError("ScriptedClient ran out of responses")
        return self._responses.pop(0)


def client_from_env() -> ModelClient:
    """Build a client from HARNESS_PROVIDER and HARNESS_MODEL (read from .env)."""
    load_dotenv()
    provider = os.getenv("HARNESS_PROVIDER", "anthropic").strip().lower()
    if provider not in DEFAULT_MODELS:
        raise ValueError(f"HARNESS_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {provider!r}")
    model = os.getenv("HARNESS_MODEL") or DEFAULT_MODELS[provider]
    return AnthropicClient(model) if provider == "anthropic" else OpenAIClient(model)
