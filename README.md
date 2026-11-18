# harness-starter

**It's not just the model.** A model call turns text into text. What turns that into a working AI application is everything *around* the model: a loop that lets it act, tools that let it touch the world safely, context management that decides what it sees, evals that tell you whether it's any good, and traces plus feedback that tell you what actually happened. That surrounding code is the *harness*. This repo is a small, readable one (about 800 lines of core code including docstrings, no agent frameworks) that you can understand in an afternoon and grow into your own.

## The five layers

**1. Model — `harness/model.py`**
One tiny interface, `ModelClient.complete(messages, tools, system) -> ModelResponse`, with adapters for Anthropic and OpenAI. The rest of the harness only speaks this neutral format, so switching providers is a one-line `.env` change and tests can swap in `ScriptedClient`, a fake that replays canned responses. *Why:* provider lock-in and untestable code both start when API calls leak everywhere.

**2. Agent loop — `harness/agent.py`**
Ask the model → if it requested tools, run them and send back the results → repeat. It stops on exactly one of three conditions: `final_answer` (the model replied without asking for a tool), `max_steps` (budget exhausted), or `model_error`. Tool failures don't crash the run; the error text goes back to the model, which can usually recover. *Why:* the loop is what turns a chatbot into something that can *do* things, and clear stop conditions keep it from running forever.

**3. Tools — `harness/tools/`**
Decorate a typed function with `@tool`; its JSON schema is generated from the type hints (via pydantic) and its description from the docstring. The `ToolRegistry` validates arguments before calling. Three examples: `calculator` (safe AST evaluation, never `eval`), `web_fetch` (timeouts, output caps), and `file_read` (can't leave the working directory). *Why:* tools are where the model gets real capability, and where you enforce what it's allowed to touch.

**4. Context — `harness/context.py`**
Before every model call, `ContextManager` trims the oldest turns to fit a token budget (always keeping the task and the latest step, and never separating a tool result from its call). It can also summarize what was trimmed (`llm_summarizer`) and inject retrieved documents through a `retriever` hook (plug in your search or vector store here). *Why:* conversations grow every step. Without a policy, you hit context limits, pay for stale tokens, and the model gets distracted.

**5. Observability: traces, feedback, evals — `harness/logging.py`, `harness/feedback.py`, `evals/`**
Every run writes `runs/<run_id>.jsonl`: each model call (latency, tokens, what it asked for) and each tool call (arguments, output, latency). `record_feedback` attaches 👍/👎 plus a note to a run ID. `evals/run_evals.py` runs the agent over a dataset and scores it with exact match and an optional LLM judge. *Why:* you can't improve what you can't see or measure. Change a prompt, rerun the evals, compare.

## 5-minute quickstart

```bash
git clone https://github.com/riteshambastha/harness-boilerplate.git
cd harness-boilerplate
uv sync
```

See it work offline first (a scripted fake model, no API key):

```bash
uv run python examples/quickstart.py --fake
```

Then add a real model:

```bash
cp .env.example .env   # set HARNESS_PROVIDER, HARNESS_MODEL and the matching API key
uv run python examples/quickstart.py
uv run python -m harness "What is 2 to the power of 20, minus 1000?"
uv run python -m harness feedback <run_id> up "correct and fast"
uv run python evals/run_evals.py --judge
uv run pytest
```

Open any file in `runs/` to see exactly what happened in each run.

## How to extend

**Add a tool.** Create `harness/tools/weather.py`:

```python
from typing import Annotated
from pydantic import Field
from harness.tools import tool

@tool
def weather(city: Annotated[str, Field(description="City name")]) -> str:
    """Get the current weather for a city."""
    return f"It's sunny in {city}."
```

Then register it: add it to `default_tools()` in `harness/tools/__init__.py`, or build your own `ToolRegistry([calculator, weather])` and pass it to `Agent`. Return strings, cap your output size, and raise on bad input; the registry turns exceptions into messages the model can read.

**Swap a model.** Change `HARNESS_PROVIDER` / `HARNESS_MODEL` in `.env`. To add a new provider, write a class with a `model` attribute and a `complete(messages, tools, system) -> ModelResponse` method (copy `OpenAIClient` as a template; many providers expose an OpenAI-compatible API) and return it from `client_from_env()`.

**Add an eval.** Append a line to `evals/dataset.jsonl`:

```json
{"id": "calc-percent", "question": "What is 15% of 240?", "expected": "36"}
```

A good habit: every thumbs-down in `runs/feedback.jsonl` becomes a new eval case once you know the right answer.

**Plug in retrieval or summarization.**

```python
from harness import Agent, ContextManager, client_from_env, default_tools
from harness.context import llm_summarizer

model = client_from_env()
context = ContextManager(max_tokens=8000, summarizer=llm_summarizer(model), retriever=my_search)
agent = Agent(model, default_tools(), context=context)
```

## Layout

```
harness/      model.py · agent.py · context.py · logging.py · feedback.py · tools/
evals/        dataset.jsonl · run_evals.py
examples/     quickstart.py
tests/        tool registry, context trimming, agent loop (fake model, no API calls)
```

MIT licensed.
