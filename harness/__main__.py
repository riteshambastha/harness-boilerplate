"""Command-line entry point.

    python -m harness "What is 17 * 23?"
    python -m harness feedback <run_id> up|down ["optional note"]
"""

from __future__ import annotations

import argparse
import sys

from harness.agent import Agent
from harness.feedback import record_feedback
from harness.model import client_from_env
from harness.tools import default_tools


def _feedback(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="python -m harness feedback")
    parser.add_argument("run_id")
    parser.add_argument("rating", choices=["up", "down"])
    parser.add_argument("note", nargs="?", default="")
    args = parser.parse_args(argv)
    record_feedback(args.run_id, args.rating, args.note)
    print(f"Recorded {args.rating} for {args.run_id} in runs/feedback.jsonl")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["feedback"]:
        return _feedback(argv[1:])

    parser = argparse.ArgumentParser(prog="python -m harness", description="Ask the agent a question.")
    parser.add_argument("question")
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args(argv)

    agent = Agent(client_from_env(), default_tools(), max_steps=args.max_steps)
    result = agent.run(args.question)
    print(result.answer)
    print(
        f"\n[{result.stop_reason} · {result.steps} step(s) · "
        f"{result.usage.input_tokens} in / {result.usage.output_tokens} out tokens]"
        f"\ntrace:    {result.trace_path}"
        f"\nfeedback: python -m harness feedback {result.run_id} up|down \"note\"",
        file=sys.stderr,
    )
    return 0 if result.stop_reason == "final_answer" else 1


if __name__ == "__main__":
    raise SystemExit(main())
