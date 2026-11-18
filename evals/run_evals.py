"""Evals: measure the whole harness, not just the model.

Runs the agent over every task in dataset.jsonl, scores each answer with
normalized exact match and (optionally, --judge) an LLM-as-judge, prints a
summary table, and saves results to runs/eval-<timestamp>.jsonl. Each case
also leaves a normal trace, linked by run_id, so failures are easy to debug.

    uv run python evals/run_evals.py            # exact match only
    uv run python evals/run_evals.py --judge    # + LLM-as-judge
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from harness.agent import DEFAULT_SYSTEM_PROMPT, Agent
from harness.model import Message, ModelClient, client_from_env
from harness.tools import default_tools

EVAL_SYSTEM_PROMPT = DEFAULT_SYSTEM_PROMPT + (
    " Reply with ONLY the final answer: no explanation, no units, no trailing punctuation."
)

JUDGE_PROMPT = """You are grading an AI assistant's answer.
Question: {question}
Reference answer: {expected}
Assistant's answer: {answer}

Does the assistant's answer convey the same result as the reference? Reply with exactly PASS or FAIL."""


def load_dataset(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def normalize(text: str) -> str:
    return " ".join(text.strip().strip(".'\"`*").lower().replace(",", "").split())


def exact_match(answer: str, expected: str) -> bool:
    a, e = normalize(answer), normalize(expected)
    try:
        return float(a) == float(e)  # "123.0" matches "123"
    except ValueError:
        return a == e


def llm_judge(model: ModelClient, question: str, expected: str, answer: str) -> bool:
    prompt = JUDGE_PROMPT.format(question=question, expected=expected, answer=answer)
    verdict = model.complete([Message(role="user", content=prompt)], tools=[]).text
    return verdict.strip().upper().startswith("PASS")


def run_evals(agent: Agent, dataset: list[dict[str, Any]], judge: ModelClient | None = None) -> list[dict[str, Any]]:
    results = []
    for case in dataset:
        result = agent.run(case["question"])
        row = {
            "id": case["id"],
            "expected": case["expected"],
            "answer": result.answer,
            "exact": exact_match(result.answer, case["expected"]),
            "judge": llm_judge(judge, case["question"], case["expected"], result.answer) if judge else None,
            "stop_reason": result.stop_reason,
            "steps": result.steps,
            "tokens": result.usage.input_tokens + result.usage.output_tokens,
            "run_id": result.run_id,
        }
        results.append(row)
        print(f"  {'✓' if row['exact'] else '✗'} {case['id']}")
    return results


def print_summary(results: list[dict[str, Any]]) -> None:
    print(f"\n{'id':<22} {'exact':<6} {'judge':<6} {'steps':<6} {'tokens':<7} answer")
    print("-" * 80)
    for r in results:
        judge = "-" if r["judge"] is None else ("pass" if r["judge"] else "fail")
        print(
            f"{r['id']:<22} {'pass' if r['exact'] else 'fail':<6} {judge:<6} "
            f"{r['steps']:<6} {r['tokens']:<7} {r['answer'][:40]!r}"
        )
    n = len(results) or 1
    print("-" * 80)
    print(f"exact match: {sum(r['exact'] for r in results)}/{len(results)} ({100 * sum(r['exact'] for r in results) / n:.0f}%)")
    judged = [r for r in results if r["judge"] is not None]
    if judged:
        print(f"llm judge:   {sum(r['judge'] for r in judged)}/{len(judged)}")


def save_results(results: list[dict[str, Any]], runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"eval-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in results))
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, default=Path(__file__).parent / "dataset.jsonl")
    parser.add_argument("--judge", action="store_true", help="also score with an LLM judge")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N cases")
    args = parser.parse_args()

    model = client_from_env()
    agent = Agent(model, default_tools(), system_prompt=EVAL_SYSTEM_PROMPT)
    dataset = load_dataset(args.dataset)[: args.limit]
    print(f"Running {len(dataset)} eval cases with {model.model}...")
    results = run_evals(agent, dataset, judge=model if args.judge else None)
    print_summary(results)
    print(f"\nsaved: {save_results(results, Path('runs'))}")


if __name__ == "__main__":
    main()
