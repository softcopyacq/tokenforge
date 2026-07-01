"""
Entrypoint for the Hybrid Token-Efficient Routing Agent.

Usage:
    python main.py                          # runs the built-in sample task set
    python main.py --tasks path/to/tasks.json
    python main.py --tasks tasks.json --output results.json

tasks.json format:
[
  {"prompt": "What is 12 * 4?", "expected_answer": 48},
  {"prompt": "Hello!", "expected_answer": null},
  {"prompt": "Explain quantum entanglement.", "expected_answer": null}
]
"""

import argparse
import json
import sys

from routing_agent import process_batch, calculate_metrics

SAMPLE_TASKS = [
    {"prompt": "What is 15 * 3?", "expected_answer": 45},
    {"prompt": "Hi!", "expected_answer": None},
    {"prompt": "1 + 1", "expected_answer": 2},
    {"prompt": "Good morning!", "expected_answer": None},
    {"prompt": "Summarize the history of the internet in 3 paragraphs.", "expected_answer": None},
    {"prompt": "How do I bake a cake?", "expected_answer": None},
    {"prompt": "Who won the world cup in 2022?", "expected_answer": "Argentina"},
    {"prompt": "Write a Python function to sort a list.", "expected_answer": None},
]


def main():
    parser = argparse.ArgumentParser(description="Run the hybrid routing agent over a task list.")
    parser.add_argument("--tasks", type=str, default=None, help="Path to a JSON file of tasks.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Semantic routing threshold.")
    parser.add_argument("--output", type=str, default="results.json", help="Where to save results.")
    args = parser.parse_args()

    if args.tasks:
        with open(args.tasks, "r") as f:
            tasks = json.load(f)
    else:
        print("No --tasks file provided, running built-in sample task set.")
        tasks = SAMPLE_TASKS

    print(f"Processing {len(tasks)} task(s) autonomously...\n")
    results = process_batch(tasks, threshold=args.threshold)
    metrics = calculate_metrics(results)

    for r in results:
        print(
            f"[{r['routing_decision'].upper():6}] "
            f"score={r['semantic_score']:.2f} "
            f"tokens={r['tokens']:>3} "
            f"correct={r['correct']} | {r['prompt'][:60]}"
        )

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    with open(args.output, "w") as f:
        json.dump({"results": results, "metrics": metrics}, f, indent=2)
    print(f"\nSaved full results to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())