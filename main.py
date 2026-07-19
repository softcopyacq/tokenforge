"""
TokenForge — Entrypoint
AMD Developer Hackathon: ACT II — Track 1

Usage:
  python main.py                              # sample tasks
  python main.py --tasks tasks.json
  python main.py --tasks tasks.json --threshold 0.6
  python main.py --tasks tasks.json --sweep   # find best threshold automatically
"""

import argparse
import json
import sys

from routing_agent import process_batch, calculate_metrics

SAMPLE_TASKS = [
    {"prompt": "What is 15 * 3?",                           "expected_answer": 45},
    {"prompt": "Hi!",                                        "expected_answer": None},
    {"prompt": "1 + 1",                                      "expected_answer": 2},
    {"prompt": "Good morning!",                              "expected_answer": None},
    {"prompt": "What is 20% of 250?",                        "expected_answer": 50},
    {"prompt": "Square root of 144",                         "expected_answer": 12},
    {"prompt": "Convert 100 Celsius to Fahrenheit",          "expected_answer": 212},
    {"prompt": "How many days in a week?",                   "expected_answer": "7"},
    {"prompt": "What is pi?",                                "expected_answer": "3.14"},
    {"prompt": "Summarize the history of the internet.",     "expected_answer": None},
    {"prompt": "Write a Python function to reverse a list.", "expected_answer": None},
    {"prompt": "Who won the 2022 FIFA World Cup?",           "expected_answer": "Argentina"},
    {"prompt": "Explain quantum entanglement.",              "expected_answer": None},
    {"prompt": "What is 2 ** 10?",                           "expected_answer": 1024},
]


def run(tasks, threshold, output_path):
    print(f"\nThreshold: {threshold} — processing {len(tasks)} task(s)...\n")
    results = process_batch(tasks, threshold=threshold)
    metrics = calculate_metrics(results)

    for r in results:
        stage = r.get("routing_stage", "?")
        correct_str = ("✓" if r["correct"] else "✗") if r["correct"] is not None else "-"
        print(
            f"[{r['routing_decision'].upper():6}][{stage:8}] "
            f"score={r['semantic_score']:.2f} tokens={r['tokens']:>3} "
            f"{correct_str} | {r['prompt'][:60]}"
        )

    print("\n--- Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    with open(output_path, "w") as f:
        json.dump({"metrics": metrics, "results": results}, f, indent=2)
    print(f"\nResults saved → {output_path}")
    return metrics


def sweep(tasks, output_path):
    """Try thresholds from 0.4 to 0.85 and report accuracy vs token tradeoff."""
    print("\n=== Threshold Sweep ===")
    print(f"{'Threshold':>10} {'Local%':>8} {'Tokens':>8} {'Accuracy':>10}")
    print("-" * 42)

    best = None
    for t in [round(x * 0.05, 2) for x in range(8, 18)]:  # 0.40 → 0.85
        results = process_batch(tasks, threshold=t)
        m = calculate_metrics(results)
        acc = m["accuracy"]
        local_pct = round(m["local_ratio"] * 100, 1)
        tokens = m["actual_tokens"]
        flag = " ← 100% acc" if acc == 1.0 or acc is None else ""
        print(f"{t:>10.2f} {local_pct:>7}% {tokens:>8} {str(acc):>10}{flag}")

        if (acc == 1.0 or acc is None) and (best is None or tokens < best["tokens"]):
            best = {"threshold": t, "tokens": tokens, "accuracy": acc, "local_pct": local_pct}

    if best:
        print(f"\n✓ Best threshold: {best['threshold']} "
              f"— {best['local_pct']}% local, {best['tokens']} tokens, {best['accuracy']*100 if best['accuracy'] else 'N/A'}% accuracy")
        print(f"\nRe-running with best threshold for full output...")
        run(tasks, best["threshold"], output_path)
    else:
        print("\nNo threshold achieved 100% accuracy — review local model coverage.")


def main():
    parser = argparse.ArgumentParser(description="TokenForge routing agent")
    parser.add_argument("--tasks",     type=str,   default=None)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output",    type=str,   default="results.json")
    parser.add_argument("--sweep",     action="store_true",
                        help="Sweep thresholds to find optimal token/accuracy tradeoff")
    args = parser.parse_args()

    if args.tasks:
        with open(args.tasks) as f:
            tasks = json.load(f)
    else:
        print("No --tasks provided, using built-in sample tasks.")
        tasks = SAMPLE_TASKS

    if args.sweep:
        sweep(tasks, args.output)
    else:
        run(tasks, args.threshold, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
