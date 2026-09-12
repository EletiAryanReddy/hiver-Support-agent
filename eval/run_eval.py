"""
Full evaluation harness. Produces:
  eval/predictions.jsonl   - raw pipeline output per golden example
  eval/eval_report.json    - aggregated headline metrics

Usage:
    export OPENAI_API_KEY=sk-...
    python -m eval.run_eval

Requires OPENAI_API_KEY. Without it, this script cannot run the live
pipeline (see README for the no-API-key fallback demonstration path using
eval/baselines.py and eval/human_calibration.py, which need no key).
"""

import json
import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

GOLDEN_PATH = Path(__file__).parent / "golden_set.jsonl"
PREDICTIONS_PATH = Path(__file__).parent / "predictions.jsonl"
REPORT_PATH = Path(__file__).parent / "eval_report.json"


def load_golden():
    records = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def run_pipeline_over_golden(golden_records, use_mock=False):
    from pipeline.retrieval import HistoricalIndex
    from pipeline.run import process_message

    if use_mock:
        from eval.mock_openai import MockOpenAI
        client = MockOpenAI()
    else:
        from pipeline.intents import get_client
        client = get_client()
    holdout_ids = {r["thread_id"] for r in golden_records if r.get("thread_id") is not None}
    index = HistoricalIndex(holdout_thread_ids=holdout_ids)

    predictions = []
    for i, rec in enumerate(golden_records):
        result = process_message(rec["customer_text"], index, client=client)
        result["id"] = rec["id"]
        predictions.append(result)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(golden_records)} processed", file=sys.stderr)

    with open(PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p) + "\n")
    return predictions


def score_intent_and_escalation(golden_records, predictions):
    id_to_gold = {r["id"]: r for r in golden_records}
    total = len(predictions)
    correct_intent = 0
    correct_intent_non_ambiguous = 0
    total_non_ambiguous = 0
    correct_escalation = 0

    confusion = defaultdict(lambda: defaultdict(int))
    per_intent_correct = defaultdict(int)
    per_intent_total = defaultdict(int)

    for p in predictions:
        gold = id_to_gold[p["id"]]
        gold_intent = gold["gold_intent"]
        pred_intent = p["predicted_intent"]

        per_intent_total[gold_intent] += 1
        if pred_intent == gold_intent:
            correct_intent += 1
            per_intent_correct[gold_intent] += 1
        confusion[gold_intent][pred_intent] += 1

        if not gold["is_ambiguous"]:
            total_non_ambiguous += 1
            if pred_intent == gold_intent or pred_intent == gold.get("gold_intent_alt"):
                correct_intent_non_ambiguous += 1

        gold_auto = gold["gold_auto_handle"]
        pred_auto = p["escalation_decision"] == "auto_handle"
        if pred_auto == gold_auto:
            correct_escalation += 1

    return {
        "n": total,
        "intent_accuracy_strict": correct_intent / total,
        "intent_accuracy_excluding_ambiguous": correct_intent_non_ambiguous / total_non_ambiguous if total_non_ambiguous else None,
        "escalation_decision_accuracy": correct_escalation / total,
        "per_intent_recall": {k: per_intent_correct[k] / v for k, v in per_intent_total.items()},
        "confusion_matrix": {k: dict(v) for k, v in confusion.items()},
    }


def score_reply_quality(golden_records, predictions, use_mock=False):
    from eval.judge import judge_reply

    client = None
    if use_mock:
        from eval.mock_openai import MockOpenAI
        client = MockOpenAI()

    id_to_gold = {r["id"]: r for r in golden_records}
    judge_results = []
    for p in predictions:
        gold = id_to_gold[p["id"]]
        js = judge_reply(
            customer_text=p["customer_text"],
            grounding_examples=p["grounding_examples"],
            drafted_reply=p["drafted_reply"],
            client=client,
        )
        judge_results.append({"id": p["id"], **js})

    n = len(judge_results)
    avg_total = sum(j["total_score"] for j in judge_results) / n
    pct_acceptable = sum(1 for j in judge_results if j["acceptable_to_send_as_is"]) / n
    avg_by_dim = {
        dim: sum(j[dim] for j in judge_results) / n
        for dim in ["grounded", "correct_intent_handling", "tone", "safety"]
    }

    return {
        "n": n,
        "avg_total_score_out_of_8": avg_total,
        "pct_acceptable_to_send_as_is": pct_acceptable,
        "avg_score_by_dimension": avg_by_dim,
        "raw_judge_results": judge_results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mock", action="store_true",
        help="Use eval/mock_openai.py instead of a real OpenAI client. "
             "Use this if you don't have an OPENAI_API_KEY -- it validates "
             "the full pipeline and eval harness wiring with a realistic, "
             "deliberately-imperfect simulated classifier/judge. See "
             "README.md 'Running without an API key (mock mode)'.",
    )
    args = parser.parse_args()

    golden = load_golden()
    print(f"Loaded {len(golden)} golden examples")

    if not args.mock and not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set, and --mock not passed.")
        print("Either:")
        print("  1) export OPENAI_API_KEY=sk-...   (real results), or")
        print("  2) python -m eval.run_eval --mock  (offline pipeline/harness validation)")
        print("See README.md 'Running without an API key (mock mode)' for details.")
        sys.exit(1)

    if args.mock:
        print("Running in --mock mode (no live API calls; see README for what this proves/doesn't prove).")

    print("Running pipeline over golden set...")
    predictions = run_pipeline_over_golden(golden, use_mock=args.mock)

    print("Scoring intent classification + escalation decisions...")
    classification_metrics = score_intent_and_escalation(golden, predictions)

    print("Running LLM-judge on drafted replies...")
    quality_metrics = score_reply_quality(golden, predictions, use_mock=args.mock)

    with open(Path(__file__).parent / "baseline_results.json") as f:
        baseline_results = json.load(f)

    report = {
        "mode": "mock" if args.mock else "live_api",
        "classification_and_escalation": classification_metrics,
        "reply_quality": quality_metrics,
        "baselines": baseline_results,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== HEADLINE RESULTS ===")
    print(f"Intent accuracy (strict): {classification_metrics['intent_accuracy_strict']:.3f}")
    print(f"Intent accuracy (excl. ambiguous): {classification_metrics['intent_accuracy_excluding_ambiguous']:.3f}")
    print(f"Escalation decision accuracy: {classification_metrics['escalation_decision_accuracy']:.3f}")
    print(f"Avg reply quality score: {quality_metrics['avg_total_score_out_of_8']:.2f}/8")
    print(f"% replies acceptable to send as-is: {quality_metrics['pct_acceptable_to_send_as_is']:.1%}")
    print(f"\nFull report written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
