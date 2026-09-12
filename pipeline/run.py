"""
End-to-end pipeline: customer message -> intent + confidence -> retrieved
grounding examples -> drafted reply -> escalation decision + reason.

Usage:
    python -m pipeline.run --text "my iphone battery drains so fast"
    python -m pipeline.run --input eval/golden_set.jsonl --output eval/predictions.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.intents import classify_intent, get_client as get_intent_client
from pipeline.retrieval import HistoricalIndex
from pipeline.draft import draft_reply
from pipeline.escalation import decide_escalation


def process_message(customer_text: str, index: HistoricalIndex, client=None) -> dict:
    intent_result = classify_intent(customer_text, client=client)
    intent = intent_result["intent"]
    confidence = intent_result.get("confidence", 0.0)

    matches = index.search(customer_text, k=3)

    esc = decide_escalation(customer_text, intent, confidence)

    if esc["decision"] == "auto_handle":
        reply = draft_reply(customer_text, matches, client=client)
    else:
        # Still draft a SUGGESTED reply for the human agent to review/edit --
        # escalation means "needs human sign-off", not "agent produces nothing".
        reply = draft_reply(customer_text, matches, client=client)

    return {
        "customer_text": customer_text,
        "predicted_intent": intent,
        "intent_confidence": confidence,
        "intent_rationale": intent_result.get("rationale", ""),
        "escalation_decision": esc["decision"],
        "escalation_reason": esc["reason"],
        "drafted_reply": reply,
        "grounding_examples": [
            {"customer_text": m.customer_text, "response": m.historical_response_text, "similarity": m.similarity}
            for m in matches
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single customer message to process")
    parser.add_argument("--input", type=str, help="JSONL file with 'customer_text' field per line")
    parser.add_argument("--output", type=str, help="Where to write predictions JSONL")
    parser.add_argument(
        "--holdout-from-index",
        action="store_true",
        help="Exclude thread_ids present in --input from the retrieval index (use for eval runs)",
    )
    args = parser.parse_args()

    client = get_intent_client()

    if args.text:
        holdout_ids = None
        index = HistoricalIndex(holdout_thread_ids=holdout_ids)
        result = process_message(args.text, index, client=client)
        print(json.dumps(result, indent=2))
        return

    if args.input:
        records = []
        with open(args.input, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        holdout_ids = None
        if args.holdout_from_index:
            holdout_ids = {r["thread_id"] for r in records if "thread_id" in r}

        index = HistoricalIndex(holdout_thread_ids=holdout_ids)

        out_path = args.output or "predictions.jsonl"
        with open(out_path, "w", encoding="utf-8") as out_f:
            for i, rec in enumerate(records):
                result = process_message(rec["customer_text"], index, client=client)
                result["thread_id"] = rec.get("thread_id")
                result["true_intent"] = rec.get("true_intent")
                result["true_auto_handle"] = rec.get("true_auto_handle")
                out_f.write(json.dumps(result) + "\n")
                if (i + 1) % 25 == 0:
                    print(f"  processed {i+1}/{len(records)}", file=sys.stderr)

        print(f"Wrote predictions to {out_path}")
        return

    parser.print_help()


if __name__ == "__main__":
    main()
