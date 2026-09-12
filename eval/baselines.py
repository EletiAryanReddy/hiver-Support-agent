"""
Two baselines to compare the LLM pipeline against, per assignment requirement.

TRIVIAL BASELINE: always predict the majority intent class, and always
auto-handle (or always escalate -- we report both variants). This tells us
the "free lunch" score any system gets by doing nothing clever.

SIMPLE BASELINE: keyword/rule-based classifier using hand-picked trigger
words per intent (no ML, no LLM). This represents what a non-ML engineer
could ship in an afternoon, and is the bar the LLM pipeline actually needs
to clear to justify its cost/latency.
"""

import json
from pathlib import Path
from collections import Counter

GOLDEN_PATH = Path(__file__).parent / "golden_set.jsonl"

KEYWORD_RULES = {
    "battery_performance": ["battery", "drain", "overheat", "hot", "slow", "lag"],
    "software_update_problem": ["update", "ios", "crash", "bug", "wifi", "touch id", "since the"],
    "device_wont_boot": ["won't turn on", "won't power", "black screen", "stuck on", "unresponsive", "won't charge", "exploded", "burn"],
    "warranty_repair": ["applecare", "cracked", "repair", "warranty", "defect", "coverage", "dead pixel"],
    "order_shipping_status": ["order", "shipping", "tracking", "delivery", "delivered", "ship"],
    "account_access": ["apple id", "icloud", "locked out", "password", "two factor", "2fa", "sign in", "login"],
    "billing_refund": ["refund", "charged", "charge", "subscription", "billed", "billing"],
    "how_to_question": ["how do i", "how to", "is there a way", "can you explain"],
    "positive_feedback": ["thank", "thanks", "shoutout", "impressed", "amazing", "appreciate"],
}

ALWAYS_ESCALATE_INTENTS = {
    "account_access", "billing_refund", "warranty_repair",
    "order_shipping_status", "device_wont_boot",
}


def load_golden():
    records = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def trivial_baseline(records):
    """Always predict majority class; always auto_handle=False (safest trivial default)."""
    majority_intent = Counter(r["gold_intent"] for r in records).most_common(1)[0][0]
    predictions = []
    for r in records:
        predictions.append({
            "id": r["id"],
            "predicted_intent": majority_intent,
            "predicted_auto_handle": False,  # "always escalate" is the safe trivial default
        })
    return predictions, {"majority_intent_used": majority_intent}


def simple_keyword_baseline(records):
    """Rule-based keyword matcher; falls back to how_to_question if nothing matches."""
    predictions = []
    for r in records:
        text = r["customer_text"].lower()
        scores = {intent: sum(1 for kw in kws if kw in text) for intent, kws in KEYWORD_RULES.items()}
        best_intent = max(scores, key=scores.get)
        if scores[best_intent] == 0:
            best_intent = "how_to_question"  # fallback, matches classifier's own fallback behavior

        auto_handle = best_intent not in ALWAYS_ESCALATE_INTENTS
        predictions.append({
            "id": r["id"],
            "predicted_intent": best_intent,
            "predicted_auto_handle": auto_handle,
        })
    return predictions, {}


def score(records, predictions):
    id_to_gold = {r["id"]: r for r in records}
    total = len(predictions)
    correct_intent = 0
    correct_auto_handle = 0
    for p in predictions:
        gold = id_to_gold[p["id"]]
        if p["predicted_intent"] == gold["gold_intent"]:
            correct_intent += 1
        if p["predicted_auto_handle"] == gold["gold_auto_handle"]:
            correct_auto_handle += 1
    return {
        "n": total,
        "intent_accuracy": correct_intent / total,
        "auto_handle_accuracy": correct_auto_handle / total,
    }


if __name__ == "__main__":
    records = load_golden()

    trivial_preds, trivial_meta = trivial_baseline(records)
    trivial_scores = score(records, trivial_preds)
    print("=== TRIVIAL BASELINE (majority class, always escalate) ===")
    print(json.dumps({**trivial_scores, **trivial_meta}, indent=2))

    simple_preds, _ = simple_keyword_baseline(records)
    simple_scores = score(records, simple_preds)
    print("\n=== SIMPLE BASELINE (keyword rules) ===")
    print(json.dumps(simple_scores, indent=2))

    # Save for report comparison
    out = {
        "trivial": {**trivial_scores, **trivial_meta},
        "simple_keyword": simple_scores,
    }
    with open(Path(__file__).parent / "baseline_results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {Path(__file__).parent / 'baseline_results.json'}")
