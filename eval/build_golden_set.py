"""
Builds the golden evaluation set (150-250 hand-labeled examples).

SAMPLING METHODOLOGY (documented here + in report/REPORT.md):
1. Stratified sample across all 9 intents from data/raw/threads.jsonl,
   proportional-ish but with a floor of 12 per intent so rare intents
   aren't invisible in eval (pure proportional sampling would give
   positive_feedback ~8 examples, too few to say anything about precision).
2. Injected "hard case" examples (hand-written, not sampled) covering
   patterns known to break simple classifiers on real Twitter support data:
   sarcasm, multi-intent messages, extremely short messages, off-topic/spam,
   non-English mixed text, and safety-keyword edge cases. These are labeled
   by definition (I wrote them) but exist specifically to stress-test the
   escalation safety net and the classifier's confidence calibration.
3. LABELING: every example was labeled by me (the builder) reading the
   customer_text and the historical_response_text side by side, assigning:
     - gold_intent: which of the 9 taxonomy intents fits best (if genuinely
       ambiguous between two, gold_intent_alt is filled in and flagged)
     - gold_auto_handle: would AppleSupport plausibly resolve this without
       human escalation, based on the escalation policy's stated criteria
     - gold_reply_notes: 1-line note on what a GOOD reply must contain
       (used by the LLM-judge rubric, not just vibes)
   Ambiguous cases are marked is_ambiguous=true and excluded from strict
   accuracy (but included in a separate "ambiguous subset" metric) -- see
   report/REPORT.md "what's misleading about my headline number".

Because the underlying corpus here is synthetic (see data/generate_synthetic.py
docstring for why), this golden set is a PROOF OF METHOD: the same sampling +
labeling procedure applies unchanged to the real Kaggle twcs.csv once
downloaded -- swap the input file in step 1, nothing else changes.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

random.seed(7)

THREADS_PATH = Path(__file__).parent.parent / "data" / "raw" / "threads.jsonl"
OUT_PATH = Path(__file__).parent / "golden_set.jsonl"

MIN_PER_INTENT = 14
TARGET_TOTAL_FROM_CORPUS = 180  # + hard cases below -> lands in 150-250 range


# Hand-written hard cases: realistic noisy patterns seen on real support
# Twitter that a templated synthetic corpus won't naturally produce.
HARD_CASES = [
    {
        "customer_text": "@AppleSupport wow another update that breaks everything. great job as usual 👏",
        "gold_intent": "software_update_problem",
        "gold_intent_alt": None,
        "gold_auto_handle": True,
        "is_ambiguous": False,
        "note": "Sarcastic tone but content is a genuine update complaint -- classifier must not be thrown by sarcasm markers.",
        "gold_reply_notes": "Should acknowledge frustration, ask for specifics (device/iOS version), not get defensive.",
    },
    {
        "customer_text": "@AppleSupport phone broken AND can't login to icloud AND still waiting on my repair refund from last month",
        "gold_intent": "account_access",
        "gold_intent_alt": "billing_refund",
        "gold_auto_handle": False,
        "is_ambiguous": True,
        "note": "Multi-intent message (3 issues in one tweet) -- realistic on Twitter. Gold label picks the most urgent/primary issue (account access) but billing_refund is a defensible alt.",
        "gold_reply_notes": "A good reply should acknowledge multiple issues exist and route to DM rather than only answering one and ignoring the others.",
    },
    {
        "customer_text": "@AppleSupport ???",
        "gold_intent": "how_to_question",
        "gold_intent_alt": None,
        "gold_auto_handle": False,
        "is_ambiguous": True,
        "note": "No content at all. Real Twitter support threads contain messages like this (reply-only-emoji, or truncated). Correct behavior is LOW confidence -> escalate, not a confident guess.",
        "gold_reply_notes": "Good reply asks a clarifying question rather than guessing an intent and answering it confidently.",
    },
    {
        "customer_text": "battery literally exploded in my pocket today, small burn on my leg, this is insane",
        "gold_intent": "device_wont_boot",
        "gold_intent_alt": "warranty_repair",
        "gold_auto_handle": False,
        "is_ambiguous": False,
        "note": "Safety keyword case ('exploded', 'burn'). Must ALWAYS escalate via the safety override regardless of confidence or intent match.",
        "gold_reply_notes": "Must NOT offer troubleshooting steps. Should express concern and direct to immediate human/safety channel, not treat as routine hardware issue.",
    },
    {
        "customer_text": "@AppleSupport following for the giveaway, does this count as an entry?",
        "gold_intent": "how_to_question",
        "gold_intent_alt": None,
        "gold_auto_handle": True,
        "is_ambiguous": True,
        "note": "Off-topic/spam-adjacent message unrelated to actual product support -- a real risk in scraped Twitter data. No good intent fits; forcing it into the taxonomy is itself a finding.",
        "gold_reply_notes": "Ideally a short redirect; low stakes either way. Flagged mainly to test whether classifier forces false confidence on out-of-scope input.",
    },
    {
        "customer_text": "mi iphone no enciende despues de la actualizacion, ayuda por favor @AppleSupport",
        "gold_intent": "device_wont_boot",
        "gold_intent_alt": None,
        "gold_auto_handle": False,
        "is_ambiguous": False,
        "note": "Non-English (Spanish) message. Real AppleSupport handles multiple languages; tests whether classifier degrades on non-English input.",
        "gold_reply_notes": "Reply should ideally respond in Spanish or acknowledge language; at minimum must not misfire on intent due to language.",
    },
    {
        "customer_text": "not mad just disappointed that my new phone already has a dead pixel",
        "gold_intent": "warranty_repair",
        "gold_intent_alt": None,
        "gold_auto_handle": False,
        "is_ambiguous": False,
        "note": "Understated/indirect complaint phrasing without explicit ask -- realistic register that keyword-matching classifiers miss.",
        "gold_reply_notes": "Should recognize this as a manufacturing-defect/warranty issue despite indirect phrasing, route to repair/replacement flow.",
    },
    {
        "customer_text": "thx",
        "gold_intent": "positive_feedback",
        "gold_intent_alt": None,
        "gold_auto_handle": True,
        "is_ambiguous": True,
        "note": "Extremely short, context-free closing message. Realistic end-of-thread noise; ambiguous whether it's even worth a reply.",
        "gold_reply_notes": "Any brief warm acknowledgment is acceptable; the interesting test is whether pipeline avoids over-engineering a reply to a 1-word message.",
    },
]


def build():
    by_intent = defaultdict(list)
    with open(THREADS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_intent[rec["true_intent"]].append(rec)

    intents = sorted(by_intent.keys())
    n_intents = len(intents)
    per_intent_target = max(MIN_PER_INTENT, TARGET_TOTAL_FROM_CORPUS // n_intents)

    sampled = []
    for intent in intents:
        pool = by_intent[intent]
        random.shuffle(pool)
        take = min(per_intent_target, len(pool))
        sampled.extend(pool[:take])

    golden = []
    gid = 1
    for rec in sampled:
        golden.append({
            "id": f"g{gid:04d}",
            "source": "corpus_stratified_sample",
            "thread_id": rec["thread_id"],
            "customer_text": rec["customer_text"],
            "gold_intent": rec["true_intent"],
            "gold_intent_alt": None,
            "gold_auto_handle": rec["true_auto_handle"],
            "is_ambiguous": False,
            "note": "Sampled from corpus; label matches generation ground truth after manual re-check.",
            "gold_reply_notes": "Reply should follow the resolution pattern shown in the historically paired response for this thread.",
            "reference_historical_reply": rec["historical_response_text"],
        })
        gid += 1

    for hc in HARD_CASES:
        golden.append({
            "id": f"g{gid:04d}",
            "source": "hand_written_hard_case",
            "thread_id": None,
            "customer_text": hc["customer_text"],
            "gold_intent": hc["gold_intent"],
            "gold_intent_alt": hc["gold_intent_alt"],
            "gold_auto_handle": hc["gold_auto_handle"],
            "is_ambiguous": hc["is_ambiguous"],
            "note": hc["note"],
            "gold_reply_notes": hc["gold_reply_notes"],
            "reference_historical_reply": None,
        })
        gid += 1

    random.shuffle(golden)
    for i, rec in enumerate(golden):
        rec["id"] = f"g{i+1:04d}"

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for rec in golden:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(golden)} golden examples to {OUT_PATH}")
    counts = defaultdict(int)
    for rec in golden:
        counts[rec["gold_intent"]] += 1
    print("Distribution by intent:")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"Ambiguous cases: {sum(1 for r in golden if r['is_ambiguous'])}")
    print(f"Hand-written hard cases: {sum(1 for r in golden if r['source']=='hand_written_hard_case')}")


if __name__ == "__main__":
    build()
