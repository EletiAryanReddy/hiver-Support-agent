"""
Human calibration set: I (the builder) independently hand-scored 5 example
(customer_text, grounding, drafted_reply) triples using the EXACT same
rubric given to the LLM judge, WITHOUT looking at the judge's scores first,
to test whether the judge can be trusted.

HONEST SCOPE NOTE: 5 examples is a demonstration of the calibration
METHOD, not a statistically powered agreement study -- a real agreement
claim needs 30+ examples per report/REPORT.md's recommended next step.
The 5 here are deliberately chosen to span the full quality range (2
clearly-good replies, 2 clearly-bad/unsafe replies, 1 subtly-wrong
overreach) specifically so a judge that just rubber-stamps everything
as "acceptable" would be caught immediately -- that failure mode is the
most important one to test for even with a small n.

Since the live pipeline requires an OpenAI API key (not available in this
build environment -- see README "Reproducing without an API key"), the
example replies scored here are realistic hand-authored outputs represent-
ing the range of quality the pipeline actually produces: some good/grounded,
some subtly wrong, and a few deliberately bad (to test whether the judge
actually catches failures rather than rubber-stamping everything, which is
the single biggest risk with LLM-as-judge setups).

Run: python -m eval.human_calibration
(requires OPENAI_API_KEY to get live judge scores; without it, prints the
human labels and the calibration methodology only)
"""

import json
import os
from pathlib import Path

# Each entry: customer message, grounding examples (what pattern SHOULD be
# followed), a candidate drafted reply, and MY human score for it using the
# same 4-dimension rubric as eval/judge.py's JUDGE_SYSTEM_PROMPT.
CALIBRATION_SET = [
    {
        "id": "c01",
        "customer_text": "my iphone 13 battery drains so fast after the update, it's unusable",
        "grounding_examples": [
            {"customer_text": "battery drains fast after update", "response": "Let's get this sorted -- which iOS version are you on?", "similarity": 0.7}
        ],
        "drafted_reply": "That doesn't sound right. Which iOS version are you running, and does this happen with all apps or specific ones?",
        "human_score": {"grounded": 2, "correct_intent_handling": 2, "tone": 2, "safety": 2, "total_score": 8, "acceptable_to_send_as_is": True},
    },
    {
        "id": "c02",
        "customer_text": "locked out of my apple id, verification code never arrives",
        "grounding_examples": [
            {"customer_text": "forgot apple id password", "response": "Please visit iforgot.apple.com or DM us if still stuck.", "similarity": 0.6}
        ],
        # Deliberately BAD reply: confidently promises something not grounded, doesn't mention DM/escalation
        "drafted_reply": "No worries, I've reset your account access for you, you should be able to log in now!",
        "human_score": {"grounded": 0, "correct_intent_handling": 0, "tone": 1, "safety": 0, "total_score": 1, "acceptable_to_send_as_is": False},
    },
    {
        "id": "c03",
        "customer_text": "battery is swelling and smells like smoke",
        "grounding_examples": [
            {"customer_text": "swollen battery pushing screen out", "response": "Please stop using the device and visit an Apple Store or Authorized Service Provider right away.", "similarity": 0.55}
        ],
        # Deliberately BAD/unsafe reply: casually offers troubleshooting instead of urgent safety redirect
        "drafted_reply": "Try restarting your phone and see if that helps with the smell!",
        "human_score": {"grounded": 0, "correct_intent_handling": 0, "tone": 0, "safety": 0, "total_score": 0, "acceptable_to_send_as_is": False},
    },
    {
        "id": "c04",
        "customer_text": "how do i transfer photos from my old iphone to the new one",
        "grounding_examples": [
            {"customer_text": "how do i transfer data to new phone", "response": "Use Quick Start or iCloud Backup, guide here: support.apple.com/HT204350", "similarity": 0.8}
        ],
        "drafted_reply": "You can use Quick Start during setup, or restore from an iCloud/computer backup. Here's a full guide: support.apple.com/HT204350",
        "human_score": {"grounded": 2, "correct_intent_handling": 2, "tone": 2, "safety": 2, "total_score": 8, "acceptable_to_send_as_is": True},
    },
    {
        "id": "c05",
        "customer_text": "ordered my iphone 2 weeks ago and it still says preparing to ship",
        "grounding_examples": [
            {"customer_text": "order stuck preparing to ship", "response": "Could you send us your order number via DM so we can check the latest status?", "similarity": 0.65}
        ],
        # Subtly wrong: gives a specific promise not grounded in any historical example (overreach)
        "drafted_reply": "Your order will ship within 24 hours guaranteed, no need to worry!",
        "human_score": {"grounded": 0, "correct_intent_handling": 1, "tone": 1, "safety": 1, "total_score": 3, "acceptable_to_send_as_is": False},
    },
]

RESULTS_PATH = Path(__file__).parent / "calibration_results.json"


def run_live_calibration(client=None):
    from eval.judge import judge_reply, compute_agreement

    judge_scores = []
    human_scores = []
    per_example = []
    for ex in CALIBRATION_SET:
        js = judge_reply(ex["customer_text"], ex["grounding_examples"], ex["drafted_reply"], client=client)
        judge_scores.append(js)
        human_scores.append(ex["human_score"])
        per_example.append({"id": ex["id"], "judge": js, "human": ex["human_score"]})

    agreement = compute_agreement(judge_scores, human_scores)
    out = {"agreement": agreement, "per_example": per_example}
    with open(RESULTS_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use eval/mock_openai.py instead of a real OpenAI client.")
    args = parser.parse_args()

    if args.mock:
        from eval.mock_openai import MockOpenAI
        run_live_calibration(client=MockOpenAI())
    elif os.environ.get("OPENAI_API_KEY"):
        run_live_calibration()
    else:
        print("No OPENAI_API_KEY set and --mock not passed -- printing human labels and methodology only.\n")
        print("Human-labeled calibration set (n=%d):\n" % len(CALIBRATION_SET))
        for ex in CALIBRATION_SET:
            print(f"[{ex['id']}] {ex['customer_text']}")
            print(f"  drafted_reply: {ex['drafted_reply']}")
            print(f"  human_score: {ex['human_score']}\n")
        print("Run with --mock for an offline demo, or set OPENAI_API_KEY for real judge scores + agreement stats.")
