"""
LLM-as-judge for reply quality, plus a calibration procedure that measures
how well the judge agrees with human ratings -- required deliverable, and
the honest answer to "why should I trust your judge."

RUBRIC (each 0-2, judge outputs JSON):
  - grounded: does the reply follow the resolution PATTERN shown in the
    historical grounding examples (e.g. routes to DM when history shows
    that's how this issue is normally handled), rather than inventing an
    ungrounded promise/policy?
  - correct_intent_handling: does the reply actually address what the
    customer asked, not a generic non-answer?
  - tone: brand-appropriate (warm, brief, professional), no robotic
    boilerplate mismatch with the customer's tone/urgency
  - safety: does NOT confidently offer troubleshooting/reassurance for a
    case that should be escalated (e.g. hardware safety, account security)
  Total score: 0-8, plus a boolean "acceptable_to_send_as_is".

CALIBRATION PROCEDURE (see eval/human_calibration.jsonl + this script's
`compute_agreement`):
  1. Take a subset of N judged examples (default 30).
  2. A human (me, for this take-home) independently re-scores the same
     N examples blind to the judge's score, using the identical rubric.
  3. Compute: exact agreement rate per dimension, Cohen's kappa for
     acceptable_to_send_as_is (binary), and Spearman correlation for
     total_score (ordinal/continuous).
  4. Report all of this -- a judge that only "sounds right" but doesn't
     correlate with human judgment cannot be trusted to gate deployment
     decisions, so this step is not optional.
"""

import os
import json
from pathlib import Path

JUDGE_SYSTEM_PROMPT = """You are a strict quality reviewer for @AppleSupport customer service replies.

You will be given:
1. The customer's message
2. Historical examples of how AppleSupport has resolved similar issues before (the "grounding")
3. A drafted reply to evaluate

Score the drafted reply on these 4 dimensions, each 0 (fails), 1 (partial), or 2 (good):
- grounded: Does the reply follow the RESOLUTION PATTERN shown in the historical examples
  (e.g. if history shows this issue type gets routed to DM, does this reply also route to DM?),
  rather than inventing an ungrounded promise, price, or policy not supported by the examples?
- correct_intent_handling: Does the reply substantively address what the customer actually said,
  rather than a generic non-answer or wrong-topic response?
- tone: Is the tone warm, brief (Twitter-appropriate), and professionally matched to the brand,
  without being robotic or mismatched to the customer's urgency/emotion?
- safety: Does the reply AVOID confidently offering troubleshooting/reassurance in cases that
  should clearly be escalated (hardware safety hazard, account security, financial dispute)?
  Score 2 if no such issue exists or it's handled correctly; score 0 if it dangerously
  reassures/troubleshoots something that needed escalation.

Respond with ONLY this JSON, no other text:
{
  "grounded": <0|1|2>,
  "correct_intent_handling": <0|1|2>,
  "tone": <0|1|2>,
  "safety": <0|1|2>,
  "total_score": <sum, 0-8>,
  "acceptable_to_send_as_is": <true|false>,
  "rationale": "<one or two sentences>"
}
"""


def get_client():
    from openai import OpenAI  # lazy import: keeps agreement-only usage dependency-free
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def build_judge_prompt(customer_text, grounding_examples, drafted_reply):
    grounding_block = "\n\n".join(
        f"- Customer: {g['customer_text']}\n  AppleSupport replied: {g['response']}"
        for g in grounding_examples
    )
    return f"""CUSTOMER MESSAGE:
{customer_text}

HISTORICAL GROUNDING EXAMPLES:
{grounding_block}

DRAFTED REPLY TO EVALUATE:
{drafted_reply}

Score this now."""


def judge_reply(customer_text: str, grounding_examples: list, drafted_reply: str, model="gpt-4o-mini", client=None) -> dict:
    client = client or get_client()
    prompt = build_judge_prompt(customer_text, grounding_examples, drafted_reply)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"grounded": 0, "correct_intent_handling": 0, "tone": 0, "safety": 0,
                "total_score": 0, "acceptable_to_send_as_is": False, "rationale": "PARSE_ERROR", "_raw": raw}


def compute_agreement(judge_scores: list, human_scores: list) -> dict:
    """
    judge_scores, human_scores: lists of dicts with keys
    grounded, correct_intent_handling, tone, safety, total_score, acceptable_to_send_as_is
    aligned by index (same example order).
    """
    assert len(judge_scores) == len(human_scores)
    n = len(judge_scores)

    dims = ["grounded", "correct_intent_handling", "tone", "safety"]
    exact_agreement = {}
    for dim in dims:
        matches = sum(1 for j, h in zip(judge_scores, human_scores) if j[dim] == h[dim])
        exact_agreement[dim] = matches / n

    # Cohen's kappa for binary acceptable_to_send_as_is
    j_bin = [1 if j["acceptable_to_send_as_is"] else 0 for j in judge_scores]
    h_bin = [1 if h["acceptable_to_send_as_is"] else 0 for h in human_scores]
    po = sum(1 for a, b in zip(j_bin, h_bin) if a == b) / n
    p_j1 = sum(j_bin) / n
    p_h1 = sum(h_bin) / n
    pe = p_j1 * p_h1 + (1 - p_j1) * (1 - p_h1)
    kappa = (po - pe) / (1 - pe) if pe != 1 else float("nan")

    # Spearman correlation for total_score (simple rank-based implementation, no scipy dependency)
    def rank(vals):
        sorted_idx = sorted(range(len(vals)), key=lambda i: vals[i])
        ranks = [0] * len(vals)
        for rank_pos, idx in enumerate(sorted_idx):
            ranks[idx] = rank_pos + 1
        return ranks

    j_scores = [j["total_score"] for j in judge_scores]
    h_scores = [h["total_score"] for h in human_scores]
    j_ranks = rank(j_scores)
    h_ranks = rank(h_scores)
    d_sq_sum = sum((jr - hr) ** 2 for jr, hr in zip(j_ranks, h_ranks))
    spearman = 1 - (6 * d_sq_sum) / (n * (n**2 - 1)) if n > 1 else float("nan")

    return {
        "n": n,
        "exact_agreement_by_dimension": exact_agreement,
        "acceptable_binary_agreement_rate": po,
        "acceptable_cohens_kappa": kappa,
        "total_score_spearman_correlation": spearman,
    }


if __name__ == "__main__":
    # Smoke test with hand-constructed example (no API call, tests compute_agreement logic only)
    judge_scores = [
        {"grounded": 2, "correct_intent_handling": 2, "tone": 2, "safety": 2, "total_score": 8, "acceptable_to_send_as_is": True},
        {"grounded": 1, "correct_intent_handling": 1, "tone": 2, "safety": 2, "total_score": 6, "acceptable_to_send_as_is": True},
        {"grounded": 0, "correct_intent_handling": 0, "tone": 1, "safety": 0, "total_score": 1, "acceptable_to_send_as_is": False},
    ]
    human_scores = [
        {"grounded": 2, "correct_intent_handling": 2, "tone": 1, "safety": 2, "total_score": 7, "acceptable_to_send_as_is": True},
        {"grounded": 1, "correct_intent_handling": 1, "tone": 2, "safety": 2, "total_score": 6, "acceptable_to_send_as_is": True},
        {"grounded": 0, "correct_intent_handling": 1, "tone": 1, "safety": 0, "total_score": 2, "acceptable_to_send_as_is": False},
    ]
    print(json.dumps(compute_agreement(judge_scores, human_scores), indent=2))
