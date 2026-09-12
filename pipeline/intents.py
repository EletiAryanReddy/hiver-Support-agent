"""
Intent taxonomy for @AppleSupport, derived from clustering the recurring
issue patterns actually present in the (synthetic, Kaggle-schema-matched)
corpus. See report/REPORT.md 'Problem framing' for why this granularity
(9 intents) was chosen over a Banking77-style fine-grained taxonomy.
"""

import os
import json

INTENTS = [
    "battery_performance",
    "software_update_problem",
    "device_wont_boot",
    "warranty_repair",
    "order_shipping_status",
    "account_access",
    "billing_refund",
    "how_to_question",
    "positive_feedback",
]

INTENT_DESCRIPTIONS = {
    "battery_performance": "Complaints about battery drain, overheating, or general slowness/performance degradation.",
    "software_update_problem": "Issues that started after or during an iOS/software update: crashes, bugs, connectivity problems, failed updates.",
    "device_wont_boot": "Device is unresponsive, won't power on, stuck on logo, black screen, or physically damaged (water/drop) and non-functional.",
    "warranty_repair": "Questions about repair cost, AppleCare coverage, physical damage assessment, or booking a repair appointment.",
    "order_shipping_status": "Questions about a pending purchase: shipping status, delivery delay, wrong/missing delivery, order changes.",
    "account_access": "Apple ID / iCloud login issues, 2FA problems, password resets, suspected unauthorized account access.",
    "billing_refund": "Disputed charges, refund requests, subscription billing issues, unauthorized purchases.",
    "how_to_question": "General how-to / feature usage questions with no error or complaint -- customer wants to know how to do something.",
    "positive_feedback": "Compliments or thanks directed at the support team, not a request needing resolution.",
}

SYSTEM_PROMPT = f"""You are an intent classifier for @AppleSupport customer service tweets.
Classify the customer's message into EXACTLY ONE of these intents:

{json.dumps(INTENT_DESCRIPTIONS, indent=2)}

Respond with ONLY a JSON object, no other text, no markdown fences:
{{"intent": "<one of the intent keys above>", "confidence": <float 0-1>, "rationale": "<one short sentence>"}}

Confidence should reflect genuine uncertainty -- use lower values (below 0.6) when the
message is ambiguous, ultra-short, sarcastic, or plausibly fits 2+ intents.
"""


def get_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def classify_intent(customer_text: str, model: str = "gpt-4o-mini", client=None) -> dict:
    client = client or get_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": customer_text},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"intent": "how_to_question", "confidence": 0.0, "rationale": "PARSE_ERROR", "_raw": raw}

    if parsed.get("intent") not in INTENTS:
        parsed["_invalid_intent_returned"] = parsed.get("intent")
        parsed["intent"] = "how_to_question"
        parsed["confidence"] = 0.0
    return parsed


if __name__ == "__main__":
    examples = [
        "my iphone 13 battery drains so fast after the update, it's unusable",
        "screen cracked, is it covered under applecare?",
        "shoutout to @AppleSupport for actually fixing my issue in 10 minutes",
    ]
    for ex in examples:
        result = classify_intent(ex)
        print(f"{ex}\n  -> {result}\n")
