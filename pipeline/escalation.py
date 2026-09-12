"""
Escalation policy: decides auto-handle vs escalate-to-human, with a stated
reason. This is a RULE-GATED policy layered on top of the classifier, not a
second model call -- see decision_log.md for why (auditability, and the
categories that must always escalate should never depend on model whim).

Policy (in priority order):
  1. Risk-tier intents ALWAYS escalate, regardless of confidence:
     - account_access      (security: wrong move = account takeover exposure)
     - billing_refund       (money: wrong move = unauthorized refund/liability)
     - warranty_repair       (cost/coverage judgment call, needs human authority)
     - order_shipping_status (requires PII/order lookup agent doesn't have access to)
     - device_wont_boot      (possible hardware safety issue, e.g. battery swelling)
  2. Low classifier confidence (< CONFIDENCE_THRESHOLD) -> escalate
     (agent unsure what the customer even needs)
  3. Safety keyword override -> escalate regardless of intent
     (injury, fire, smoke, explode, etc. -- always human, always fast)
  4. Otherwise -> auto-handle
"""

CONFIDENCE_THRESHOLD = 0.65

ALWAYS_ESCALATE_INTENTS = {
    "account_access",
    "billing_refund",
    "warranty_repair",
    "order_shipping_status",
    "device_wont_boot",
}

SAFETY_KEYWORDS = [
    "fire", "smoke", "explode", "explosion", "burn", "burned", "burning",
    "injury", "injured", "hurt me", "caught fire", "sparks", "swollen battery",
    "swelling",
]


def check_safety_override(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in SAFETY_KEYWORDS)


def decide_escalation(customer_text: str, intent: str, confidence: float) -> dict:
    """Returns {"decision": "auto_handle"|"escalate", "reason": str}"""

    if check_safety_override(customer_text):
        return {
            "decision": "escalate",
            "reason": "Safety keyword detected (possible device safety hazard) -- always routed to a human regardless of intent or confidence.",
        }

    if intent in ALWAYS_ESCALATE_INTENTS:
        return {
            "decision": "escalate",
            "reason": f"Intent '{intent}' is in the always-escalate risk tier "
                       f"(requires account verification, financial authority, "
                       f"coverage judgment, PII/order lookup, or hardware safety assessment "
                       f"the agent cannot perform).",
        }

    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "decision": "escalate",
            "reason": f"Classifier confidence ({confidence:.2f}) below threshold "
                      f"({CONFIDENCE_THRESHOLD}) -- message is ambiguous enough that "
                      f"auto-handling risks a mismatched or unhelpful reply.",
        }

    return {
        "decision": "auto_handle",
        "reason": f"Intent '{intent}' is low-risk and classifier confidence "
                  f"({confidence:.2f}) is above threshold -- a grounded templated "
                  f"reply is safe to send without human review.",
    }


if __name__ == "__main__":
    tests = [
        ("my iphone battery drains fast", "battery_performance", 0.9),
        ("locked out of my apple id", "account_access", 0.95),
        ("battery is swelling and smells like smoke", "device_wont_boot", 0.9),
        ("not sure what's wrong tbh just weird", "how_to_question", 0.4),
    ]
    for text, intent, conf in tests:
        print(f"{text!r}\n  -> {decide_escalation(text, intent, conf)}\n")
