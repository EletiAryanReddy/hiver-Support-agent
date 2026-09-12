"""
Mock OpenAI client used ONLY to validate the pipeline runs end-to-end and
the eval harness computes metrics correctly, in this sandboxed environment
where no network egress is available to call the real OpenAI API.

This is NOT a substitute for real results -- see README.md and
report/REPORT.md, which are explicit that headline numbers reported there
are either (a) from this mock validation run, clearly labeled as such, or
(b) the deterministic baseline/agreement numbers that need no LLM at all.

The mock simulates a REALISTIC classifier: mostly correct, with keyword-
based guessing (similar signal to the simple baseline, plus some
intentional noise) so the eval harness sees a non-trivial, non-perfect
score distribution -- a mock that always returns the right answer would
make the eval harness itself untestable.
"""

import json
import random

random.seed(123)

KEYWORD_HINTS = {
    "battery_performance": ["battery", "drain", "overheat", "hot", "lose", "slow"],
    "software_update_problem": ["update", "ios", "crash", "wifi", "touch id", "bug"],
    "device_wont_boot": ["won't turn on", "black screen", "stuck on", "unresponsive", "won't charge", "exploded", "burn", "smoke"],
    "warranty_repair": ["applecare", "cracked", "repair", "warranty", "defect", "coverage", "dead pixel"],
    "order_shipping_status": ["order", "shipping", "tracking", "delivery", "ship"],
    "account_access": ["apple id", "icloud", "locked out", "password", "2fa", "sign in", "login", "two factor"],
    "billing_refund": ["refund", "charged", "subscription", "billed", "billing"],
    "how_to_question": ["how do i", "how to", "way to", "explain"],
    "positive_feedback": ["thank", "shoutout", "impressed", "amazing", "appreciate", "thx"],
}

INTENTS = list(KEYWORD_HINTS.keys())


class MockChoice:
    def __init__(self, content):
        self.message = MockMessage(content)


class MockMessage:
    def __init__(self, content):
        self.content = content


class MockResponse:
    def __init__(self, content):
        self.choices = [MockChoice(content)]


class MockCompletions:
    def create(self, model, messages, temperature=0, max_tokens=None, response_format=None):
        system = messages[0]["content"]
        user = messages[1]["content"]

        if "intent classifier" in system.lower():
            return self._mock_intent(user)
        elif "quality reviewer" in system.lower():
            return self._mock_judge(user)
        elif "drafting a reply" in system.lower():
            return self._mock_draft(user)
        else:
            return MockResponse(json.dumps({"error": "unrecognized system prompt in mock"}))

    def _classify_text(self, text):
        lowered = text.lower()
        scores = {intent: sum(1 for kw in kws if kw in lowered) for intent, kws in KEYWORD_HINTS.items()}
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = random.choice(INTENTS)
            conf = round(random.uniform(0.15, 0.45), 2)
        else:
            conf = round(random.uniform(0.68, 0.95), 2)
            # Inject occasional realistic misclassification / low confidence
            # even on non-zero-score cases (~12% of the time) so the eval
            # harness sees a realistic, imperfect score distribution.
            if random.random() < 0.12:
                other_intents = [i for i in INTENTS if i != best]
                best = random.choice(other_intents)
                conf = round(random.uniform(0.4, 0.64), 2)
        return best, conf

    def _mock_intent(self, user_text):
        best, conf = self._classify_text(user_text)
        result = {"intent": best, "confidence": conf, "rationale": "mock classification based on keyword signal"}
        return MockResponse(json.dumps(result))

    def _mock_draft(self, user_prompt):
        # FIDELITY NOTE: a real LLM drafts from the grounding examples it's
        # given in the prompt (see pipeline/draft.py build_user_prompt), not
        # from an independently re-run intent guess. This mock now parses
        # the grounding examples' historical AppleSupport replies out of the
        # prompt and reuses the top match's reply pattern, matching that
        # real behavior. An earlier version of this mock re-classified the
        # customer text from scratch with fresh randomness, which could
        # diverge from the already-computed intent label and produce
        # intent/reply mismatches purely as a mock artifact -- fixed here.
        import re
        grounding_replies = re.findall(r"AppleSupport replied: (.+)", user_prompt)
        if grounding_replies:
            return MockResponse(grounding_replies[0].strip())

        # Fallback (no grounding examples parsed): classify from the
        # customer message itself, deterministically (no noise injection,
        # since this path exists only as a safety net, not to simulate
        # classifier error -- error injection belongs to _mock_intent only).
        lines = user_prompt.split("NEW CUSTOMER MESSAGE TO REPLY TO:")
        customer_msg = lines[1].strip().split("\n")[0] if len(lines) > 1 else ""
        lowered = customer_msg.lower()
        scores = {intent: sum(1 for kw in kws if kw in lowered) for intent, kws in KEYWORD_HINTS.items()}
        best = max(scores, key=scores.get) if any(scores.values()) else "how_to_question"

        canned = {
            "battery_performance": "Sorry to hear that! Could you let us know your iOS version and whether this happens with specific apps? We're happy to help troubleshoot.",
            "software_update_problem": "That's not expected. Try a force restart first, and if it persists, DM us your device model and iOS version.",
            "device_wont_boot": "Please try charging for 30+ minutes with a different cable, then attempt a force restart. If it doesn't respond, DM us your serial number.",
            "warranty_repair": "We'd recommend booking a check-in at your nearest Apple Store or starting a repair request at getsupport.apple.com.",
            "order_shipping_status": "We can help check that. Could you DM us your order number so we can look into the latest status?",
            "account_access": "For account security, please visit iforgot.apple.com to reset your password, or DM us if you're still stuck.",
            "billing_refund": "Sorry about the charge. Please DM us your order/receipt number and we'll look into a refund.",
            "how_to_question": "Great question! Check Settings for the relevant option, or see support.apple.com for a full guide.",
            "positive_feedback": "That means a lot, thank you for letting us know! 😊",
        }
        return MockResponse(canned.get(best, "Thanks for reaching out, let us know a bit more so we can help!"))

    def _mock_judge(self, user_prompt):
        # Detect the deliberately-bad calibration replies to simulate judge
        # correctly flagging them (tests whether harness computes agreement right)
        bad_markers = ["I've reset your account access for you", "restarting your phone and see if that helps with the smell",
                       "will ship within 24 hours guaranteed"]
        is_bad = any(m in user_prompt for m in bad_markers)

        if is_bad:
            result = {
                "grounded": 0, "correct_intent_handling": random.choice([0, 1]),
                "tone": 1, "safety": 0, "total_score": 1,
                "acceptable_to_send_as_is": False,
                "rationale": "Reply makes an ungrounded promise not supported by historical resolution pattern.",
            }
        else:
            g = random.choice([1, 2, 2, 2])
            c = random.choice([1, 2, 2])
            t = random.choice([1, 2, 2])
            s = 2
            total = g + c + t + s
            result = {
                "grounded": g, "correct_intent_handling": c, "tone": t, "safety": s,
                "total_score": total,
                "acceptable_to_send_as_is": total >= 6,
                "rationale": "Reply follows the general pattern of historical resolutions with appropriate tone.",
            }
        return MockResponse(json.dumps(result))


class MockChatNamespace:
    def __init__(self):
        self.completions = MockCompletions()


class MockOpenAI:
    """Drop-in mock for `openai.OpenAI` client, exposing `.chat.completions.create(...)`."""
    def __init__(self, api_key=None):
        self.chat = MockChatNamespace()
