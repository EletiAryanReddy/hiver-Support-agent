"""
Draft a reply grounded in historically similar resolved threads.
The model is instructed to base its reply on the retrieved examples'
STYLE and RESOLUTION PATTERN, not to invent new policy (e.g. it should
never independently promise a refund amount or repair cost that wasn't
in a retrieved example).
"""

import os
import json
from pipeline.retrieval import HistoricalMatch

SYSTEM_PROMPT = """You are drafting a reply as @AppleSupport on Twitter/X.

Rules:
- Match the brand's real tone: warm, brief (under 280 characters), never
  discloses account/order specifics in the open reply -- route those to DM.
- Ground your reply in the RESOLUTION PATTERN shown in the historical
  examples below. Do not invent policy, prices, or specific promises that
  aren't supported by the historical pattern.
- If the historical examples show this category of issue is normally
  routed to DM, do the same.
- Output ONLY the reply text, nothing else. No quotes, no explanation.
"""


def build_user_prompt(customer_text: str, matches: list[HistoricalMatch]) -> str:
    examples_block = "\n\n".join(
        f"Historical similar case (similarity={m.similarity:.2f}):\n"
        f"Customer: {m.customer_text}\n"
        f"AppleSupport replied: {m.historical_response_text}"
        for m in matches
    )
    return f"""HISTORICAL GROUNDING EXAMPLES:
{examples_block}

NEW CUSTOMER MESSAGE TO REPLY TO:
{customer_text}

Draft the reply now."""


def get_client():
    from openai import OpenAI
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


def draft_reply(customer_text: str, matches: list[HistoricalMatch], model: str = "gpt-4o-mini", client=None) -> str:
    client = client or get_client()
    user_prompt = build_user_prompt(customer_text, matches)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=150,
    )
    return resp.choices[0].message.content.strip()


if __name__ == "__main__":
    from pipeline.retrieval import HistoricalIndex

    idx = HistoricalIndex()
    query = "my iphone battery is draining super fast after the update"
    matches = idx.search(query, k=3)
    reply = draft_reply(query, matches)
    print(f"Customer: {query}\n")
    print(f"Draft reply: {reply}")
