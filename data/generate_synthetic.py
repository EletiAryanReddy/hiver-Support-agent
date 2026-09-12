"""
Generates a synthetic corpus that mirrors the schema and conversational
structure of the Kaggle "Customer Support on Twitter" dataset
(thoughtvector/customer-support-on-twitter), scoped to @AppleSupport.

WHY SYNTHETIC: the real dataset (~3M rows, ~1.6GB) requires a Kaggle account
and could not be downloaded in this environment (no network egress). This
generator reproduces the exact schema and realistic AppleSupport phrasing
patterns so the pipeline, taxonomy, and eval harness are validated end-to-end.

TO SWITCH TO THE REAL DATASET:
  1. Download twcs.csv from Kaggle (thoughtvector/customer-support-on-twitter)
  2. Place it at data/raw/twcs.csv
  3. Run: python data/filter_brand.py --brand AppleSupport
  4. Everything downstream (pipeline/, eval/) reads from
     data/raw/threads.jsonl regardless of source -- no other code changes.

Schema (matches Kaggle twcs.csv columns):
  tweet_id, author_id, inbound, created_at, text,
  response_tweet_id, in_response_to_tweet_id
"""

import json
import random
import csv
from datetime import datetime, timedelta
from pathlib import Path

random.seed(42)

OUT_DIR = Path(__file__).parent / "raw"
OUT_DIR.mkdir(exist_ok=True)

BRAND = "AppleSupport"

# ---------------------------------------------------------------------------
# Scenario bank: each scenario = one recurring real-world AppleSupport issue
# pattern, with realistic customer phrasing variants and the actual style of
# resolution AppleSupport gives (short, routes to DM, asks for device info,
# points to a support article, or asks to try a canned fix).
# These patterns are modeled on publicly known AppleSupport conventions
# (DM handoff for anything account-specific, standard troubleshooting asks,
# never discusses order/billing specifics in the open thread).
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "intent": "battery_performance",
        "customer_templates": [
            "my iphone {model} battery drains so fast after the update, it's unusable",
            "why does my {model} lose 20% battery in an hour just sitting there",
            "battery health is at {pct}% and phone dies by 2pm, is this normal",
            "@AppleSupport ever since ios {ios_ver} my battery has been terrible",
            "is it normal for my {model} to get hot and lose battery this fast",
        ],
        "resolution_templates": [
            "That doesn't sound right. Let's get this sorted -- which iOS version are you on, and does this happen with specific apps or all the time?",
            "We'd like to check a few things with you. Can you check Settings > Battery > Battery Health & Charging and let us know the max capacity shown?",
            "Sorry to hear this! A few background app or settings checks can help here. Mind moving to DM so we can look into your device specifics?",
        ],
        "auto_handle": True,
    },
    {
        "intent": "software_update_problem",
        "customer_templates": [
            "ios {ios_ver} update bricked my {model}, stuck on apple logo",
            "update to ios {ios_ver} and now my apps keep crashing constantly",
            "my phone won't update past ios {ios_ver}, keeps failing",
            "@AppleSupport since the {ios_ver} update wifi keeps dropping every few minutes",
            "downloaded the new update and now touch id doesn't work at all",
        ],
        "resolution_templates": [
            "Let's help get this resolved. Have you tried a force restart? Press and quickly release Volume Up, then Volume Down, then hold the Side button.",
            "That's not expected after an update. Can you tell us the exact iOS version and model? We'll go from there.",
            "Sorry about that. First, try restarting your device. If the issue persists, we can look deeper -- mind sending us a DM?",
        ],
        "auto_handle": True,
    },
    {
        "intent": "device_wont_boot",
        "customer_templates": [
            "my {model} won't turn on at all, screen is completely black",
            "phone fell in water and now it won't power on, please help",
            "stuck on apple logo for over an hour now, tried restarting",
            "@AppleSupport my ipad won't charge or turn on, tried different cables",
            "screen is unresponsive and phone is warm, won't do anything",
        ],
        "resolution_templates": [
            "We know how important your device is, let's work through this. Have you tried a force restart and letting it charge for at least 30 minutes first?",
            "Sorry to hear this. If a force restart doesn't help, this may need a closer look. Can you DM us your device serial number?",
            "Please try charging with a different cable/adapter for 15+ minutes, then attempt a force restart. Let us know what happens.",
        ],
        "auto_handle": False,  # hardware failure risk -> escalate for safety/warranty
    },
    {
        "intent": "warranty_repair",
        "customer_templates": [
            "screen cracked, is it covered under applecare?",
            "how much would it cost to fix a cracked back glass on {model}",
            "my phone has a manufacturing defect, camera bump is uneven, can i get a replacement",
            "@AppleSupport battery swelled up and is pushing the screen out, is this covered?",
            "need to book a repair appointment for water damage",
        ],
        "resolution_templates": [
            "We'd recommend booking a check-in at your nearest Apple Store or starting a repair request online so we can assess coverage.",
            "This sounds like it needs a closer look by a technician. You can check your coverage and start a service request at getsupport.apple.com.",
            "A swollen battery should be looked at right away for safety. Please stop using the device and visit an Apple Store or Authorized Service Provider as soon as possible.",
        ],
        "auto_handle": False,  # coverage/cost decisions require human judgment
    },
    {
        "intent": "order_shipping_status",
        "customer_templates": [
            "ordered my {model} 2 weeks ago and it still says preparing to ship",
            "tracking hasn't updated in 5 days, where is my order",
            "@AppleSupport my order says delivered but i never got it",
            "can i change the shipping address on my pending order",
            "estimated delivery keeps getting pushed back, what's going on",
        ],
        "resolution_templates": [
            "We can help look into that. Could you send us your order number via DM so we can check the latest status?",
            "Sorry for the delay! Please DM us your order number and we'll check with the shipping carrier on our end.",
            "That's not the experience we want. Let's take a closer look -- can you send your order confirmation number in a DM?",
        ],
        "auto_handle": False,  # requires account/order-specific lookup, PII
    },
    {
        "intent": "account_access",
        "customer_templates": [
            "locked out of my apple id, verification code never arrives",
            "forgot my apple id password and the reset email isn't coming",
            "@AppleSupport two factor authentication won't send code to my trusted device",
            "someone else is using my apple id, how do i secure it",
            "can't sign into icloud on my new phone, says account disabled",
        ],
        "resolution_templates": [
            "Let's get your account secured. Please visit iforgot.apple.com to reset your password, or DM us if you're still stuck.",
            "Sorry for the trouble. For account security we can't handle Apple ID specifics here -- please DM us so we can verify and assist further.",
            "If you believe your account was accessed by someone else, please change your password immediately at iforgot.apple.com and enable two-factor authentication.",
        ],
        "auto_handle": False,  # account security -> always escalate/DM
    },
    {
        "intent": "billing_refund",
        "customer_templates": [
            "charged twice for the same app purchase, need a refund",
            "cancelled my subscription but still got charged this month",
            "@AppleSupport my kid made in app purchases without permission, can i get refunded",
            "billed for icloud storage i already cancelled",
            "requested a refund 2 weeks ago and haven't heard back",
        ],
        "resolution_templates": [
            "We understand, let's help resolve this. You can request a refund at reportaproblem.apple.com, or DM us your order ID and we'll check.",
            "Sorry about the unexpected charge. Please DM us the receipt/order number so we can look into a refund on our end.",
            "For unauthorized purchases, please report it directly at reportaproblem.apple.com. If you need further help, DM us the order details.",
        ],
        "auto_handle": False,  # money involved -> escalate
    },
    {
        "intent": "how_to_question",
        "customer_templates": [
            "how do i transfer photos from my old iphone to the new one",
            "@AppleSupport how do i turn off read receipts in imessage",
            "is there a way to see which apps are using the most battery",
            "how do i free up storage on my {model} without deleting photos",
            "can you explain how family sharing works for app purchases",
        ],
        "resolution_templates": [
            "Great question! You can use Quick Start or iCloud Backup to transfer everything over -- here's a guide: support.apple.com/HT204350",
            "You can toggle that off in Settings > Messages > Send Read Receipts. Let us know if you run into any issues!",
            "Check Settings > General > iPhone Storage for a breakdown by app, plus recommendations to free up space.",
        ],
        "auto_handle": True,
    },
    {
        "intent": "positive_feedback",
        "customer_templates": [
            "shoutout to @AppleSupport for actually fixing my issue in 10 minutes",
            "genuinely impressed with how fast apple support resolved my battery issue",
            "@AppleSupport thank you so much for the help yesterday, phone works great now",
            "customer service today was amazing, appreciate you guys",
        ],
        "resolution_templates": [
            "That means a lot, thank you for letting us know! 😊",
            "We're so glad we could help! Thanks for the kind words.",
            "Awesome to hear! Don't hesitate to reach out if anything else comes up.",
        ],
        "auto_handle": True,
    },
]

MODELS = ["11", "12", "12 Pro", "13", "13 mini", "14", "SE", "XR", "XS"]
IOS_VERSIONS = ["16.1", "16.2", "16.4", "17.0", "17.1", "15.7"]


def render(template: str) -> str:
    return template.format(
        model=f"iPhone {random.choice(MODELS)}",
        ios_ver=random.choice(IOS_VERSIONS),
        pct=random.randint(62, 84),
    )


def gen_thread(thread_id: int, scenario: dict, start_id: int):
    """Generate one inbound customer tweet + one AppleSupport response tweet,
    matching the Kaggle schema's threading fields."""
    cust_tweet_id = start_id
    resp_tweet_id = start_id + 1

    base_time = datetime(2023, 1, 1) + timedelta(
        days=random.randint(0, 400), hours=random.randint(0, 23), minutes=random.randint(0, 59)
    )

    customer_text = render(random.choice(scenario["customer_templates"]))
    if "@AppleSupport" not in customer_text:
        customer_text = f"@AppleSupport {customer_text}"

    response_text = render(random.choice(scenario["resolution_templates"]))

    cust_row = {
        "tweet_id": cust_tweet_id,
        "author_id": f"cust_{thread_id:05d}",
        "inbound": True,
        "created_at": base_time.strftime("%a %b %d %H:%M:%S +0000 %Y"),
        "text": customer_text,
        "response_tweet_id": str(resp_tweet_id),
        "in_response_to_tweet_id": "",
        # ground-truth label, only used for building golden set / eval,
        # NOT passed to the pipeline at inference time
        "_true_intent": scenario["intent"],
        "_true_auto_handle": scenario["auto_handle"],
    }
    resp_row = {
        "tweet_id": resp_tweet_id,
        "author_id": BRAND,
        "inbound": False,
        "created_at": (base_time + timedelta(minutes=random.randint(3, 90))).strftime(
            "%a %b %d %H:%M:%S +0000 %Y"
        ),
        "text": response_text,
        "response_tweet_id": "",
        "in_response_to_tweet_id": str(cust_tweet_id),
        "_true_intent": None,
        "_true_auto_handle": None,
    }
    return cust_row, resp_row


def main(n_threads: int = 1200):
    rows = []
    tid = 1000000
    for i in range(n_threads):
        scenario = random.choice(SCENARIOS)
        c, r = gen_thread(i, scenario, tid)
        rows.append(c)
        rows.append(r)
        tid += 2

    # Write full raw CSV (mimics twcs.csv schema exactly, minus our debug cols)
    csv_path = OUT_DIR / "twcs_applesupport_synthetic.csv"
    fieldnames = [
        "tweet_id", "author_id", "inbound", "created_at", "text",
        "response_tweet_id", "in_response_to_tweet_id",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    # Write paired threads.jsonl (customer msg + resolved response + label)
    # This is what the pipeline and eval harness actually consume.
    threads_path = OUT_DIR / "threads.jsonl"
    with open(threads_path, "w", encoding="utf-8") as f:
        for i in range(0, len(rows), 2):
            c, r = rows[i], rows[i + 1]
            record = {
                "thread_id": i // 2,
                "customer_tweet_id": c["tweet_id"],
                "customer_text": c["text"],
                "response_tweet_id": r["tweet_id"],
                "historical_response_text": r["text"],
                "true_intent": c["_true_intent"],
                "true_auto_handle": c["_true_auto_handle"],
                "created_at": c["created_at"],
            }
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(rows)} tweets ({len(rows)//2} threads) to:")
    print(f"  {csv_path}")
    print(f"  {threads_path}")
    print(f"Scenarios used: {[s['intent'] for s in SCENARIOS]}")


if __name__ == "__main__":
    main()
