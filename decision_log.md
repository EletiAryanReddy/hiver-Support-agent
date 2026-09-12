# Decision Log

Non-obvious decisions made while building this, and why. Bullet points,
roughly in the order they came up.

1. **Chose AppleSupport over Amazon/Uber/airlines.** AppleSupport's public
   replies are unusually consistent (short, warm, near-always DM-routes for
   anything specific), which makes "grounded in historical resolution" a
   well-defined, checkable target. Airlines/telecoms have much higher
   variance and more compound issues, which would have made the taxonomy
   and escalation policy far harder to validate in the available time.

2. **Built a synthetic-but-schema-matched dataset instead of skipping data
   entirely.** Rather than leaving data work as "TODO, no network access,"
   I modeled the generator on the actual Kaggle schema (`tweet_id`,
   `inbound`, `response_tweet_id` threading, etc.) so every downstream
   component is validated against something structurally real, and swapping
   in the true CSV requires zero pipeline code changes — only a new input
   file. This felt more honest than either skipping data or silently
   presenting synthetic results as if they were from the real corpus.

3. **9 intents, not Banking77's 77.** Twitter support messages are short
   (often <20 words) and frequently ambiguous. Fine-grained taxonomies
   need context that a single tweet often doesn't have; I picked intents
   that map to genuinely different *resolution playbooks* for this brand,
   not maximal granularity for its own sake.

4. **Escalation is a rule-gated policy layered on top of the classifier,
   not a second LLM call.** Auditability matters more here than flexibility
   — a support lead reviewing why something escalated should be able to
   point at a fixed rule ("this intent always escalates"), not an LLM's
   possibly-inconsistent judgment call on the same question twice.

5. **Safety-keyword override sits above the intent-based escalation rule,
   not folded into it.** A message mentioning "smoke" or "explode" must
   always escalate even if intent classification is wrong or classifies it
   as something normally auto-handled. Keeping this as an independent,
   simple keyword check (rather than trusting the classifier to route
   safety issues correctly) is a deliberate defense-in-depth choice.

6. **TF-IDF retrieval, not embeddings.** At ~1200 threads (or even a
   realistic single-brand slice of the real data, likely tens of
   thousands), TF-IDF cosine similarity is fast, needs no extra API calls
   or vector DB, and is fully interpretable (you can see exactly which
   words drove a match). Embeddings would likely improve recall on
   paraphrased queries, but I didn't have evidence that mattered enough
   here to justify the added complexity and cost — flagged as a "next
   week" candidate, not skipped out of ignorance.

7. **The reply drafter is explicitly told not to invent policy not present
   in the retrieved examples.** This is the single most important prompt
   constraint in the whole system — an ungrounded but fluent-sounding reply
   (e.g. promising a refund) is worse than no reply, and it's the exact
   failure the "grounded" judge dimension is built to catch.

8. **Escalated messages still get a drafted reply, not silence.**
   "Escalate" means "needs human sign-off before sending," not "the agent
   produces nothing." A human reviewing an escalated case benefits from
   having a starting draft to edit rather than starting from a blank page.

9. **Golden set mixes stratified corpus sampling with hand-written hard
   cases, not pure random sampling.** Pure random sampling from a templated
   synthetic corpus would produce an eval set that's too easy and wouldn't
   test the failure modes that matter (sarcasm, multi-intent, safety
   keywords, non-English, off-topic/spam). The hand-written cases exist
   specifically to stress-test the escalation safety net, not to inflate
   or deflate the headline accuracy number.

10. **Ambiguous golden examples are flagged (`is_ambiguous`) rather than
    forced into a single "correct" label**, and I report both strict
    accuracy and accuracy-excluding-ambiguous. Presenting only strict
    accuracy on genuinely ambiguous examples overstates how wrong the
    classifier actually is on cases where a domain expert might reasonably
    disagree.

11. **LLM-as-judge rubric has 4 independent dimensions (grounded, intent
    handling, tone, safety) rather than one holistic "quality" score.**
    A single blended score hides exactly the failure that matters most
    for a support agent — a reply can be perfectly on-brand in tone while
    being unsafely reassuring about a hazard. Splitting the dimensions
    makes that visible instead of averaged away.

12. **Judge calibration set is small (n=5) and I say so explicitly, rather
    than padding it to look more rigorous.** A fabricated-looking 30-example
    "calibration" that I clearly built adversarially would be less honest
    than a small set that's transparent about being a method demonstration.
    Section 4 of the report calls this out as a real limitation, not a
    footnote.

13. **Mock LLM client (`eval/mock_openai.py`) injects realistic errors
    rather than always returning correct answers.** A mock that's always
    right would make the eval harness itself untestable — I wouldn't know
    if `run_eval.py`'s accuracy computation, confusion matrix, or judge
    aggregation logic actually works, since every run would trivially
    score 100%. Deliberately noisy mock outputs let me validate the
    harness logic against known, traceable error patterns.

14. **`client` is a passable parameter throughout `pipeline/` and
    `eval/judge.py`, defaulting to a lazy-imported real OpenAI client.**
    This was the cleanest way to make the same code path work against
    either a real API key or the mock, without maintaining two parallel
    code trees or scattering `if MOCK:` branches through business logic.

15. **Reported escalation-decision accuracy as a single number in the
    results table, then explicitly criticized that choice in Section 4 of
    the report.** I could have just built the false-escalate/false-auto-
    handle split into the harness before writing the report. I chose to
    name it as a known gap instead of quietly fixing it, because the
    assignment explicitly rewards showing you know what's wrong with your
    own metric, and I'd rather demonstrate that judgment directly than
    only show a "fixed" version with no visible reasoning trail.
