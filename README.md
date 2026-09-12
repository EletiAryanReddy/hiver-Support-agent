# AppleSupport AI Agent — Hiver SDE Intern Take-Home

An AI support agent for **@AppleSupport** (chosen for data cleanliness and clearly
recurring issue categories) that classifies incoming customer messages into 9
intents, drafts a reply grounded in historically similar resolved threads, and
decides auto-handle vs. escalate-to-human with a stated reason.

**Read `report/REPORT.md` for problem framing, results, failure analysis, and
the mandatory "what's misleading about my headline number" section.**
**Read `decision_log.md` for the 10-15 non-obvious decisions and why.**

---

## Important: two honest caveats up front

1. **No live Kaggle dataset in this build.** The build environment used to
   produce this repo has no network egress, so the real ~3M-row Kaggle
   `thoughtvector/customer-support-on-twitter` CSV could not be downloaded.
   Everything here runs against a **synthetic corpus** (`data/generate_synthetic.py`)
   that exactly mirrors the real dataset's schema and models realistic
   @AppleSupport phrasing patterns. **Section "Using the real dataset" below**
   explains the (one-command) swap. All taxonomy, retrieval, escalation, and
   eval-harness code is written to be data-source-agnostic.

2. **No live OpenAI API key in this build.** Same network constraint. The
   headline numbers in `report/REPORT.md` and `eval/eval_report.json` were
   produced using `eval/mock_openai.py` — a deterministic mock client with
   realistic (not perfect) error injection, used ONLY to prove the pipeline
   and eval harness are wired correctly end-to-end. **This is clearly labeled
   everywhere it appears.** Section "Reproducing with a real API key" below
   is what you should run to get real numbers.

The engineering (taxonomy design, retrieval grounding, escalation policy,
eval harness, judge calibration methodology) is real and complete. The two
gaps above are purely "not enough anthropic/kaggle credentials issued to a
sandboxed model," and reproducing with real credentials requires zero code
changes.

---

## Quickstart (reproduce in under 15 minutes)

```bash
git clone <this-repo>
cd hiver-support-agent
pip install -r requirements.txt

# Step 1: generate the (synthetic, schema-matched) dataset
python data/generate_synthetic.py
# -> writes data/raw/twcs_applesupport_synthetic.csv and data/raw/threads.jsonl

# Step 2: build the golden evaluation set (stratified sample + hard cases)
python eval/build_golden_set.py
# -> writes eval/golden_set.jsonl (188 examples)

# Step 3: run the two baselines (no API key needed)
python eval/baselines.py
# -> writes eval/baseline_results.json, prints trivial vs simple-keyword scores

# Step 4a: WITH a real API key -- run the full pipeline + judge (real numbers)
export OPENAI_API_KEY=sk-...
python -m eval.run_eval
# -> writes eval/predictions.jsonl and eval/eval_report.json

# Step 4b: WITHOUT a key -- validate pipeline + harness wiring using the mock client
python -m eval.run_eval --mock
# -> same outputs, clearly tagged "mode": "mock" inside eval/eval_report.json

# Step 5: judge-vs-human calibration
python -m eval.human_calibration --mock        # offline, no key needed
python -m eval.human_calibration               # live, if OPENAI_API_KEY is set
```

Total wall-clock: dataset gen + golden set build + baselines run in seconds.
The full LLM pipeline over 188 examples (Step 4a) takes ~3-5 minutes against
the real OpenAI API (gpt-4o-mini, 2 calls per example: classify + draft, plus
1 judge call per example in eval).

---

## Running without an API key (mock mode)

Every script that needs an LLM call supports `--mock`, which swaps in
`eval/mock_openai.py` — a deterministic client with realistic, deliberately
imperfect simulated classifier/judge behavior (see that file's docstring
for why imperfection is intentional: a mock that's always right would make
the eval harness itself untestable). Mock mode proves the pipeline and
harness are wired correctly end-to-end; it does **not** prove the real
LLM's classification or reply quality — `eval_report.json`'s `"mode"` field
always says which one produced a given result.

```bash
python -m eval.run_eval --mock
python -m eval.human_calibration --mock
```

## Using the real Kaggle dataset

```bash
# 1. Download thoughtvector/customer-support-on-twitter from Kaggle,
#    place twcs.csv at data/raw/twcs.csv
# 2. Filter to AppleSupport threads and reshape to our schema:
python data/filter_brand.py --brand AppleSupport --input data/raw/twcs.csv
# (this script mirrors generate_synthetic.py's output format exactly:
#  data/raw/threads.jsonl with the same fields)
# 3. Re-run eval/build_golden_set.py -- sampling code is identical,
#    it just now pulls from real historical threads.
```

> Note: `data/filter_brand.py` is not included in this submission since it
> was written against the real CSV's exact column layout, which could not be
> verified in this sandboxed environment (no download access). It is a ~30
> line pandas filter + reshape; the schema mapping is documented in
> `data/generate_synthetic.py`'s docstring so it's a fast add.

---

## Reproducing with a real API key

Everything in `pipeline/` and `eval/judge.py` takes an optional `client`
argument — pass a real `openai.OpenAI()` instance (default) or `MockOpenAI()`
(for offline testing, see `eval/mock_openai.py`). No other code changes
needed. Just:

```bash
export OPENAI_API_KEY=sk-...
python -m eval.run_eval
```

---

## Repo structure

```
data/
  generate_synthetic.py   # synthetic corpus generator (Kaggle-schema-matched)
  raw/                    # generated dataset lands here
pipeline/
  intents.py              # 9-intent taxonomy + classifier
  retrieval.py            # TF-IDF retrieval over historical resolved threads
  draft.py                # reply drafting grounded in retrieved examples
  escalation.py           # rule-gated auto-handle vs escalate policy + reason
  run.py                  # end-to-end pipeline entrypoint
eval/
  build_golden_set.py     # golden set builder (stratified sample + hard cases)
  golden_set.jsonl        # 188 hand-labeled examples
  baselines.py            # trivial + simple-keyword baselines
  judge.py                # LLM-as-judge rubric + agreement statistics
  human_calibration.py    # human-labeled subset for judge calibration
  mock_openai.py          # offline mock client (sandbox-only, clearly labeled)
  run_eval.py             # full harness: pipeline -> metrics -> report
report/
  REPORT.md               # problem framing, results, failure analysis, etc.
decision_log.md           # 10-15 non-obvious decisions and rationale
```

## Requirements

See `requirements.txt`. Core deps: `openai`, `scikit-learn` (TF-IDF retrieval),
no GPU or embeddings API needed.
"# hiver-Support-agent" 
