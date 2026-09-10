# AmazonHelp Historical-Resolution Support Agent

An evaluated, retrieval-grounded AI support-agent prototype built for the Hiver SDE Intern take-home assignment. The system uses real AmazonHelp Twitter-support conversations to classify incoming customer messages, retrieve historically similar resolutions, draft a grounded response, and decide whether to auto-handle or escalate to a human.

> Status: research prototype. It is deliberately conservative and is **not** suitable for unsupervised customer-facing deployment. See [Results](#results) and [Limitations](#limitations).

## What it does

For every incoming customer message, the pipeline produces:

1. **Intent** - one of eight support intents, plus a confidence score.
2. **Historical grounding** - top-k similar prior customer messages and their actual AmazonHelp replies.
3. **Drafted reply** - a redacted, extractive historical-style response rather than an invented policy answer.
4. **Routing decision** - `auto_handle` or `escalate`, with an explicit policy reason.

### Intent taxonomy

| Intent | Meaning |
|---|---|
| `delivery_order` | Late, missing, incorrect, or tracking-related delivery/order issue |
| `returns_refunds` | Return, refund, replacement, or cancellation |
| `payments_gift_cards` | Charges, payment, invoice, gift-card, or balance issue |
| `account_prime` | Login, account access, verification, Prime, or subscription issue |
| `digital_devices` | Kindle, Alexa, Fire, streaming, download, or digital-product issue |
| `product_information` | Availability, policy, how-to, or pre-purchase question |
| `feedback_complaint` | Praise, feedback, or a general complaint without a clear action request |
| `safety_legal_other` | Fraud, safety, legal, abuse, or unclear/high-risk request |

## Architecture

```text
incoming customer tweet
         |
         v
TF-IDF + logistic-regression intent classifier
         |
         +------------------------------+
         |                              |
         v                              v
TF-IDF historical-resolution       escalation policy
retrieval (top-k similar cases)    - high-risk intent
         |                          - low confidence
         v                          - strong negative sentiment
extractive/redacted reply                  |
         |                                 v
         +----------------------> auto_handle or escalate + reason
```

The core system works locally and does not need an LLM API key. An optional Groq LLM is used only for independent reply-quality judging.

## Repository layout

```text
configs/config.yaml                 Pipeline settings
data/resolution_pairs.csv           5,000 sampled historical AmazonHelp pairs
data/train_resolution_pairs.csv     4,800 pairs used for leakage-free train/retrieval
data/golden_inbound.csv             200 held-out messages
eval/golden_set.csv                 200 human-labelled evaluation examples
eval/human_judge_subset.csv         40 human-scored reply-quality examples
outputs/                            Predictions, metrics, and judge scores
reports/report.md                   Final assignment report
reports/decision_log.md             Design decisions and rationale
src/                                Pipeline and evaluation source code
```

The full raw Kaggle `twcs.csv`, local `.joblib` files, and `.env` are intentionally ignored by Git.

## Quick start: run the agent

### 1. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 2. Add messages

Edit `data/new_messages.csv`. It must contain these columns:

```csv
message_id,customer_msg
demo_1,"My order was due yesterday and tracking has not changed. Can you help?"
demo_2,"How do I cancel my Prime membership?"
```

### 3. Run predictions

```powershell
python src/pipeline.py --input data/new_messages.csv --config configs/config.yaml --out outputs/new_predictions.csv
```

Open `outputs/new_predictions.csv`. Key output columns:

| Column | Description |
|---|---|
| `predicted_intent` | Predicted intent label |
| `intent_confidence` | Classifier confidence from 0 to 1 |
| `drafted_reply` | Retrieved/redacted historical-style reply |
| `grounded_on` | Retrieval evidence IDs and cosine-similarity scores |
| `action` | `auto_handle` or `escalate` |
| `escalation_reason` | Explainable routing rationale |

**Operational guidance:** treat `auto_handle` as a recommendation for review, not permission to send automatically. Never pass real order numbers, addresses, account credentials, or payment information through this prototype.

## Reproduce from the raw Kaggle dataset

1. Download the Customer Support on Twitter Kaggle dataset (`thoughtvector/customer-support-on-twitter`).
2. Put `twcs.csv` at `data/twcs.csv`.
3. Run the data/build pipeline:

```powershell
python src/data_prep.py --config configs/config.yaml
python src/retrieval.py --config configs/config.yaml
python src/pipeline.py --input data/sample_inbound.csv --config configs/config.yaml --out outputs/predictions.csv
```

`data_prep.py` identifies direct inbound customer tweet -> AmazonHelp reply pairs, caps repeated boilerplate responses, samples 5,000 pairs, and trains the development intent model.

## Golden-set evaluation protocol

The project includes a completed 200-row golden set. To repeat the process from scratch:

1. Generate candidate predictions and a stratified annotation file:

```powershell
python src/eval/build_golden_set.py --predictions outputs/predictions.csv --n 200 --out eval/golden_set.csv
```

2. Hand-label every row in `eval/golden_set.csv`:

   - `true_intent`
   - `should_auto_handle` (`true` or `false`)
   - `escalation_reason`
   - `notes`

3. Create a strict held-out split. This removes all golden message IDs from **both** classifier training and retrieval, preventing exact-match retrieval leakage:

```powershell
python src/eval/create_holdout.py
python src/retrieval.py --config configs/config.yaml --pairs data/train_resolution_pairs.csv --out data/retrieval_holdout.joblib
python src/pipeline.py --input data/golden_inbound.csv --config configs/config.yaml --out outputs/predictions_holdout.csv --model-path data/intent_classifier_holdout.joblib --index-path data/retrieval_holdout.joblib
python src/eval/metrics.py --predictions outputs/predictions_holdout.csv --golden eval/golden_set.csv --out outputs/metrics_holdout.json
```

## Results

Results are measured on a 200-example manually labelled, leakage-free held-out set. The classifier/retrieval corpus contains the other 4,800 examples.

| Metric | Trivial baseline | Simple baseline | System |
|---|---:|---:|---:|
| Intent accuracy | 35.5% majority class | 40.0% keyword rules | **43.0%** |
| Intent macro-F1 | N/A | not reported | **35.2%** |
| Auto-handle precision | not run | not run | **13.3%** |
| Auto-handle recall | not run | not run | **7.1%** |
| Auto-handle F1 | not run | not run | **9.3%** |

The system produced 26 unsafe auto-handles and 52 unnecessary escalations. These values are a safety signal: the current system should be used for triage/draft assistance with human review, not autonomous reply sending.

## LLM-as-judge evaluation

The 40-row `eval/human_judge_subset.csv` is balanced at five examples per intent. A human and Groq `openai/gpt-oss-20b` independently scored each response from 1 to 5 for groundedness, correctness, tone match, and actionability. Judge/human agreement uses quadratic Cohen's kappa.

| Dimension | Groq mean | Human mean | Kappa |
|---|---:|---:|---:|
| Groundedness | 2.18 | 2.68 | 0.365 |
| Correctness | 2.83 | 2.60 | 0.401 |
| Tone match | 2.78 | 3.65 | 0.146 |
| Actionability | 2.55 | 2.50 | 0.397 |
| Mean agreement | - | - | **0.327** |

To reproduce judge scoring, copy `.env.example` to `.env`, add a Groq API key, and run chunks (the chunking is deliberate for free-tier rate limits):

```env
GROQ_API_KEY=your_key_here
GROQ_JUDGE_MODEL=openai/gpt-oss-20b
```

```powershell
python src/eval/run_judge.py --input eval/human_judge_subset.csv --out outputs/judge_chunk_00.csv --summary-out outputs/judge_chunk_00.json --offset 0 --limit 5 --workers 5
```

Run equivalent commands with offsets `5`, `10`, ..., `35`, then aggregate the eight output CSVs:

```powershell
python src/eval/aggregate_judge_scores.py --chunks outputs/judge_chunk_00.csv outputs/judge_chunk_05.csv outputs/judge_chunk_10.csv outputs/judge_chunk_15.csv outputs/judge_chunk_20.csv outputs/judge_chunk_25.csv outputs/judge_chunk_30.csv outputs/judge_chunk_35.csv --out outputs/judge_scores.csv --summary-out outputs/judge_agreement.json
```

## Limitations

- Intent training uses weak keyword-derived labels because the raw Twitter data has no intent labels.
- 43.0% accuracy is only three percentage points above the keyword baseline.
- The golden set is stratified for coverage, not prevalence-representative of production traffic.
- Historical replies may be grounded yet generic, stale, or wrong for the new customer's account context.
- The system escalates 85% of held-out messages; its automation value is currently limited.
- LLM-judge agreement is low-to-moderate (mean kappa 0.327), especially for tone.

Read the fuller analysis, including real failure examples, in [reports/report.md](reports/report.md).

## Sources and acknowledgements

- Dataset: Kaggle `thoughtvector/customer-support-on-twitter`
- Libraries: pandas, scikit-learn, joblib, PyYAML, OpenAI-compatible API client
- Independent reply judge: Groq `openai/gpt-oss-20b`
- This repository was developed with AI coding assistance; all methodology and limitations are documented in the report and decision log.
