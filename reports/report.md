# Report - AmazonHelp historical-resolution support agent

## Problem framing

This system classifies an incoming AmazonHelp tweet, retrieves similar historical AmazonHelp resolutions, drafts a grounded reply, and decides auto-handle or escalate with a reason. Good means a reply is traceable to historical evidence, does not invent account facts, gives a useful next step, and routes risky cases to a person. It is a triage/drafting prototype, not an autonomous account-resolution system: I did not build multi-turn state, order lookup, multilingual translation, or a handoff UI.

The taxonomy is `delivery_order`, `returns_refunds`, `payments_gift_cards`, `account_prime`, `digital_devices`, `product_information`, `feedback_complaint`, and `safety_legal_other`.

## Data and evaluation protocol

I joined direct inbound tweets to direct AmazonHelp replies from the Kaggle Twitter-support data and sampled 5,000 pairs. I hand-labelled a 200-example golden set. For valid evaluation, every golden message ID was removed from both the 4,800-pair classifier training corpus and historical reply retrieval index. Results below are therefore from a strict held-out split.

## Results versus baselines

| Metric | Trivial baseline | Simple baseline | System |
|---|---:|---:|---:|
| Intent accuracy | 35.5% majority class | 40.0% keyword rules | **43.0%** |
| Intent macro-F1 | N/A | not reported | **35.2%** |
| Auto-handle precision | not run | not run | **13.3%** |
| Auto-handle recall | not run | not run | **7.1%** |
| Auto-handle F1 | not run | not run | **9.3%** |

The system had 26 unsafe auto-handles and 52 unnecessary escalations. It is not ready for unsupervised customer-facing automation.

### LLM-as-judge

On a balanced held-out 40-example subset, I scored replies myself and Groq `openai/gpt-oss-20b` independently scored the same four 1-5 dimensions at temperature 0. Agreement is quadratic Cohen's kappa.

| Dimension | Groq mean | Human mean | Kappa |
|---|---:|---:|---:|
| Groundedness | 2.18 | 2.68 | 0.365 |
| Correctness | 2.83 | 2.60 | 0.401 |
| Tone match | 2.78 | 3.65 | 0.146 |
| Actionability | 2.55 | 2.50 | 0.397 |
| Mean kappa | - | - | **0.327** |

Agreement is low-to-moderate, especially for tone, so judge scores are diagnostic rather than a substitute for human review.

## Failure analysis - five observed modes

1. **General complaints go to the safety catch-all.** Tweet `279683` is human-labelled `feedback_complaint` but predicted `safety_legal_other`; weak seed labels lack complaint coverage.
2. **Cancellation/return issues blur with delivery.** Tweet `831080` is `returns_refunds` but predicted `delivery_order`, then receives an irrelevant survey reply.
3. **Multilingual and terse messages are over-escalated.** `505782` ('Vendido x Amazon EU') is `product_information` but becomes `safety_legal_other`; the English-heavy vocabulary has low confidence.
4. **Historical retrieval can be contextually wrong.** In `782539`, a refund request retrieves delivery-estimate wording; lexical similarity misses the requested action.
5. **Repeated delivery failures get auto-handled.** Tweet `490381` describes repeated late packages but was auto-handled; the simple negative-sentiment lexicon misses this form of frustration.

## What is misleading about my headline number?

43.0% accuracy is only three points above the keyword baseline, and the evaluation set is intent-stratified rather than prevalence-representative. Training labels are weak keyword labels, not independent human labels. The system escalates 85% of held-out messages (170/200), which limits unsafe automation but makes raw classification metrics insufficient. Extractive replies may be grounded but generic or stale, and judge-vs-human kappa is only 0.327.

## What I would do with one more week

- Double-label and adjudicate golden examples.
- Replace weak labels with reviewed training labels and use a chronological split.
- Add language identification and action-aware retrieval/reranking.
- Calibrate confidence and optimize escalation against a cost-weighted safety objective.

## Citations

Dataset: Kaggle `thoughtvector/customer-support-on-twitter`. Implementation uses pandas and scikit-learn TF-IDF/logistic-regression APIs. Groq `openai/gpt-oss-20b` was used as the independent reply judge. Code was developed with AI coding assistance.
