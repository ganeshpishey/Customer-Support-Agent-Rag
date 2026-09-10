# AmazonHelp historical-resolution support agent

This is a CPU-first, reproducible support-agent take-home. Given an incoming tweet, it predicts one of eight AmazonHelp-specific intents, retrieves historically similar AmazonHelp resolution pairs, drafts an extractive grounded reply, and either auto-handles or escalates with an explicit policy reason.

## Reproduce (under 15 minutes after the CSV is available)

Place the Kaggle `twcs.csv` at `data/twcs.csv`, then run:

```powershell
python -m pip install -r requirements.txt
python src/data_prep.py --config configs/config.yaml
python src/retrieval.py --config configs/config.yaml
python src/pipeline.py --input data/sample_inbound.csv --config configs/config.yaml --out outputs/predictions.csv
python src/eval/build_golden_set.py --predictions outputs/predictions.csv --n 200
```

Hand-label every row of `eval/golden_set.csv`, then run:

```powershell
python src/eval/metrics.py --predictions outputs/predictions.csv --golden eval/golden_set.csv
```

After hand-labelling, create the leakage-free split and headline results:

```powershell
python src/eval/create_holdout.py
python src/retrieval.py --config configs/config.yaml --pairs data/train_resolution_pairs.csv --out data/retrieval_holdout.joblib
python src/pipeline.py --input data/golden_inbound.csv --config configs/config.yaml --out outputs/predictions_holdout.csv --model-path data/intent_classifier_holdout.joblib --index-path data/retrieval_holdout.joblib
python src/eval/metrics.py --predictions outputs/predictions_holdout.csv --golden eval/golden_set.csv --out outputs/metrics_holdout.json
```

Do not publish a headline score until the golden set is fully hand-labelled and held out from both training and retrieval. The repository intentionally fails rather than reporting a number from blank labels.

## Method

`data_prep.py` joins an AmazonHelp outbound tweet to its direct inbound parent and caps repeated boilerplate replies at 100, preserving common actual resolution patterns without letting a single DM template dominate. A TF-IDF/logistic-regression classifier is weakly supervised from keyword seed labels; the golden set is the authoritative evaluation and should be used for the next iteration. Retrieval is TF-IDF cosine similarity over the customer side of historical pairs. The reply is a redacted historical response, rather than a creative LLM completion, to avoid unsupported policy claims. The escalation policy sends safety/legal cases, low-confidence classifications, and strongly negative messages to a person.

## Golden-set protocol

Sample 200 examples stratified by predicted intent. Include at least 10 each of: multi-issue tweets, terse tweets, misspellings/slang, and high-risk/sarcastic messages. One author labels intent and auto-handle eligibility while blinded to the system prediction; hand-score 40 replies on the four judge rubric dimensions. `src/eval/judge.py` uses GPT-4o-mini at temperature 0 when `OPENAI_API_KEY` is set and reports quadratic Cohen's kappa against those 40 human scores. Its offline heuristic is smoke-test-only and must not be used in the report.

## Limits and citations

The implementation uses pandas and scikit-learn TF-IDF/logistic regression APIs; see their official documentation. Dataset: Kaggle `thoughtvector/customer-support-on-twitter`. No result values are included because the data file was not supplied in this workspace.
