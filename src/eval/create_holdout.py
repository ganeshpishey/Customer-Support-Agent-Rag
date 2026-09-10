"""Create a leakage-free train/retrieval corpus from a labelled golden set."""
import argparse
from pathlib import Path
import sys
import pandas as pd

def read_csv_robust(path):
    try:
        return pd.read_csv(path, dtype=str)
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding='latin1')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pairs', default='data/resolution_pairs.csv')
    parser.add_argument('--golden', default='eval/golden_set.csv')
    parser.add_argument('--train-out', default='data/train_resolution_pairs.csv')
    parser.add_argument('--test-out', default='data/golden_inbound.csv')
    parser.add_argument('--model-out', default='data/intent_classifier_holdout.joblib')
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from intents import IntentClassifier
    pairs = pd.read_csv(args.pairs, dtype=str).fillna('')
    golden = read_csv_robust(args.golden).fillna('')
    test_ids = set(golden['message_id'].astype(str))
    test = pairs[pairs['message_id'].astype(str).isin(test_ids)].drop_duplicates('message_id')
    missing = test_ids - set(test['message_id'].astype(str))
    if missing:
        raise ValueError(f'{len(missing)} golden IDs were not found in resolution pairs; do not evaluate a partial holdout.')
    train = pairs[~pairs['message_id'].astype(str).isin(test_ids)].copy()
    if len(test) != len(golden):
        raise ValueError(f'Expected {len(golden)} test examples, found {len(test)} after deduplication.')
    train.to_csv(args.train_out, index=False)
    test[['message_id','customer_msg']].to_csv(args.test_out, index=False)
    IntentClassifier(args.model_out).fit(train.customer_msg, train.weak_intent)
    print(f'Leakage-free split: train/retrieval={len(train)}, held-out golden={len(test)}')

if __name__ == '__main__':
    main()
