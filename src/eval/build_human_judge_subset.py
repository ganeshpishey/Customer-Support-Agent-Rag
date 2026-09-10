"""Create a balanced 40-row human-scoring sheet from held-out predictions."""
import argparse
import json
from pathlib import Path
import pandas as pd

def read_robust(path):
    try: return pd.read_csv(path, dtype=str)
    except UnicodeDecodeError: return pd.read_csv(path, dtype=str, encoding='latin1')

def normalize_id(value):
    value = str(value)
    return value[:-2] if value.endswith('.0') else value

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--golden', default='eval/golden_set.csv')
    parser.add_argument('--predictions', default='outputs/predictions_holdout.csv')
    parser.add_argument('--train-pairs', default='data/train_resolution_pairs.csv')
    parser.add_argument('--n-per-intent', type=int, default=5)
    parser.add_argument('--out', default='eval/human_judge_subset.csv')
    args = parser.parse_args()
    golden = read_robust(args.golden).fillna('')
    predictions = pd.read_csv(args.predictions, dtype=str).fillna('')
    pairs = pd.read_csv(args.train_pairs, dtype=str).fillna('')
    pair_reply = {normalize_id(row.message_id): row.brand_reply for _, row in pairs.iterrows()}
    data = golden.merge(predictions[['message_id','drafted_reply','grounded_on']], on='message_id')
    subset = data.groupby('true_intent', group_keys=False).sample(n=args.n_per_intent, random_state=42)
    def evidence(value):
        ids = [normalize_id(item.get('message_id','')) for item in json.loads(value)]
        return '\n---\n'.join(pair_reply.get(item, '[source unavailable]') for item in ids[:3])
    subset['retrieved_historical_replies'] = subset['grounded_on'].map(evidence)
    keep = ['message_id','customer_msg','true_intent','drafted_reply','retrieved_historical_replies',
            'human_groundedness','human_correctness','human_tone_match','human_actionability']
    subset[keep].sort_values('true_intent').to_csv(args.out, index=False, encoding='utf-8')
    print(f'Wrote {len(subset)} rows to {args.out}')
if __name__ == '__main__':
    main()
