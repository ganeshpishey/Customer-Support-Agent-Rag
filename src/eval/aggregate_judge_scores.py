"""Combine resumable judge chunks and report rubric means plus human agreement."""
import argparse
import json
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

DIMENSIONS = ['groundedness', 'correctness', 'tone_match', 'actionability']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', nargs='+', required=True)
    parser.add_argument('--out', default='outputs/judge_scores.csv')
    parser.add_argument('--summary-out', default='outputs/judge_agreement.json')
    args = parser.parse_args()
    scores = pd.concat([pd.read_csv(path) for path in args.chunks], ignore_index=True)
    if len(scores) != 40 or scores.message_id.nunique() != 40:
        raise ValueError(f'Expected 40 unique scores, got {len(scores)} rows / {scores.message_id.nunique()} IDs.')
    scores.to_csv(args.out, index=False, encoding='utf-8')
    kappa = {dimension: float(cohen_kappa_score(scores[f'human_{dimension}'], scores[f'judge_{dimension}'], weights='quadratic')) for dimension in DIMENSIONS}
    judge_mean = {dimension: float(scores[f'judge_{dimension}'].mean()) for dimension in DIMENSIONS}
    human_mean = {dimension: float(scores[f'human_{dimension}'].mean()) for dimension in DIMENSIONS}
    summary = {'n': len(scores), 'quadratic_kappa': kappa, 'mean_quadratic_kappa': sum(kappa.values()) / len(kappa), 'judge_mean': judge_mean, 'human_mean': human_mean}
    with open(args.summary_out, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(json.dumps(summary, indent=2))

if __name__ == '__main__':
    main()
