"""Run the LLM judge and compare its four scores with human annotations."""
import argparse
import json
from pathlib import Path
import sys
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judge import judge_reply, human_judge_agreement

DIMENSIONS = ['groundedness', 'correctness', 'tone_match', 'actionability']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='eval/human_judge_subset.csv')
    parser.add_argument('--out', default='outputs/judge_scores.csv')
    parser.add_argument('--summary-out', default='outputs/judge_agreement.json')
    parser.add_argument('--limit', type=int, help='Optional smoke-test row limit.')
    parser.add_argument('--offset', type=int, default=0, help='Starting row for a resumable chunk.')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    data = pd.read_csv(args.input, dtype=str).fillna('')
    if args.offset:
        data = data.iloc[args.offset:]
    if args.limit:
        data = data.head(args.limit)
    required = [f'human_{dimension}' for dimension in DIMENSIONS]
    if data[required].eq('').any().any():
        raise ValueError('All human rubric scores must be filled before calculating agreement.')
    def score_row(item):
        number, row = item
        scores = judge_reply(row.customer_msg, row.drafted_reply,
                             [{'brand_reply': row.retrieved_historical_replies}])
        result = row.to_dict()
        for dimension in DIMENSIONS:
            value = scores.get(dimension)
            if not isinstance(value, int) or value not in range(1, 6):
                raise ValueError(f'Invalid judge score at row {number}: {scores}')
            result[f'judge_{dimension}'] = value
        return number, result
    rows = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for completed, (number, result) in enumerate(executor.map(score_row, data.iterrows()), start=1):
            rows.append(result)
            print(f'Judged {completed}/{len(data)}', flush=True)
    scored = pd.DataFrame(rows)
    scored.to_csv(args.out, index=False, encoding='utf-8')
    agreement = {dimension: human_judge_agreement(
        scored[f'judge_{dimension}'].astype(int).tolist(),
        scored[f'human_{dimension}'].astype(int).tolist()) for dimension in DIMENSIONS}
    agreement['mean_quadratic_kappa'] = sum(agreement.values()) / len(DIMENSIONS)
    with open(args.summary_out, 'w', encoding='utf-8') as output:
        json.dump({'n': len(scored), 'quadratic_kappa': agreement}, output, indent=2)
    print(json.dumps({'n': len(scored), 'quadratic_kappa': agreement}, indent=2))

if __name__ == '__main__':
    main()
