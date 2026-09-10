"""Persistent TF-IDF retrieval over historical resolutions; no network/API needed."""
import argparse
from functools import lru_cache
import joblib, pandas as pd, yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def build(pairs_path='data/resolution_pairs.csv', index_path='data/retrieval.joblib'):
    pairs=pd.read_csv(pairs_path).fillna(''); vectorizer=TfidfVectorizer(ngram_range=(1,2), min_df=1, sublinear_tf=True, stop_words='english')
    joblib.dump({'pairs':pairs,'vectorizer':vectorizer,'matrix':vectorizer.fit_transform(pairs.customer_msg)}, index_path)
@lru_cache(maxsize=2)
def load_index(index_path):
    return joblib.load(index_path)
def retrieve(query, k=5, index_path='data/retrieval.joblib'):
    store=load_index(index_path); score=cosine_similarity(store['vectorizer'].transform([str(query)]),store['matrix']).ravel(); order=score.argsort()[::-1][:k]
    return [{**store['pairs'].iloc[i].to_dict(),'similarity':float(score[i])} for i in order]
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--pairs',default='data/resolution_pairs.csv'); ap.add_argument('--out'); a=ap.parse_args(); cfg=yaml.safe_load(open(a.config)); build(a.pairs, a.out or cfg['retrieval']['index_path']); print('Wrote retrieval index.')
if __name__=='__main__': main()
