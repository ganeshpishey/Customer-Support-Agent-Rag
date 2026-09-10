"""Create customer-message -> AmazonHelp-resolution pairs from twcs.csv."""
import argparse
from pathlib import Path
import pandas as pd
import yaml
from intents import keyword_classify, IntentClassifier

def ids(value):
    if pd.isna(value): return []
    return [x.strip() for x in str(value).replace('[','').replace(']','').split(',') if x.strip() and x.strip() != 'nan']
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config', required=True); a=ap.parse_args()
    cfg=yaml.safe_load(open(a.config)); raw=Path(cfg['brand']['raw_csv_path'])
    if not raw.exists(): raise FileNotFoundError(f"Dataset not found: {raw}. Put Kaggle twcs.csv there or change raw_csv_path.")
    use=['tweet_id','author_id','inbound','created_at','text','in_response_to_tweet_id']
    df=pd.read_csv(raw, usecols=lambda c: c in use, dtype=str, keep_default_na=False)
    df['tweet_id']=df.tweet_id.astype(str); lookup=df.set_index('tweet_id'); brand=cfg['brand']['handle'].lower()
    rows=[]
    for _, reply in df[df.author_id.str.lower().eq(brand)].iterrows():
        for parent_id in ids(reply.in_response_to_tweet_id):
            if parent_id not in lookup.index: continue
            parent=lookup.loc[parent_id]
            if isinstance(parent, pd.DataFrame): parent=parent.iloc[0]
            if str(parent.get('inbound','')).lower() not in ('true','1'): continue
            msg=str(parent.text).strip(); answer=str(reply.text).strip()
            if msg and answer: rows.append({'conversation_id':parent_id,'message_id':parent_id,'customer_msg':msg,'brand_reply':answer,'customer_msg_created_at':parent.created_at})
    pairs=pd.DataFrame(rows).drop_duplicates(['message_id','brand_reply'])
    if pairs.empty: raise ValueError(f'No direct inbound -> {cfg["brand"]["handle"]} reply pairs. Check handle/csv schema.')
    pairs=pairs.groupby('brand_reply', group_keys=False).head(100).sample(min(len(pairs), cfg['sampling']['n_conversations']), random_state=cfg['sampling']['seed'])
    pairs['weak_intent']=[keyword_classify(x)[0] for x in pairs.customer_msg]
    pairs.to_csv('data/resolution_pairs.csv',index=False); pairs[['message_id','customer_msg']].to_csv('data/sample_inbound.csv',index=False)
    IntentClassifier().fit(pairs.customer_msg,pairs.weak_intent); print(f'Wrote {len(pairs)} pairs and trained a weakly-supervised intent model.')
if __name__=='__main__': main()
