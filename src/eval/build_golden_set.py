import argparse, pandas as pd
def main():
 p=argparse.ArgumentParser(); p.add_argument('--predictions',required=True); p.add_argument('--n',type=int,default=200); p.add_argument('--out',default='eval/golden_set.csv'); a=p.parse_args(); df=pd.read_csv(a.predictions)
 parts=[x.sample(min(len(x),max(1,a.n//df.predicted_intent.nunique())),random_state=42) for _,x in df.groupby('predicted_intent')]; sample=pd.concat(parts)
 # Fill shortages from remaining examples, preserving representation of rare intents.
 if len(sample) < min(a.n, len(df)):
  remaining=df.drop(sample.index); sample=pd.concat([sample,remaining.sample(min(a.n-len(sample),len(remaining)),random_state=42)])
 sample=sample.sample(min(a.n,len(sample)),random_state=42)
 sample[['message_id','customer_msg']].assign(true_intent='',should_auto_handle='',escalation_reason='',notes='',human_groundedness='',human_correctness='',human_tone_match='',human_actionability='').to_csv(a.out,index=False); print(f'Wrote {len(sample)} stratified rows. Hand-label every row before reporting results.')
if __name__=='__main__': main()
