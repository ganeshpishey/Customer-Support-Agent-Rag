import argparse, json, re
import pandas as pd, yaml
from intents import IntentClassifier
from reply_generator import draft_reply
from escalation import decide
def sentiment(text): return 'strongly_negative' if re.search(r'worst|hate|angry|terrible|fraud|scam|never|disgust',str(text),re.I) else 'neutral'
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--input',required=True); ap.add_argument('--config',required=True); ap.add_argument('--out',required=True); ap.add_argument('--limit',type=int); ap.add_argument('--model-path',default='data/intent_classifier.joblib'); ap.add_argument('--index-path'); a=ap.parse_args(); cfg=yaml.safe_load(open(a.config)); clf=IntentClassifier(a.model_path); rows=[]
 input_df=pd.read_csv(a.input).fillna('')
 if a.limit: input_df=input_df.head(a.limit)
 for _,row in input_df.iterrows():
  msg=row.customer_msg; intent,confidence=clf.predict(msg); draft=draft_reply(msg,intent,cfg['retrieval']['top_k'],a.index_path or cfg['retrieval']['index_path']); d=decide(intent,confidence,sentiment(msg),cfg)
  rows.append({'message_id':row.get('message_id',''),'customer_msg':msg,'predicted_intent':intent,'intent_confidence':round(confidence,3),'drafted_reply':draft['reply'],'grounded_on':json.dumps([{'message_id':x.get('message_id'),'similarity':round(x['similarity'],3)} for x in draft['grounded_on']]),'action':d['action'],'escalation_reason':d['reason']})
 pd.DataFrame(rows).to_csv(a.out,index=False); print(f'Wrote {len(rows)} predictions to {a.out}')
if __name__=='__main__': main()
