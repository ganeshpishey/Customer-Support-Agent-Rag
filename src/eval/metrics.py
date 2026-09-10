import argparse, json, sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intents import keyword_classify
def read_golden(path):
 try: return pd.read_csv(path,dtype=str)
 except UnicodeDecodeError: return pd.read_csv(path,dtype=str,encoding='latin1')
def main():
 p=argparse.ArgumentParser(); p.add_argument('--predictions',required=True); p.add_argument('--golden',required=True); p.add_argument('--out',default='outputs/metrics.json'); a=p.parse_args(); pred=pd.read_csv(a.predictions,dtype=str); gold=read_golden(a.golden).replace('',pd.NA).dropna(subset=['true_intent','should_auto_handle']); df=gold.merge(pred,on='message_id',suffixes=('_gold',''))
 if df.empty: raise ValueError('Golden set has no completed labels; metrics would be misleading.')
 y=df.true_intent; system=df.predicted_intent; majority=y.mode()[0]; keyword=[keyword_classify(x)[0] for x in df.customer_msg]; wanted=df.should_auto_handle.str.lower().isin(['true','1','yes']); got=df.action.eq('auto_handle'); pr,rc,f,_=precision_recall_fscore_support(wanted,got,average='binary',zero_division=0)
 result={'n':len(df),'intent':{'system_accuracy':accuracy_score(y,system),'system_macro_f1':f1_score(y,system,average='macro',zero_division=0),'trivial_majority_accuracy':accuracy_score(y,[majority]*len(y)),'simple_keyword_accuracy':accuracy_score(y,keyword)},'auto_handle':{'precision':pr,'recall':rc,'f1':f,'unsafe_auto_handles':int((~wanted & got).sum()),'unnecessary_escalations':int((wanted & ~got).sum())}}
 with open(a.out,'w') as fh: json.dump(result,fh,indent=2)
 print(json.dumps(result,indent=2))
if __name__=='__main__': main()
