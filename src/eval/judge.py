"""LLM-as-judge rubric with explicit offline fallback for development only."""
import os
import json
import re
from dotenv import load_dotenv
from sklearn.metrics import cohen_kappa_score
load_dotenv()
RUBRIC = "Score groundedness, correctness, tone_match and actionability from 1 (poor) to 5 (excellent). Return JSON only."
def judge_reply(customer_msg, drafted_reply, grounded_on):
    if os.getenv('OPENAI_API_KEY') or os.getenv('XAI_API_KEY') or os.getenv('GROQ_API_KEY'):
        from openai import OpenAI
        is_groq = bool(os.getenv('GROQ_API_KEY'))
        is_xai = bool(os.getenv('XAI_API_KEY')) and not is_groq and not os.getenv('OPENAI_API_KEY')
        api_key = os.getenv('GROQ_API_KEY') if is_groq else (os.getenv('XAI_API_KEY') if is_xai else os.getenv('OPENAI_API_KEY'))
        base_url = 'https://api.groq.com/openai/v1' if is_groq else ('https://api.x.ai/v1' if is_xai else None)
        client = OpenAI(api_key=api_key, base_url=base_url)
        prompt=f'{RUBRIC}\nCustomer: {customer_msg}\nDraft: {drafted_reply}\nHistorical evidence: {grounded_on}'
        model = os.getenv('GROQ_JUDGE_MODEL','openai/gpt-oss-20b') if is_groq else (os.getenv('XAI_JUDGE_MODEL','grok-4.6') if is_xai else os.getenv('JUDGE_MODEL','gpt-4o-mini'))
        try:
            response=client.chat.completions.create(model=model,messages=[{'role':'user','content':prompt}],response_format={'type':'json_object'},temperature=0)
            return json.loads(response.choices[0].message.content)
        except Exception:
            fallback = prompt + '\nReturn exactly this JSON object, with integer values only: {"groundedness": 1, "correctness": 1, "tone_match": 1, "actionability": 1}'
            response=client.chat.completions.create(model=model,messages=[{'role':'user','content':fallback}],temperature=0)
            content=response.choices[0].message.content
            match=re.search(r'\{.*\}',content,re.S)
            if not match: raise ValueError(f'Judge did not return JSON: {content[:200]}')
            return json.loads(match.group(0))
    evidence=' '.join(str(x.get('brand_reply','')) for x in grounded_on).lower(); reply=str(drafted_reply).lower(); overlap=len(set(reply.split()) & set(evidence.split()))
    return {'groundedness':4 if overlap>=3 else 2,'correctness':3,'tone_match':3,'actionability':4 if '?' in reply or 'please' in reply else 2,'judge_mode':'offline_smoke_test'}
def human_judge_agreement(judge_scores, human_scores): return float(cohen_kappa_score(judge_scores,human_scores,weights='quadratic'))
