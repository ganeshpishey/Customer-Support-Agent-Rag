"""Extractive RAG reply draft: historical wording plus provenance, never invented facts."""
import re
from retrieval import retrieve
def draft_reply(customer_msg, intent, k=5, index_path='data/retrieval.joblib'):
    evidence=retrieve(customer_msg,k,index_path); reply=str(evidence[0].get('brand_reply') if evidence else 'Thanks for reaching out. Please contact our support team so we can help.')
    return {'reply':re.sub(r'\b\d{8,}\b','[order details]',re.sub(r'@\w+','@AmazonHelp',reply)), 'grounded_on':evidence}
