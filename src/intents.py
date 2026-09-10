"""AmazonHelp intent taxonomy and local intent classifier."""
from __future__ import annotations
import re
from pathlib import Path
import joblib
import numpy as np

INTENTS = {
 "delivery_order": "Late, missing, wrongly delivered, or unavailable orders and deliveries.",
 "returns_refunds": "Returns, refunds, replacements, cancellations, or their status.",
 "payments_gift_cards": "Charges, payment methods, invoices, gift cards, or promotional balances.",
 "account_prime": "Sign-in, verification, account access, Prime membership, or subscription problems.",
 "digital_devices": "Kindle, Alexa, Fire, apps, streaming, downloads, or other digital products.",
 "product_information": "Pre-purchase product, policy, availability, or how-to questions.",
 "feedback_complaint": "Praise, feedback, or a complaint without a specific resolvable issue.",
 "safety_legal_other": "Safety, fraud, legal/media threats, abuse, or unclear high-risk requests.",
}
KEYWORDS = {
 "delivery_order": r"deliver|delivery|package|parcel|order|ship|tracking|arriv|late|missing",
 "returns_refunds": r"refund|return|replacement|replace|cancel|cancell",
 "payments_gift_cards": r"charg|payment|paid|card|invoice|bill|gift card|balance",
 "account_prime": r"login|log in|sign.?in|password|account|verify|prime|membership|subscribe",
 "digital_devices": r"kindle|alexa|echo|fire ?tv|firestick|amazon music|video|download|stream",
 "product_information": r"how (do|can)|what (is|are)|where|when|availability|available|policy|price",
 "feedback_complaint": r"thank|love|great|awesome|worst|terrible|disappoint|hate|feedback",
 "safety_legal_other": r"fraud|scam|police|lawyer|legal|lawsuit|unsafe|danger|media|press",
}

def keyword_classify(text):
    scores = {label: len(re.findall(pattern, str(text).lower())) for label, pattern in KEYWORDS.items()}
    label, score = max(scores.items(), key=lambda item: item[1])
    return (label if score else "safety_legal_other"), min(.95, .45 + .15 * score)

class IntentClassifier:
    def __init__(self, model_path="data/intent_classifier.joblib"):
        self.model_path = Path(model_path)
        self.model = joblib.load(self.model_path) if self.model_path.exists() else None
    def fit(self, messages, labels):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        self.model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=500, class_weight="balanced"))
        self.model.fit(list(messages), list(labels)); self.model_path.parent.mkdir(parents=True, exist_ok=True); joblib.dump(self.model, self.model_path)
    def predict(self, text):
        if self.model is None: return keyword_classify(text)
        p = self.model.predict_proba([str(text)])[0]; i = int(np.argmax(p)); return str(self.model.classes_[i]), float(p[i])
