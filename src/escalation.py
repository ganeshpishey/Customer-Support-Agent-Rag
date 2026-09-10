"""Conservative, explainable escalation policy."""

def decide(intent, confidence, sentiment, config):
    policy = config["escalation"]
    if intent in policy.get("always_escalate_intents", []):
        return {"action": "escalate", "reason": "high-risk intent requires human review"}
    if confidence < policy["low_confidence_threshold"]:
        return {"action": "escalate", "reason": "low intent-classification confidence"}
    if policy.get("escalate_on_negative_sentiment") and sentiment == "strongly_negative":
        return {"action": "escalate", "reason": "strong negative sentiment; preserve customer recovery"}
    return {"action": "auto_handle", "reason": "high-confidence, low-risk intent with historical grounding"}
