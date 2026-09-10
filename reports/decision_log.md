# Decision log

1. Chose AmazonHelp because it is a high-volume, general retail support handle with diverse resolution patterns.
2. Capped development to 5,000 direct pairs so full reproduction stays CPU-friendly.
3. Used direct parent-child tweet links rather than fragile full-thread reconstruction.
4. Retained boilerplate replies but capped each at 100 to reflect authentic DM handoff while limiting dominance.
5. Defined eight operationally distinct intents, with a safety catch-all.
6. Used weak supervision only for development; hand labels remain the source of truth.
7. Used TF-IDF retrieval for transparent, offline reproducibility.
8. Used extractive historical replies to minimize hallucinated policies.
9. Redacted long numeric tokens before returning a historic reply.
10. Set a 0.52 confidence threshold to favor human review under ambiguity.
11. Always escalate safety/legal/fraud language and strongly negative sentiment.
12. Stratify the golden set by predicted intent to expose rare-class failures.
13. Compare against majority and keyword baselines to isolate real lift.
14. Use deterministic Groq `openai/gpt-oss-20b` judging and compare it with 40 blinded human scores.
15. Refuse to report metrics on blank gold labels because it would create false evidence.
16. Held out all 200 golden message IDs from both training and retrieval before final scoring to prevent exact-match retrieval leakage.
17. Used Groq `openai/gpt-oss-20b` as the judge after xAI API credits were unavailable; judge agreement is reported rather than assumed.
