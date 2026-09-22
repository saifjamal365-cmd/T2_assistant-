# T2 Assistant - Evaluation Report

Dataset: 20 questions (18 answerable, 2 deliberately unanswerable).

## Headline numbers

| Metric | Value | Requirement |
|---|---|---|
| Retrieval recall (top-5) | 88.9% | NFR-02 |
| Answer accuracy (answerable) | 88.9% | TO-7 |
| Honesty rate (says "I don't know" when it should) | 100.0% | NFR-03 |
| False-answer rate (should decline, didn't) | 0.0% | NFR-03 |
| Overall accuracy | 90.0% | TO-7 |
| Error rate | 0.0% | NFR-07 |
| Average answer time | 25064 ms | NFR-01 |
| 95th percentile answer time | 53294 ms | NFR-01 |
| Convergence score (answer route) | 0.94 | — |

## By question type

| Family | Total | Correct | Accuracy |
|---|---|---|---|
| faq_exact | 11 | 10 | 90.9% |
| faq_paraphrased | 7 | 6 | 85.7% |
| unanswerable | 2 | 2 | 100.0% |

## By department (answerable questions)

| Department | Total | Correct | Accuracy |
|---|---|---|---|
| FAC | 3 | 3 | 100.0% |
| FIN | 2 | 2 | 100.0% |
| HR | 6 | 6 | 100.0% |
| IT | 2 | 2 | 100.0% |
| LEG | 4 | 2 | 50.0% |
| MKT | 1 | 1 | 100.0% |

## By language (NFR-05, bilingual quality)

| Language | Total | Correct | Accuracy |
|---|---|---|---|
| ar | 12 | 10 | 83.3% |
| en | 8 | 8 | 100.0% |

## Convergence (answer-agent efficiency)

How many LLM calls the answer agent actually needed versus the fewest it could ever need for a fresh question - the router's own fixed call is excluded, since it never varies. 1.0 means no wasted retries or judge passes; lower means the retry/judge machinery is triggering more than strictly necessary.

| Avg steps | Avg optimal steps | Convergence score | Sample size |
|---|---|---|---|
| 2.3 | 2 | 0.94 | 20 |

| Family | N | Avg steps | Avg optimal | Convergence |
|---|---|---|---|---|
| faq_exact | 11 | 2 | 2 | 1.0 |
| faq_paraphrased | 7 | 2 | 2 | 1.0 |
| unanswerable | 2 | 5 | 2 | 0.4 |

## Routes chosen

| Route | Count |
|---|---|
| answer | 20 |
