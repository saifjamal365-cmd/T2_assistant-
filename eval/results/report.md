# T2 Assistant - Evaluation Report

Dataset: 50 questions (39 answerable, 11 deliberately unanswerable).

## Headline numbers

| Metric | Value | Requirement |
|---|---|---|
| Retrieval recall (top-5) | 100.0% | NFR-02 |
| Answer accuracy (answerable) | 100.0% | TO-7 |
| Honesty rate (says "I don't know" when it should) | 90.9% | NFR-03 |
| False-answer rate (should decline, didn't) | 9.1% | NFR-03 |
| Overall accuracy | 98.0% | TO-7 |
| Error rate | 0.0% | NFR-07 |
| Average answer time | 11781 ms | NFR-01 |
| 95th percentile answer time | 25876 ms | NFR-01 |

## By question type

| Family | Total | Correct | Accuracy |
|---|---|---|---|
| faq_paraphrased | 20 | 20 | 100.0% |
| faq_exact | 19 | 19 | 100.0% |
| unanswerable | 11 | 10 | 90.9% |

## By department (answerable questions)

| Department | Total | Correct | Accuracy |
|---|---|---|---|
| FAC | 7 | 7 | 100.0% |
| FIN | 5 | 5 | 100.0% |
| HR | 12 | 12 | 100.0% |
| IT | 7 | 7 | 100.0% |
| LEG | 4 | 4 | 100.0% |
| MKT | 4 | 4 | 100.0% |

## By language (NFR-05, bilingual quality)

| Language | Total | Correct | Accuracy |
|---|---|---|---|
| ar | 25 | 24 | 96.0% |
| en | 25 | 25 | 100.0% |

## Routes chosen

| Route | Count |
|---|---|
| answer | 50 |
