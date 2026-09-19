# T2 Assistant

An agentic AI knowledge assistant for company policy questions and meeting-note
summarisation, in Arabic and English.

This is a clean rebuild of the assistant. An earlier prototype was built and
reviewed; its approach is reference only. Full context, requirements and the
phase-by-phase log are in **`T2_Assistant_Project_Documentation.docx`**.

## Status

| Phase | Focus | State |
|---|---|---|
| 0 | Project setup and planning | done |
| 1 | Requirements and architecture | done (Sections 7 and 9) |
| 2 | Data and knowledge base | done — 2000 documents (1000 EN + 1000 AR) + manifest |
| 3 | Agentic core (router + specialist agents) | done — router, greeting, clarify, summarise, API, tracing, tests |
| 4 | Retrieval and the answer agent | done — index build, Chroma search, grounded sourced answers |
| 5 | Persistence and memory | done — SQLite store, conversations, follow-up questions |
| 6 | Observability | done — run records linked to traces, 👍/👎 feedback logged to MLflow |
| 7 | Interface | done — web chat page at `/` (bilingual, conversation sidebar, feedback) |
| 8 | Evaluation and hardening | done — 500-question fair eval set, scored 50-question run (98% overall accuracy), API hardening |

## Folder structure

```
t2-assistant/
├── T2_Assistant_Project_Documentation.docx   the project documentation (living)
├── pyproject.toml            project metadata + pinned dependencies
├── .env.example              template for secrets (copy to .env)
├── scripts/
│   ├── generate_kb.py         builds the synthetic knowledge base
│   ├── build_eval_set.py      builds eval/dataset.json (fair questions, sized to the KB)
│   └── build_eval_subset.py   picks a smaller, still-fair slice for a quick run
├── data/
│   └── knowledge_base/       2000 generated policy documents + manifest.csv
│       ├── en/  ar/          1000 documents each
│       └── manifest.csv      one row per document (id, title, dept, type, ...)
├── eval/
│   ├── dataset.json           the full evaluation set (grows with the KB)
│   ├── dataset_subset.json    a fair 50-question sample (free-tier daily cap)
│   └── results/                report.md, report.json, answers.jsonl
├── src/t2_assistant/
│   ├── config.py             settings loaded from .env
│   ├── tracing.py            switches MLflow tracing on
│   ├── store.py              conversations, messages, run records (SQLite)
│   ├── conversation.py       runs one chat turn with memory (loads/saves history)
│   ├── observability.py      trace links + 👍/👎 feedback (also sent to MLflow)
│   ├── api.py                the HTTP API + serves the web page at / (input validation, clean error responses)
│   ├── web/index.html        the chat page (one file, no build step)
│   ├── agents/
│   │   ├── router.py         picks the route with the LLM (no keyword list)
│   │   ├── graph.py          wires router -> specialist
│   │   ├── greeting.py  clarify.py  summarise.py   working specialists
│   │   └── answer.py         retrieves passages, writes a sourced answer
│   ├── knowledge/
│   │   ├── embeddings.py     text -> vectors (multilingual E5, local)
│   │   ├── chunking.py       split a document into overlapping passages
│   │   ├── vector_store.py   opens / creates the Chroma index
│   │   ├── build_index.py    rebuild the index from the knowledge base
│   │   └── search.py         question -> closest passages
│   └── evaluation/
│       ├── dataset.py         loads an eval dataset json file
│       ├── retrieval.py       retrieval-only pass (no LLM cost)
│       ├── quality.py         full-pipeline pass, calls the real assistant
│       ├── report.py          scores both passes into report.md / report.json
│       └── run.py             CLI: python -m t2_assistant.evaluation.run
└── tests/                    test_graph.py, test_api.py, test_store.py, test_evaluation.py, ...
```

The API keeps the conversation history: `POST /chat` with a `conversation_id`
continues a thread (leave it out to start one); `GET /conversations` lists them
and `GET /conversations/{id}` returns one with all its messages. `GET /runs`
lists past requests each with a link to its MLflow trace, and
`POST /runs/{id}/feedback` records a 👍/👎 (also attached to the trace).
`GET /kb/documents` browses the knowledge base itself (filter by `department`,
`language`, `topic`, or a title search with `q`), and
`GET /kb/documents/{doc_id}` returns one document's full text - the web page's
"Knowledge Base" tab is built on these two.

## Run it

```bash
python -m venv .venv
.venv\Scripts\pip install -e ".[dev]" --only-binary=:all:   # Windows; --only-binary needed for chromadb
copy .env.example .env                          # then paste your Groq API key
python -m t2_assistant.knowledge.build_index    # build the search index (downloads the model once)
python -m t2_assistant                          # chat page at http://127.0.0.1:8000/  (API docs at /docs)
mlflow ui --backend-store-uri sqlite:///mlflow.db   # traces at http://127.0.0.1:5000
```

Run every command from the project folder. Checks: `ruff check src tests` ·
`mypy src tests` · `pytest -m "not integration"` (add `-m integration` to test
against the real index and Groq).

## The knowledge base

`data/knowledge_base/` holds 2000 synthetic company-policy documents. They are
**not real** — they are generated to give the retrieval layer a realistic
scale and shape to work against, and are replaced by real documents when those
become available (no code change).

- **114 topics** across 8 departments (HR, IT, Finance, Facilities,
  Legal & Compliance, Operations, Marketing & Communications, Procurement).
- **4 document types** per topic: Policy, Procedure, FAQ, Quick Reference.
- **English and Arabic** (1000 each).
- **Per-office variants** — some policies differ by office (Riyadh, Jeddah,
  Dubai, Cairo, Remote), with different numbers and scope.
- **Old versions** — a number of documents are marked `Superseded` and point
  to the current version, so retrieval has to prefer the live one.
- Every document has a header: document ID, department, type, version,
  effective date, status, owner, who it applies to, and related documents.

### Rebuild it

```bash
python scripts/generate_kb.py                 # 2000 documents (default)
python scripts/generate_kb.py --count 300     # smaller set for quick tests
python scripts/generate_kb.py --out data/kb2  # write elsewhere
```

The generator is seeded, so it produces the same corpus every time. It needs
only the Python standard library.

## Evaluation

`eval/dataset.json` is built straight from the same generator that built the
knowledge base (never hand-written), so every expected answer traces back to
a real document - answerable questions (exact wording + paraphrased) and a
block of deliberately unanswerable ones, split evenly EN/AR. Its size grows
with the knowledge base (run `python scripts/build_eval_set.py` after
regenerating the KB to refresh it).

```bash
python -m t2_assistant.evaluation.run                              # the full set
python -m t2_assistant.evaluation.run --dataset eval/dataset_subset.json  # fair 50-question sample
python -m t2_assistant.evaluation.run --resume                     # keep successes, retry only failures
```

Groq's free plan caps tokens per day as well as per minute; `--resume` picks
up a run that was cut short by the daily cap without re-asking questions it
already got right. Results (`report.md`, `report.json`, `answers.jsonl`) land
in `eval/results/` and are also logged as an MLflow run named "evaluation".
See Section 12 and 13.8 of the documentation for the current results.

## Stack

Python 3.12 · LangGraph — a small team of agents: a **router agent** that picks
a route by reasoning (no keyword list) and specialist agents for **greeting**,
**answer**, **summary** and **clarify** · Groq LLM · multilingual‑E5 embeddings
(local; BGE‑M3 is higher quality but needs a GPU) · Chroma vector store (local) ·
MLflow tracing (local) · SQLite · FastAPI. See Section 9 of the documentation for
the reasoning.
