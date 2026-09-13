"""T2 Assistant — an agentic AI knowledge assistant.

The package is organised one module per job:

    config.py        settings loaded from .env
    tracing.py       MLflow tracing, set up once
    agents/          the team of agents (router + specialists)
    knowledge/       the knowledge base: build the index, search it
    store.py         conversations and run records in SQLite
    conversation.py  run one chat turn, with memory (loads / saves history)
    observability.py trace links and thumbs-up / thumbs-down feedback
    api.py           the HTTP API + serves the web page at /
"""

__version__ = "0.1.0"
