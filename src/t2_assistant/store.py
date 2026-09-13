"""Saving conversations and run records.

Until now every request stood alone. This module keeps them in a small SQLite
file (settings.store_db) with three tables:

    conversations   one row per chat thread (id, title, timestamps)
    messages        every user and assistant turn, in order
    runs            one row per request: the route taken, timing, trace id

Plain `sqlite3` - no ORM. Each call opens its own short-lived connection, which
is simple and safe when FastAPI runs endpoints on different threads.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from t2_assistant.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id),
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    created_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_conversation ON messages(conversation_id, id);

CREATE TABLE IF NOT EXISTS runs (
    id                TEXT PRIMARY KEY,
    conversation_id   TEXT NOT NULL REFERENCES conversations(id),
    user_message      TEXT NOT NULL,
    route             TEXT NOT NULL,
    route_reason      TEXT NOT NULL,
    reply             TEXT NOT NULL,
    trace_id          TEXT,
    duration_ms       INTEGER NOT NULL,
    created_at        TEXT NOT NULL,
    feedback          TEXT,           -- 'helpful' | 'not_helpful' | NULL
    feedback_comment  TEXT
);
"""

_schema_ready = False


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _connect() -> sqlite3.Connection:
    global _schema_ready
    settings.store_db.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(settings.store_db)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    if not _schema_ready:
        connection.executescript(_SCHEMA)
        # add columns that older store.db files are missing
        for column in ("feedback TEXT", "feedback_comment TEXT"):
            try:
                connection.execute(f"ALTER TABLE runs ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass  # already there
        connection.commit()
        _schema_ready = True
    return connection


# ---- data returned to callers ---------------------------------------


@dataclass
class Message:
    role: str
    content: str
    created_at: str


@dataclass
class ConversationSummary:
    id: str
    title: str
    updated_at: str
    message_count: int


@dataclass
class Conversation:
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[Message]


@dataclass
class Run:
    id: str
    conversation_id: str
    user_message: str
    route: str
    route_reason: str
    reply: str
    trace_id: str | None
    duration_ms: int
    created_at: str
    feedback: str | None
    feedback_comment: str | None


# ---- conversations -------------------------------------------------


def create_conversation(title: str) -> str:
    """Start a new conversation and return its id."""
    conversation_id = uuid.uuid4().hex
    now = _now()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (conversation_id, title.strip() or "New conversation", now, now),
        )
    return conversation_id


def conversation_exists(conversation_id: str) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    return row is not None


def list_conversations(limit: int = 50) -> list[ConversationSummary]:
    """Most recently updated conversations first."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.updated_at, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        ConversationSummary(
            id=row["id"],
            title=row["title"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
        )
        for row in rows
    ]


def get_conversation(conversation_id: str) -> Conversation | None:
    with _connect() as connection:
        head = connection.execute(
            "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if head is None:
            return None
        rows = connection.execute(
            "SELECT role, content, created_at FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return Conversation(
        id=head["id"],
        title=head["title"],
        created_at=head["created_at"],
        updated_at=head["updated_at"],
        messages=[Message(r["role"], r["content"], r["created_at"]) for r in rows],
    )


def get_messages(conversation_id: str) -> list[Message]:
    conversation = get_conversation(conversation_id)
    return conversation.messages if conversation else []


# ---- writing a turn ----------------------------------------------


def add_message(conversation_id: str, role: str, content: str) -> None:
    with _connect() as connection:
        connection.execute(
            "INSERT INTO messages (conversation_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, _now()),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?",
            (_now(), conversation_id),
        )


def save_run(
    *,
    conversation_id: str,
    user_message: str,
    route: str,
    route_reason: str,
    reply: str,
    trace_id: str | None,
    duration_ms: int,
) -> str:
    """Record one request and return the run id."""
    run_id = uuid.uuid4().hex
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO runs
              (id, conversation_id, user_message, route, route_reason, reply,
               trace_id, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                conversation_id,
                user_message,
                route,
                route_reason,
                reply,
                trace_id,
                duration_ms,
                _now(),
            ),
        )
    return run_id


# ---- reading runs (observability) -------------------------------


def _row_to_run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        conversation_id=row["conversation_id"],
        user_message=row["user_message"],
        route=row["route"],
        route_reason=row["route_reason"],
        reply=row["reply"],
        trace_id=row["trace_id"],
        duration_ms=row["duration_ms"],
        created_at=row["created_at"],
        feedback=row["feedback"],
        feedback_comment=row["feedback_comment"],
    )


def list_runs(limit: int = 100) -> list[Run]:
    """Most recent runs first."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row_to_run(row) for row in rows]


def get_run(run_id: str) -> Run | None:
    with _connect() as connection:
        row = connection.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
    return _row_to_run(row) if row else None


def set_feedback(run_id: str, feedback: str, comment: str | None) -> bool:
    """Record a viewer's rating of a run. Returns False if the run is unknown."""
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE runs SET feedback = ?, feedback_comment = ? WHERE id = ?",
            (feedback, comment, run_id),
        )
    return cursor.rowcount > 0
