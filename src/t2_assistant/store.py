"""Saving conversations and run records.

Until now every request stood alone. This module keeps them in a small SQLite
file (settings.store_db) with five tables:

    conversations   one row per chat thread (id, title, folder, timestamps)
    folders         one row per user-created folder, to group conversations
    messages        every user and assistant turn, in order
    runs            one row per request: the route taken, timing, trace id
    sources         the exact passages a run's answer cited (if any), so a
                     source in the reply can be clicked to show the real text
                     behind it, not just the document id

Plain `sqlite3` - no ORM. Each call opens its own short-lived connection, which
is simple and safe when FastAPI runs endpoints on different threads.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from t2_assistant.config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS folders (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL,
    folder_id   TEXT REFERENCES folders(id),
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

-- The exact passages an answer cited, so the app can show the user precisely
-- what grounded it (not just the document id) when they click a source.
CREATE TABLE IF NOT EXISTS sources (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL REFERENCES runs(id),
    doc_id        TEXT NOT NULL,
    title         TEXT NOT NULL,
    department    TEXT NOT NULL,
    doc_type      TEXT NOT NULL,
    version       TEXT NOT NULL,
    passage_text  TEXT NOT NULL,
    score         REAL NOT NULL,
    highlights    TEXT NOT NULL DEFAULT '[]'  -- JSON list of sentences to highlight
);
CREATE INDEX IF NOT EXISTS ix_sources_run ON sources(run_id);
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
        for table, column in (
            ("runs", "feedback TEXT"),
            ("runs", "feedback_comment TEXT"),
            ("sources", "highlights TEXT NOT NULL DEFAULT '[]'"),
            ("conversations", "folder_id TEXT REFERENCES folders(id)"),
        ):
            try:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass  # already there
        # only safe to create now: folder_id is guaranteed to exist by this point,
        # whether the table was just created above or just migrated
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_conversations_folder ON conversations(folder_id)"
        )
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
class Folder:
    id: str
    name: str
    created_at: str


@dataclass
class ConversationSummary:
    id: str
    title: str
    folder_id: str | None
    updated_at: str
    message_count: int


@dataclass
class Conversation:
    id: str
    title: str
    folder_id: str | None
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


@dataclass
class Source:
    doc_id: str
    title: str
    department: str
    doc_type: str
    version: str
    text: str
    score: float
    highlights: list[str]


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
            SELECT c.id, c.title, c.folder_id, c.updated_at, COUNT(m.id) AS message_count
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
            folder_id=row["folder_id"],
            updated_at=row["updated_at"],
            message_count=row["message_count"],
        )
        for row in rows
    ]


def get_conversation(conversation_id: str) -> Conversation | None:
    with _connect() as connection:
        head = connection.execute(
            "SELECT id, title, folder_id, created_at, updated_at FROM conversations WHERE id = ?",
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
        folder_id=head["folder_id"],
        created_at=head["created_at"],
        updated_at=head["updated_at"],
        messages=[Message(r["role"], r["content"], r["created_at"]) for r in rows],
    )


def get_messages(conversation_id: str) -> list[Message]:
    conversation = get_conversation(conversation_id)
    return conversation.messages if conversation else []


def rename_conversation(conversation_id: str, title: str) -> bool:
    title = title.strip()
    if not title:
        return False
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id)
        )
    return cursor.rowcount > 0


def set_conversation_folder(conversation_id: str, folder_id: str | None) -> bool:
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE conversations SET folder_id = ? WHERE id = ?", (folder_id, conversation_id)
        )
    return cursor.rowcount > 0


def delete_conversation(conversation_id: str) -> bool:
    """Delete a conversation completely: its messages, its run records, and
    those runs' sources - deleting a chat means it's actually gone, not
    lingering in the observability views the user doesn't see."""
    with _connect() as connection:
        run_ids = [
            row["id"]
            for row in connection.execute(
                "SELECT id FROM runs WHERE conversation_id = ?", (conversation_id,)
            ).fetchall()
        ]
        for run_id in run_ids:
            connection.execute("DELETE FROM sources WHERE run_id = ?", (run_id,))
        connection.execute("DELETE FROM runs WHERE conversation_id = ?", (conversation_id,))
        connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cursor = connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
    return cursor.rowcount > 0


# ---- folders --------------------------------------------------------


def create_folder(name: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("folder name cannot be empty")
    folder_id = uuid.uuid4().hex
    with _connect() as connection:
        connection.execute(
            "INSERT INTO folders (id, name, created_at) VALUES (?, ?, ?)",
            (folder_id, name, _now()),
        )
    return folder_id


def list_folders() -> list[Folder]:
    with _connect() as connection:
        rows = connection.execute("SELECT id, name, created_at FROM folders ORDER BY name").fetchall()
    return [Folder(id=r["id"], name=r["name"], created_at=r["created_at"]) for r in rows]


def rename_folder(folder_id: str, name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    with _connect() as connection:
        cursor = connection.execute("UPDATE folders SET name = ? WHERE id = ?", (name, folder_id))
    return cursor.rowcount > 0


def delete_folder(folder_id: str) -> bool:
    """Delete a folder. Its conversations are kept, just un-filed - deleting
    a folder should never delete chat history."""
    with _connect() as connection:
        connection.execute(
            "UPDATE conversations SET folder_id = NULL WHERE folder_id = ?", (folder_id,)
        )
        cursor = connection.execute("DELETE FROM folders WHERE id = ?", (folder_id,))
    return cursor.rowcount > 0


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


def save_sources(run_id: str, passages: list[dict]) -> None:
    """Record the exact passages a run's answer cited (skipped entirely for a
    decline or a non-answer route, since `passages` is empty for those)."""
    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO sources
              (run_id, doc_id, title, department, doc_type, version, passage_text, score, highlights)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    p["doc_id"],
                    p["title"],
                    p["department"],
                    p["doc_type"],
                    p["version"],
                    p["text"],
                    p["score"],
                    json.dumps(p.get("highlights", [])),
                )
                for p in passages
            ],
        )


def get_sources(run_id: str) -> list[Source]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT doc_id, title, department, doc_type, version, passage_text, score, highlights "
            "FROM sources WHERE run_id = ? ORDER BY id",
            (run_id,),
        ).fetchall()
    return [
        Source(
            doc_id=r["doc_id"],
            title=r["title"],
            department=r["department"],
            doc_type=r["doc_type"],
            version=r["version"],
            text=r["passage_text"],
            score=r["score"],
            highlights=json.loads(r["highlights"]),
        )
        for r in rows
    ]


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
