"""Saving conversations and run records.

Until now every request stood alone. This module keeps them in a small SQLite
file (settings.store_db) with seven tables:

    conversations   one row per chat thread (id, title, owner, folder, timestamps)
    folders         one row per user-created folder, to group conversations
    messages        every user and assistant turn, in order
    runs            one row per request: the route taken, timing, trace id
    sources         the exact passages a run's answer cited (if any), so a
                     source in the reply can be clicked to show the real text
                     behind it, not just the document id
    users           a registered email + its (hashed) password
    sessions        a signed-in session (email -> session token), created
                     once a password is verified

conversations and folders each carry a `user_email` owner - every function
that reads or changes one takes the caller's email and only ever touches
rows that belong to them; a non-owner gets exactly the same `None`/`False`
a missing id would. `runs` (and `sources`, which hangs off a run) carry no
email of their own - ownership is derived by joining back to the run's
conversation, since a run can never outlive the conversation it belongs to
(see delete_conversation).

Plain `sqlite3` - no ORM. Each call opens its own short-lived connection, which
is simple and safe when FastAPI runs endpoints on different threads.
"""

from __future__ import annotations

import json
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from t2_assistant.config import settings

# Backfill target for conversations/folders that predate per-user ownership -
# a real, already-allowed test account (AUTH_ALLOWED_TEST_EMAILS).
_LEGACY_OWNER_EMAIL = "saifjamal365@gmail.com"

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

-- A registered sign-in: email + password hash. `password_hash` is a salted
-- hash (see auth.py), never the raw password.
CREATE TABLE IF NOT EXISTS users (
    email          TEXT PRIMARY KEY,
    password_hash  TEXT NOT NULL,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    token       TEXT PRIMARY KEY,
    email       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    expires_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sessions_expires ON sessions(expires_at);
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
            ("runs", f"model TEXT NOT NULL DEFAULT '{settings.llm_model}'"),
            ("conversations", f"user_email TEXT NOT NULL DEFAULT '{_LEGACY_OWNER_EMAIL}'"),
            ("folders", f"user_email TEXT NOT NULL DEFAULT '{_LEGACY_OWNER_EMAIL}'"),
        ):
            try:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {column}")
            except sqlite3.OperationalError:
                pass  # already there
        # only safe to create now: folder_id/user_email are guaranteed to exist
        # by this point, whether the table was just created above or migrated
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_conversations_folder ON conversations(folder_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_conversations_user_email ON conversations(user_email)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS ix_folders_user_email ON folders(user_email)"
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
    model: str | None


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


def create_conversation(title: str, user_email: str) -> str:
    """Start a new conversation, owned by `user_email`, and return its id."""
    conversation_id = uuid.uuid4().hex
    now = _now()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, user_email) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, title.strip() or "New conversation", now, now, user_email),
        )
    return conversation_id


def conversation_exists(conversation_id: str, user_email: str) -> bool:
    with _connect() as connection:
        row = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_email = ?",
            (conversation_id, user_email),
        ).fetchone()
    return row is not None


def list_conversations(user_email: str, limit: int = 50) -> list[ConversationSummary]:
    """This user's conversations, most recently updated first."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT c.id, c.title, c.folder_id, c.updated_at, COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.user_email = ?
            GROUP BY c.id
            ORDER BY c.updated_at DESC, c.rowid DESC
            LIMIT ?
            """,
            (user_email, limit),
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


def get_conversation(conversation_id: str, user_email: str) -> Conversation | None:
    with _connect() as connection:
        head = connection.execute(
            "SELECT id, title, folder_id, created_at, updated_at FROM conversations "
            "WHERE id = ? AND user_email = ?",
            (conversation_id, user_email),
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


def get_messages(conversation_id: str, user_email: str) -> list[Message]:
    conversation = get_conversation(conversation_id, user_email)
    return conversation.messages if conversation else []


def rename_conversation(conversation_id: str, title: str, user_email: str) -> bool:
    title = title.strip()
    if not title:
        return False
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ? AND user_email = ?",
            (title, conversation_id, user_email),
        )
    return cursor.rowcount > 0


def set_conversation_folder(conversation_id: str, folder_id: str | None, user_email: str) -> bool:
    """Move a conversation into `folder_id` (or unfile it, if None). Fails if
    the conversation isn't the caller's, or the target folder isn't either -
    otherwise a user could file their own conversation into someone else's
    folder."""
    with _connect() as connection:
        if folder_id is not None:
            owned_folder = connection.execute(
                "SELECT 1 FROM folders WHERE id = ? AND user_email = ?", (folder_id, user_email)
            ).fetchone()
            if owned_folder is None:
                return False
        cursor = connection.execute(
            "UPDATE conversations SET folder_id = ? WHERE id = ? AND user_email = ?",
            (folder_id, conversation_id, user_email),
        )
    return cursor.rowcount > 0


def delete_conversation(conversation_id: str, user_email: str) -> bool:
    """Delete a conversation completely: its messages, its run records, and
    those runs' sources - deleting a chat means it's actually gone, not
    lingering in the observability views the user doesn't see.

    Ownership is checked up front, before any deletion - the cascade below
    looks runs up by conversation_id alone, so filtering only the final
    DELETE would still let a non-owner trigger it against someone else's
    runs."""
    with _connect() as connection:
        owned = connection.execute(
            "SELECT 1 FROM conversations WHERE id = ? AND user_email = ?",
            (conversation_id, user_email),
        ).fetchone()
        if owned is None:
            return False
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


def create_folder(name: str, user_email: str) -> str:
    name = name.strip()
    if not name:
        raise ValueError("folder name cannot be empty")
    folder_id = uuid.uuid4().hex
    with _connect() as connection:
        connection.execute(
            "INSERT INTO folders (id, name, created_at, user_email) VALUES (?, ?, ?, ?)",
            (folder_id, name, _now(), user_email),
        )
    return folder_id


def list_folders(user_email: str) -> list[Folder]:
    with _connect() as connection:
        rows = connection.execute(
            "SELECT id, name, created_at FROM folders WHERE user_email = ? ORDER BY name",
            (user_email,),
        ).fetchall()
    return [Folder(id=r["id"], name=r["name"], created_at=r["created_at"]) for r in rows]


def rename_folder(folder_id: str, name: str, user_email: str) -> bool:
    name = name.strip()
    if not name:
        return False
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE folders SET name = ? WHERE id = ? AND user_email = ?",
            (name, folder_id, user_email),
        )
    return cursor.rowcount > 0


def delete_folder(folder_id: str, user_email: str) -> bool:
    """Delete a folder. Its conversations are kept, just un-filed - deleting
    a folder should never delete chat history. Ownership checked up front,
    same reasoning as delete_conversation: the unfiling UPDATE below matches
    by folder_id alone, so it must never run against a folder that isn't
    the caller's."""
    with _connect() as connection:
        owned = connection.execute(
            "SELECT 1 FROM folders WHERE id = ? AND user_email = ?", (folder_id, user_email)
        ).fetchone()
        if owned is None:
            return False
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
    model: str,
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
               model, trace_id, duration_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                conversation_id,
                user_message,
                route,
                route_reason,
                reply,
                model,
                trace_id,
                duration_ms,
                _now(),
            ),
        )
    return run_id


def save_sources(run_id: str, passages: list[dict[str, object]]) -> None:
    """Record the exact passages a run's answer cited (skipped entirely for a
    decline or a non-answer route, since `passages` is empty for those)."""
    with _connect() as connection:
        connection.executemany(
            """
            INSERT INTO sources
              (run_id, doc_id, title, department, doc_type, version, passage_text, score,
               highlights)
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
        model=row["model"],
    )


def list_runs(user_email: str, limit: int = 100) -> list[Run]:
    """This user's runs, most recent first - ownership comes from the
    conversation a run belongs to (runs carry no email of their own)."""
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT runs.* FROM runs
            JOIN conversations ON conversations.id = runs.conversation_id
            WHERE conversations.user_email = ?
            ORDER BY runs.created_at DESC, runs.rowid DESC
            LIMIT ?
            """,
            (user_email, limit),
        ).fetchall()
    return [_row_to_run(row) for row in rows]


def get_run(run_id: str, user_email: str) -> Run | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT runs.* FROM runs
            JOIN conversations ON conversations.id = runs.conversation_id
            WHERE runs.id = ? AND conversations.user_email = ?
            """,
            (run_id, user_email),
        ).fetchone()
    return _row_to_run(row) if row else None


def set_feedback(run_id: str, feedback: str, comment: str | None) -> bool:
    """Record a viewer's rating of a run. Returns False if the run is unknown."""
    with _connect() as connection:
        cursor = connection.execute(
            "UPDATE runs SET feedback = ?, feedback_comment = ? WHERE id = ?",
            (feedback, comment, run_id),
        )
    return cursor.rowcount > 0


# ---- sign-in: accounts and sessions ----------------------------------


def get_password_hash(email: str) -> str | None:
    """The stored password hash for this email, or None if no account exists.
    `store` treats the hash as an opaque string - hashing and verifying it is
    auth.py's job, not this module's."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT password_hash FROM users WHERE email = ?", (email,)
        ).fetchone()
    return str(row["password_hash"]) if row else None


def create_user(email: str, password_hash: str) -> None:
    """Register a new account. Raises sqlite3.IntegrityError if the email is
    already registered - callers check get_password_hash() first."""
    with _connect() as connection:
        connection.execute(
            "INSERT INTO users (email, password_hash, created_at) VALUES (?, ?, ?)",
            (email, password_hash, _now()),
        )


def create_session(email: str) -> str:
    token = secrets.token_urlsafe(32)
    expires_at = (datetime.now(UTC) + timedelta(days=settings.session_ttl_days)).isoformat(
        timespec="milliseconds"
    )
    with _connect() as connection:
        connection.execute(
            "INSERT INTO sessions (token, email, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, email, _now(), expires_at),
        )
    return token


def session_email(token: str) -> str | None:
    """The signed-in email for this session token, or None if it's missing,
    unknown, or expired."""
    with _connect() as connection:
        row = connection.execute(
            "SELECT email, expires_at FROM sessions WHERE token = ?", (token,)
        ).fetchone()
    if row is None or row["expires_at"] < _now():
        return None
    return str(row["email"])


def delete_session(token: str) -> None:
    with _connect() as connection:
        connection.execute("DELETE FROM sessions WHERE token = ?", (token,))
