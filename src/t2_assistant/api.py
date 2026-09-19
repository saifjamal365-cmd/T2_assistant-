"""The HTTP API.

    GET  /                      -> the web chat page
    GET  /health                -> {"status": "ok"}
    POST /chat                  -> send a message, get the reply + route + run id
    GET    /conversations              -> list past conversations
    GET    /conversations/{id}         -> one conversation with all its messages
    PATCH  /conversations/{id}         -> rename a conversation
    PUT    /conversations/{id}/folder  -> move a conversation into a folder (or null to unfile it)
    DELETE /conversations/{id}         -> delete a conversation and its messages
    GET    /folders                    -> list folders
    POST   /folders                    -> create a folder
    PATCH  /folders/{id}               -> rename a folder
    DELETE /folders/{id}               -> delete a folder (its conversations are kept, just unfiled)
    GET    /runs                       -> recent requests, each with a link to its trace
    GET    /runs/{id}                  -> one request
    POST   /runs/{id}/feedback         -> rate a run (helpful / not) - also sent to MLflow
    GET    /kb/documents               -> browse the knowledge base (filter by department,
                                           language, topic, or a title search)
    GET    /kb/documents/{doc_id}      -> one document's full text and metadata
    POST   /auth/request-code          -> email a sign-in code to an allowed address
    POST   /auth/verify-code           -> check a code, sign in on success
    POST   /auth/logout                -> end the current session
    GET    /auth/me                    -> the signed-in email, or null

Every route other than the page itself, /health, and /auth/* requires a
signed-in session (a cookie set by /auth/verify-code) - see require_session
below.

The server owns the conversation history now: send a `conversation_id` to
continue a thread, or leave it out to start a new one.

Hardening (NFR-07): a bad message is rejected before it reaches the agent
(422); a failed call to the language model is caught and turned into a clear
503 instead of a raw crash; anything else unexpected still returns clean JSON
(500) rather than an unhandled-error page. Every failure is still traced -
the @mlflow.trace on chat() records it even when it raises.

Run it with:  python -m t2_assistant     (see __main__.py)
Web page:  http://localhost:8000/     Interactive API docs:  http://localhost:8000/docs
"""

from __future__ import annotations

import csv
import logging
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from groq import APIError
from pydantic import BaseModel, Field, field_validator

from t2_assistant import auth, store
from t2_assistant.config import settings
from t2_assistant.conversation import run_turn
from t2_assistant.observability import record_feedback, trace_url

logger = logging.getLogger("t2_assistant.api")

app = FastAPI(title="T2 Assistant", version="0.1.0")

_INDEX_HTML = Path(__file__).parent / "web" / "index.html"
_MAX_MESSAGE_LENGTH = 4000  # generous for a question or a page of meeting notes
_SESSION_COOKIE = "t2_session"
_PUBLIC_PATHS = {"/", "/health", "/docs", "/redoc", "/openapi.json"}


@app.middleware("http")
async def require_session(request: Request, call_next):  # type: ignore[no-untyped-def]
    """Every route needs a signed-in session, except the page itself, the
    health check, API docs, and the sign-in endpoints (which must be usable
    before signing in)."""
    path = request.url.path
    if path in _PUBLIC_PATHS or path.startswith("/auth/"):
        return await call_next(request)
    token = request.cookies.get(_SESSION_COOKIE)
    email = store.session_email(token) if token else None
    if email is None:
        return JSONResponse(status_code=401, content={"detail": "sign in required"})
    request.state.user_email = email
    return await call_next(request)


@app.exception_handler(Exception)
async def unhandled_error(_request: Request, exc: Exception) -> JSONResponse:
    """Anything not already handled: log it, and never leak a stack trace."""
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": f"Unexpected error: {exc}"[:300]})


class ChatIn(BaseModel):
    message: str = Field(description="The user's new message.")
    conversation_id: str | None = Field(
        default=None, description="Continue this conversation; omit to start a new one."
    )

    @field_validator("message")
    @classmethod
    def _not_blank_and_not_huge(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be empty")
        if len(stripped) > _MAX_MESSAGE_LENGTH:
            raise ValueError(f"message is too long (max {_MAX_MESSAGE_LENGTH} characters)")
        return stripped


class SourceOut(BaseModel):
    doc_id: str
    title: str
    department: str
    doc_type: str
    version: str
    text: str
    score: float
    highlights: list[str] = []


class ChatOut(BaseModel):
    conversation_id: str
    run_id: str
    reply: str
    route: str
    route_reason: str
    trace_url: str | None
    sources: list[SourceOut] = []


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationSummaryOut(BaseModel):
    id: str
    title: str
    folder_id: str | None
    updated_at: str
    message_count: int


class ConversationOut(BaseModel):
    id: str
    title: str
    folder_id: str | None
    created_at: str
    updated_at: str
    messages: list[MessageOut]


class RenameIn(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class SetFolderIn(BaseModel):
    folder_id: str | None = Field(
        description="Folder to move this conversation into, or null to unfile it."
    )


class FolderOut(BaseModel):
    id: str
    name: str
    created_at: str


class FolderIn(BaseModel):
    name: str = Field(min_length=1, max_length=100)


class RunOut(BaseModel):
    id: str
    conversation_id: str
    user_message: str
    route: str
    route_reason: str
    reply: str
    duration_ms: int
    created_at: str
    trace_url: str | None
    feedback: str | None
    feedback_comment: str | None


class FeedbackIn(BaseModel):
    helpful: bool = Field(description="Thumbs up (true) or thumbs down (false).")
    comment: str | None = Field(default=None, description="Optional note.")


class KbDocumentSummary(BaseModel):
    doc_id: str
    title: str
    department: str
    topic: str
    type: str
    language: str
    office: str
    version: str
    status: str
    effective_date: str


class KbDocumentOut(KbDocumentSummary):
    text: str


@lru_cache
def _kb_manifest() -> list[KbDocumentSummary]:
    """The manifest, read once and cached - it does not change while the
    server runs (rebuilding the knowledge base restarts the process)."""
    with (settings.knowledge_base_dir / "manifest.csv").open(encoding="utf-8") as handle:
        return [
            KbDocumentSummary(
                doc_id=row["doc_id"],
                title=row["title"],
                department=row["department"],
                topic=row["topic_key"],
                type=row["type"],
                language=row["language"],
                office=row["office"],
                version=row["version"],
                status=row["status"],
                effective_date=row["effective_date"],
            )
            for row in csv.DictReader(handle)
        ]


@lru_cache
def _kb_file_by_id() -> dict[str, str]:
    """doc_id -> its file path (relative to knowledge_base_dir), also cached."""
    with (settings.knowledge_base_dir / "manifest.csv").open(encoding="utf-8") as handle:
        return {row["doc_id"]: row["file"] for row in csv.DictReader(handle)}


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


class RequestCodeIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)


class VerifyCodeIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    code: str = Field(min_length=4, max_length=10)


class MeOut(BaseModel):
    email: str | None


@app.post("/auth/request-code")
def request_code(body: RequestCodeIn) -> dict[str, str]:
    try:
        auth.request_code(body.email)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="that email can't sign in here") from exc
    except httpx.HTTPError as exc:
        logger.warning("failed to send sign-in email: %s", exc)
        raise HTTPException(
            status_code=502, detail="couldn't send the email right now - please try again"
        ) from exc
    return {"status": "sent"}


@app.post("/auth/verify-code")
def verify_code(body: VerifyCodeIn, response: Response) -> dict[str, str]:
    token = auth.verify_code(body.email, body.code)
    if token is None:
        raise HTTPException(status_code=401, detail="that code is incorrect or has expired")
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_days * 24 * 3600,
        httponly=True,
        samesite="lax",
        # not marked secure: this app is served over plain http on localhost;
        # set secure=True once it's deployed behind https.
    )
    return {"status": "ok", "email": body.email.strip().lower()}


@app.post("/auth/logout")
def logout(request: Request, response: Response) -> dict[str, str]:
    token = request.cookies.get(_SESSION_COOKIE)
    if token:
        store.delete_session(token)
    response.delete_cookie(_SESSION_COOKIE)
    return {"status": "ok"}


@app.get("/auth/me", response_model=MeOut)
def get_me(request: Request) -> MeOut:
    token = request.cookies.get(_SESSION_COOKIE)
    return MeOut(email=store.session_email(token) if token else None)


@app.post("/chat", response_model=ChatOut)
def post_chat(body: ChatIn) -> ChatOut:
    try:
        result = run_turn(body.message, body.conversation_id)
    except APIError as exc:
        # the language model provider failed or is rate-limited - not our bug,
        # but the user still needs a clear answer instead of a crash (NFR-07)
        logger.warning("language model call failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="The assistant is temporarily unavailable. Please try again."
        ) from exc

    return ChatOut(
        conversation_id=result.conversation_id,
        run_id=result.run_id,
        reply=result.reply,
        route=result.route,
        route_reason=result.route_reason,
        trace_url=trace_url(result.trace_id),
        sources=[SourceOut.model_validate(s) for s in result.sources],
    )


@app.get("/conversations", response_model=list[ConversationSummaryOut])
def get_conversations() -> list[ConversationSummaryOut]:
    return [
        ConversationSummaryOut(
            id=c.id,
            title=c.title,
            folder_id=c.folder_id,
            updated_at=c.updated_at,
            message_count=c.message_count,
        )
        for c in store.list_conversations()
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str) -> ConversationOut:
    conversation = store.get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return ConversationOut(
        id=conversation.id,
        title=conversation.title,
        folder_id=conversation.folder_id,
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageOut(role=m.role, content=m.content, created_at=m.created_at)
            for m in conversation.messages
        ],
    )


@app.patch("/conversations/{conversation_id}", response_model=ConversationSummaryOut)
def rename_conversation(conversation_id: str, body: RenameIn) -> ConversationSummaryOut:
    if not store.rename_conversation(conversation_id, body.title):
        raise HTTPException(status_code=404, detail="conversation not found")
    updated = next(c for c in store.list_conversations() if c.id == conversation_id)
    return ConversationSummaryOut(
        id=updated.id,
        title=updated.title,
        folder_id=updated.folder_id,
        updated_at=updated.updated_at,
        message_count=updated.message_count,
    )


@app.put("/conversations/{conversation_id}/folder", response_model=ConversationSummaryOut)
def move_conversation(conversation_id: str, body: SetFolderIn) -> ConversationSummaryOut:
    if not store.set_conversation_folder(conversation_id, body.folder_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    updated = next(c for c in store.list_conversations() if c.id == conversation_id)
    return ConversationSummaryOut(
        id=updated.id,
        title=updated.title,
        folder_id=updated.folder_id,
        updated_at=updated.updated_at,
        message_count=updated.message_count,
    )


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict[str, str]:
    if not store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"status": "deleted"}


@app.get("/folders", response_model=list[FolderOut])
def get_folders() -> list[FolderOut]:
    return [FolderOut(id=f.id, name=f.name, created_at=f.created_at) for f in store.list_folders()]


@app.post("/folders", response_model=FolderOut)
def post_folder(body: FolderIn) -> FolderOut:
    folder_id = store.create_folder(body.name)
    created = next(f for f in store.list_folders() if f.id == folder_id)
    return FolderOut(id=created.id, name=created.name, created_at=created.created_at)


@app.patch("/folders/{folder_id}", response_model=FolderOut)
def patch_folder(folder_id: str, body: FolderIn) -> FolderOut:
    if not store.rename_folder(folder_id, body.name):
        raise HTTPException(status_code=404, detail="folder not found")
    updated = next(f for f in store.list_folders() if f.id == folder_id)
    return FolderOut(id=updated.id, name=updated.name, created_at=updated.created_at)


@app.delete("/folders/{folder_id}")
def remove_folder(folder_id: str) -> dict[str, str]:
    if not store.delete_folder(folder_id):
        raise HTTPException(status_code=404, detail="folder not found")
    return {"status": "deleted"}


def _run_out(run: store.Run) -> RunOut:
    return RunOut(
        id=run.id,
        conversation_id=run.conversation_id,
        user_message=run.user_message,
        route=run.route,
        route_reason=run.route_reason,
        reply=run.reply,
        duration_ms=run.duration_ms,
        created_at=run.created_at,
        trace_url=trace_url(run.trace_id),
        feedback=run.feedback,
        feedback_comment=run.feedback_comment,
    )


@app.get("/runs", response_model=list[RunOut])
def get_runs() -> list[RunOut]:
    return [_run_out(run) for run in store.list_runs()]


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str) -> RunOut:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_out(run)


@app.post("/runs/{run_id}/feedback")
def post_feedback(run_id: str, body: FeedbackIn) -> dict[str, str]:
    if not record_feedback(run_id, body.helpful, body.comment):
        raise HTTPException(status_code=404, detail="run not found")
    return {"status": "recorded"}


@app.get("/kb/documents", response_model=list[KbDocumentSummary])
def list_kb_documents(
    department: str | None = None,
    language: str | None = None,
    topic: str | None = None,
    q: str | None = None,
) -> list[KbDocumentSummary]:
    docs = _kb_manifest()
    if department:
        docs = [d for d in docs if d.department == department]
    if language:
        docs = [d for d in docs if d.language == language]
    if topic:
        docs = [d for d in docs if d.topic == topic]
    if q:
        needle = q.strip().lower()
        docs = [d for d in docs if needle in d.title.lower()]
    return docs


@app.get("/kb/documents/{doc_id}", response_model=KbDocumentOut)
def get_kb_document(doc_id: str) -> KbDocumentOut:
    summary = next((d for d in _kb_manifest() if d.doc_id == doc_id), None)
    file_path = _kb_file_by_id().get(doc_id)
    if summary is None or file_path is None:
        raise HTTPException(status_code=404, detail="document not found")
    text = (settings.knowledge_base_dir / file_path).read_text(encoding="utf-8")
    return KbDocumentOut(**summary.model_dump(), text=text)
