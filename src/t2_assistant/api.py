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
    POST   /auth/login                 -> sign in with email + password (registers
                                           a new allowed email on first use)
    POST   /auth/logout                -> end the current session
    GET    /auth/me                    -> the signed-in email, or null
    GET    /voice                      -> the voice page (speak a question, hear the answer)
    POST   /voice/turn                 -> a recording (raw audio body), answered as a
                                           stream: what was heard, the reply, then
                                           each spoken sentence with its audio
    POST   /voice/ask                  -> the same for a typed question

Every route other than the two pages, /health, and /auth/* requires a
signed-in session (a cookie set by /auth/login) - see require_session below.

The server owns the conversation history now: send a `conversation_id` to
continue a thread, or leave it out to start a new one.

Hardening (NFR-07): a bad message is rejected before it reaches the agent
(422); a failed call to the language model is caught and turned into a clear
503 instead of a raw crash; anything else unexpected still returns clean JSON
(500) rather than an unhandled-error page. Every failure is still traced -
the @mlflow.trace on chat() records it even when it raises.

A voice turn is one MLflow trace named `voice_turn`, whichever of the two
voice routes started it: hearing the question, the same chat turn as /chat,
the rewrite into dialect, and reading each sentence aloud are steps inside it
(see _voice_turn below).

Run it with:  python -m t2_assistant     (see __main__.py)
Web page:  http://localhost:8000/     Interactive API docs:  http://localhost:8000/docs
"""

from __future__ import annotations

import asyncio
import base64
import csv
import json
import logging
from collections.abc import AsyncIterator
from functools import lru_cache
from pathlib import Path

import mlflow
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from groq import APIError
from mlflow.entities import SpanType
from pydantic import BaseModel, Field, field_validator

from t2_assistant import auth, store, voice
from t2_assistant.agents.llm import ModelId
from t2_assistant.config import settings
from t2_assistant.conversation import TurnResult, run_turn
from t2_assistant.observability import record_feedback, trace_url

logger = logging.getLogger("t2_assistant.api")

app = FastAPI(title="T2 Assistant", version="0.1.0")

_INDEX_HTML = Path(__file__).parent / "web" / "index.html"
_VOICE_HTML = Path(__file__).parent / "web" / "voice.html"
_MAX_MESSAGE_LENGTH = 4000  # generous for a question or a page of meeting notes
_MAX_AUDIO_BYTES = 10 * 1024 * 1024  # minutes of compressed speech; a question is seconds
# what the browser recorded -> the file extension Groq uses to recognise the format
_AUDIO_EXTENSIONS = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/mpeg": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
}
_SESSION_COOKIE = "t2_session"
_PUBLIC_PATHS = {"/", "/voice", "/health", "/docs", "/redoc", "/openapi.json"}


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
    model: ModelId | None = Field(
        default=None, description="Groq model id for this turn; omit to use the server default."
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
    model: str


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
    model: str | None
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


class LoginIn(BaseModel):
    email: str = Field(min_length=3, max_length=200)
    password: str = Field(min_length=8, max_length=200)


class LoginOut(BaseModel):
    status: str
    email: str
    created: bool = Field(description="True if this call just registered the account.")


class MeOut(BaseModel):
    email: str | None


@app.post("/auth/login", response_model=LoginOut)
def login(body: LoginIn, response: Response) -> LoginOut:
    try:
        result = auth.register_or_sign_in(body.email, body.password)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail="that email can't sign in here") from exc
    if result is None:
        raise HTTPException(status_code=401, detail="incorrect password")
    token, created = result
    response.set_cookie(
        _SESSION_COOKIE,
        token,
        max_age=settings.session_ttl_days * 24 * 3600,
        httponly=True,
        samesite="lax",
        # not marked secure: this app is served over plain http on localhost;
        # set secure=True once it's deployed behind https.
    )
    return LoginOut(status="ok", email=body.email.strip().lower(), created=created)


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
def post_chat(body: ChatIn, request: Request) -> ChatOut:
    try:
        result = run_turn(
            body.message,
            body.conversation_id,
            model=body.model,
            user_email=request.state.user_email,
        )
    except APIError as exc:
        # the language model provider failed or is rate-limited - not our bug,
        # but the user still needs a clear answer instead of a crash (NFR-07)
        logger.warning("language model call failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="The assistant is temporarily unavailable. Please try again."
        ) from exc

    return _chat_out(result)


def _chat_out(result: TurnResult) -> ChatOut:
    return ChatOut(
        conversation_id=result.conversation_id,
        run_id=result.run_id,
        reply=result.reply,
        route=result.route,
        route_reason=result.route_reason,
        trace_url=trace_url(result.trace_id),
        sources=[SourceOut.model_validate(s) for s in result.sources],
        model=result.model,
    )


class VoiceAskOut(ChatOut):
    spoken: str = Field(description="The reply as Saudi-dialect text to read aloud.")


@app.get("/voice", response_class=HTMLResponse)
def voice_page() -> HTMLResponse:
    return HTMLResponse(_VOICE_HTML.read_text(encoding="utf-8"))


# The events of one voice turn, in order; None marks the end.
_VoiceEvents = asyncio.Queue[dict[str, object] | None]


async def _voice_turn(
    events: _VoiceEvents,
    *,
    user_email: str,
    conversation_id: str | None,
    model: str | None,
    message: str | None = None,
    recording: tuple[bytes, str] | None = None,
) -> None:
    """One whole voice turn, put on `events` piece by piece as it is produced:
    what was heard (only when a `recording` - audio and its file name - was
    sent instead of a typed `message`), the answer, each spoken sentence.

    The turn is one MLflow trace, `voice_turn`. Everything below runs inside
    that span, so each step files itself under it: `speech_to_text`, `chat`
    (the very same turn /chat runs, with its router, searches and model
    calls), `voice_dialect`, and a `text_to_speech` per sentence. The trace is
    tagged `channel: voice` so voice turns can be told from typed chat.

    It runs as a task of its own rather than inside the response stream, so
    the span is opened and closed by one task: a stream can be dropped by the
    client at any point, and a span left open across that would be closed from
    somewhere else, or not at all.

    Only the written reply is stored in the conversation - the spoken text is
    a rendering of it, not a second answer.
    """
    outputs: dict[str, object] = {}
    interrupted = False
    with mlflow.start_span(name="voice_turn", span_type=SpanType.AGENT) as turn:
        # the recording itself is never put in a trace - only its size
        turn.set_inputs(
            {"audio_bytes": len(recording[0]), "conversation_id": conversation_id}
            if recording
            else {"message": message, "conversation_id": conversation_id}
        )
        mlflow.update_current_trace(
            tags={"channel": "voice", "voice.input": "speech" if recording else "text"},
            metadata={"mlflow.trace.user": user_email},
        )
        try:
            if recording:
                message = await run_in_threadpool(voice.transcribe, *recording)
                outputs["heard"] = message
                events.put_nowait({"type": "heard", "text": message})
            if message:  # nothing understood in the recording = nothing to answer
                result = await run_in_threadpool(
                    run_turn,
                    message,
                    conversation_id,
                    model=model,
                    user_email=user_email,
                    highlights=False,  # the voice page lists sources by name only
                )
                spoken = await run_in_threadpool(voice.to_spoken, result.reply, result.model)
                answer = VoiceAskOut(**_chat_out(result).model_dump(), spoken=spoken)
                outputs.update(route=result.route, reply=result.reply, spoken=spoken)
                events.put_nowait({"type": "answer", **answer.model_dump()})

                sentences = without_audio = 0
                outputs.update(sentences=0, sentences_without_audio=0)
                async for sentence, speech in voice.speak(spoken):
                    events.put_nowait(
                        {
                            "type": "sentence",
                            "index": sentences,
                            "text": sentence,
                            # null when the voice failed on this sentence - the
                            # page shows its text anyway
                            "audio": base64.b64encode(speech.audio).decode("ascii")
                            if speech
                            else None,
                            "word_starts_ms": speech.word_starts_ms if speech else [],
                            "speech_end_ms": speech.end_ms if speech else 0,
                        }
                    )
                    sentences += 1
                    without_audio += speech is None
                    outputs.update(sentences=sentences, sentences_without_audio=without_audio)
            events.put_nowait({"type": "done"})
        except asyncio.CancelledError:
            # the listener cut the answer short (or left) while it was still
            # being produced: worth seeing in the trace, but not a failure
            interrupted = True
            mlflow.update_current_trace(tags={"voice.interrupted": "true"})
        except APIError as exc:
            # the language model or speech recognition failed or is rate-limited
            logger.warning("voice turn: a model call failed: %s", exc)
            turn.record_exception(exc)
            events.put_nowait(
                {
                    "type": "error",
                    "status": 503,
                    "detail": "The assistant is temporarily unavailable. Please try again.",
                }
            )
        except Exception as exc:
            # same promise as unhandled_error above: never a raw crash
            logger.exception("voice turn failed")
            turn.record_exception(exc)
            events.put_nowait(
                {"type": "error", "status": 500, "detail": f"Unexpected error: {exc}"[:300]}
            )
        finally:
            turn.set_outputs(outputs)
            events.put_nowait(None)
    if interrupted:
        raise asyncio.CancelledError


async def _voice_response(
    *,
    user_email: str,
    conversation_id: str | None,
    model: str | None,
    message: str | None = None,
    recording: tuple[bytes, str] | None = None,
) -> StreamingResponse:
    """Start a voice turn and stream its events as JSON lines.

    The response only begins once the turn has produced its first event, so a
    turn that fails before anything was produced is still a normal error
    response (503 / 500) rather than a stream that opens and reports failure.
    """
    events: _VoiceEvents = asyncio.Queue()
    turn = asyncio.create_task(
        _voice_turn(
            events,
            user_email=user_email,
            conversation_id=conversation_id,
            model=model,
            message=message,
            recording=recording,
        )
    )
    try:
        first = await events.get()
    except asyncio.CancelledError:
        turn.cancel()
        raise
    if first is not None and first["type"] == "error":
        raise HTTPException(status_code=int(str(first["status"])), detail=str(first["detail"]))

    async def lines() -> AsyncIterator[str]:
        try:
            event = first
            while event is not None:
                yield json.dumps(event, ensure_ascii=False) + "\n"
                event = await events.get()
        finally:
            turn.cancel()  # the client stopped listening; does nothing once the turn is over

    return StreamingResponse(lines(), media_type="application/x-ndjson")


@app.post("/voice/ask")
async def post_voice_ask(body: ChatIn, request: Request) -> StreamingResponse:
    """A typed question, answered aloud.

    The response is a stream of JSON lines, so the page can start talking
    before the whole answer has been turned into audio:

        {"type": "answer", ...}    the /chat fields plus `spoken`
        {"type": "sentence", "index", "text", "audio", "word_starts_ms",
         "speech_end_ms"}          one per spoken sentence, in order; `audio`
                                   is base64 mp3, `word_starts_ms` says when
                                   each word of `text` begins in it, and
                                   `speech_end_ms` when the last one ends
                                   (the clip has silence after that)
        {"type": "done"}
        {"type": "error", "status", "detail"}
                                   instead of the rest, if the turn fails
                                   after the stream has started

    A failure before the first line is a normal error response (503), not a
    stream.
    """
    return await _voice_response(
        user_email=request.state.user_email,
        conversation_id=body.conversation_id,
        model=body.model,
        message=body.message,
    )


@app.post("/voice/turn")
async def post_voice_turn(
    request: Request, conversation_id: str | None = None, model: ModelId | None = None
) -> StreamingResponse:
    """A spoken question, answered aloud.

    The request body is the recording itself; Content-Type says its format.
    `conversation_id` continues a conversation, as in /chat.

    The response is the same stream as /voice/ask, with one line before the
    answer: {"type": "heard", "text": ...} - what the recording was understood
    to say. If that text is empty there is nothing to answer, and the stream
    ends there.
    """
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    extension = _AUDIO_EXTENSIONS.get(content_type)
    if extension is None:
        raise HTTPException(status_code=415, detail="unsupported audio format")
    audio = await request.body()
    if not audio:
        raise HTTPException(status_code=422, detail="the recording is empty")
    if len(audio) > _MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="the recording is too long")
    return await _voice_response(
        user_email=request.state.user_email,
        conversation_id=conversation_id,
        model=model,
        recording=(audio, f"speech.{extension}"),
    )


@app.get("/conversations", response_model=list[ConversationSummaryOut])
def get_conversations(request: Request) -> list[ConversationSummaryOut]:
    return [
        ConversationSummaryOut(
            id=c.id,
            title=c.title,
            folder_id=c.folder_id,
            updated_at=c.updated_at,
            message_count=c.message_count,
        )
        for c in store.list_conversations(request.state.user_email)
    ]


@app.get("/conversations/{conversation_id}", response_model=ConversationOut)
def get_conversation(conversation_id: str, request: Request) -> ConversationOut:
    conversation = store.get_conversation(conversation_id, request.state.user_email)
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
def rename_conversation(
    conversation_id: str, body: RenameIn, request: Request
) -> ConversationSummaryOut:
    if not store.rename_conversation(conversation_id, body.title, request.state.user_email):
        raise HTTPException(status_code=404, detail="conversation not found")
    updated = next(
        c
        for c in store.list_conversations(request.state.user_email)
        if c.id == conversation_id
    )
    return ConversationSummaryOut(
        id=updated.id,
        title=updated.title,
        folder_id=updated.folder_id,
        updated_at=updated.updated_at,
        message_count=updated.message_count,
    )


@app.put("/conversations/{conversation_id}/folder", response_model=ConversationSummaryOut)
def move_conversation(
    conversation_id: str, body: SetFolderIn, request: Request
) -> ConversationSummaryOut:
    if not store.set_conversation_folder(
        conversation_id, body.folder_id, request.state.user_email
    ):
        raise HTTPException(status_code=404, detail="conversation not found")
    updated = next(
        c
        for c in store.list_conversations(request.state.user_email)
        if c.id == conversation_id
    )
    return ConversationSummaryOut(
        id=updated.id,
        title=updated.title,
        folder_id=updated.folder_id,
        updated_at=updated.updated_at,
        message_count=updated.message_count,
    )


@app.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict[str, str]:
    if not store.delete_conversation(conversation_id, request.state.user_email):
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"status": "deleted"}


@app.get("/folders", response_model=list[FolderOut])
def get_folders(request: Request) -> list[FolderOut]:
    return [
        FolderOut(id=f.id, name=f.name, created_at=f.created_at)
        for f in store.list_folders(request.state.user_email)
    ]


@app.post("/folders", response_model=FolderOut)
def post_folder(body: FolderIn, request: Request) -> FolderOut:
    folder_id = store.create_folder(body.name, request.state.user_email)
    created = next(f for f in store.list_folders(request.state.user_email) if f.id == folder_id)
    return FolderOut(id=created.id, name=created.name, created_at=created.created_at)


@app.patch("/folders/{folder_id}", response_model=FolderOut)
def patch_folder(folder_id: str, body: FolderIn, request: Request) -> FolderOut:
    if not store.rename_folder(folder_id, body.name, request.state.user_email):
        raise HTTPException(status_code=404, detail="folder not found")
    updated = next(f for f in store.list_folders(request.state.user_email) if f.id == folder_id)
    return FolderOut(id=updated.id, name=updated.name, created_at=updated.created_at)


@app.delete("/folders/{folder_id}")
def remove_folder(folder_id: str, request: Request) -> dict[str, str]:
    if not store.delete_folder(folder_id, request.state.user_email):
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
        model=run.model,
        duration_ms=run.duration_ms,
        created_at=run.created_at,
        trace_url=trace_url(run.trace_id),
        feedback=run.feedback,
        feedback_comment=run.feedback_comment,
    )


@app.get("/runs", response_model=list[RunOut])
def get_runs(request: Request) -> list[RunOut]:
    return [_run_out(run) for run in store.list_runs(request.state.user_email)]


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: str, request: Request) -> RunOut:
    run = store.get_run(run_id, request.state.user_email)
    if run is None:
        raise HTTPException(status_code=404, detail="run not found")
    return _run_out(run)


@app.post("/runs/{run_id}/feedback")
def post_feedback(run_id: str, body: FeedbackIn, request: Request) -> dict[str, str]:
    if not record_feedback(run_id, body.helpful, body.comment, request.state.user_email):
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
