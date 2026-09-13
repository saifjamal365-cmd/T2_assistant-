"""The HTTP API.

    GET  /                      -> the web chat page
    GET  /health                -> {"status": "ok"}
    POST /chat                  -> send a message, get the reply + route + run id
    GET  /conversations         -> list past conversations
    GET  /conversations/{id}    -> one conversation with all its messages
    GET  /runs                  -> recent requests, each with a link to its trace
    GET  /runs/{id}             -> one request
    POST /runs/{id}/feedback    -> rate a run (helpful / not) - also sent to MLflow

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

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from groq import APIError
from pydantic import BaseModel, Field, field_validator

from t2_assistant import store
from t2_assistant.conversation import run_turn
from t2_assistant.observability import record_feedback, trace_url

logger = logging.getLogger("t2_assistant.api")

app = FastAPI(title="T2 Assistant", version="0.1.0")

_INDEX_HTML = Path(__file__).parent / "web" / "index.html"
_MAX_MESSAGE_LENGTH = 4000  # generous for a question or a page of meeting notes


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


class ChatOut(BaseModel):
    conversation_id: str
    run_id: str
    reply: str
    route: str
    route_reason: str
    trace_url: str | None


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: str


class ConversationSummaryOut(BaseModel):
    id: str
    title: str
    updated_at: str
    message_count: int


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    messages: list[MessageOut]


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


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(_INDEX_HTML.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
    )


@app.get("/conversations", response_model=list[ConversationSummaryOut])
def get_conversations() -> list[ConversationSummaryOut]:
    return [
        ConversationSummaryOut(
            id=c.id, title=c.title, updated_at=c.updated_at, message_count=c.message_count
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
        created_at=conversation.created_at,
        updated_at=conversation.updated_at,
        messages=[
            MessageOut(role=m.role, content=m.content, created_at=m.created_at)
            for m in conversation.messages
        ],
    )


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
