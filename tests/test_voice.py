"""Voice mode: the dialect rewrite's accuracy guard, the /voice endpoints, and
the MLflow trace a voice turn leaves behind.

No test here calls Groq or the speech service - the model and both speech
steps are faked, so what is checked is this project's own logic around them.
Traces go to a throwaway MLflow store (see conftest.py).
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import edge_tts
import httpx
import mlflow
import pytest
from edge_tts.exceptions import NoAudioReceived
from fastapi.testclient import TestClient
from groq import APIConnectionError
from langchain_core.messages import AIMessage
from mlflow.entities import Trace

from t2_assistant import api, conversation, store, voice
from t2_assistant.agents.answer import _DONT_KNOW_AR, _DONT_KNOW_EN
from t2_assistant.api import app
from t2_assistant.conversation import TurnResult

client = TestClient(app)
client.cookies.set("t2_session", store.create_session("test@t2.sa"))

WRITTEN = "يستحق الموظف 30 يوم إجازة سنوية بعد إكمال 12 شهراً.\nالمصادر: HR-0001, HR-0015"


def _fake_llm(reply: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Make the dialect rewrite return `reply`; collects what it was asked to rewrite."""
    seen: list[str] = []

    class FakeLLM:
        def invoke(self, messages: list[object]) -> SimpleNamespace:
            seen.append(str(messages[-1].content))  # type: ignore[attr-defined]
            return SimpleNamespace(content=reply)

    monkeypatch.setattr(voice, "get_llm", lambda model: FakeLLM())
    return seen


def test_spoken_keeps_a_rewrite_with_the_same_numbers(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = _fake_llm("لك 30 يوم إجازة سنوية بعد ما تكمل 12 شهر.", monkeypatch)
    assert voice.to_spoken(WRITTEN, "m") == "لك 30 يوم إجازة سنوية بعد ما تكمل 12 شهر."
    # the model is never shown the sources line or a document code
    assert "HR-0001" not in seen[0]
    assert "المصادر" not in seen[0]


def test_spoken_accepts_arabic_indic_digits(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_llm("لك ٣٠ يوم إجازة بعد ١٢ شهر.", monkeypatch)
    assert voice.to_spoken(WRITTEN, "m") == "لك ٣٠ يوم إجازة بعد ١٢ شهر."


@pytest.mark.parametrize(
    "rewrite",
    [
        "لك 21 يوم إجازة سنوية بعد ما تكمل 12 شهر.",  # a number changed
        "لك 30 يوم إجازة سنوية.",  # a condition (and its number) dropped
        "لك 30 يوم إجازة بعد 12 شهر، وتقدر ترحّل 5 أيام.",  # a fact added
        "",  # the model returned nothing
    ],
)
def test_spoken_falls_back_to_the_written_answer(
    rewrite: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    _fake_llm(rewrite, monkeypatch)
    assert voice.to_spoken(WRITTEN, "m") == "يستحق الموظف 30 يوم إجازة سنوية بعد إكمال 12 شهراً."


def test_spoken_strips_codes_and_markdown_the_model_left_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_llm("**لك 30 يوم** إجازة بعد 12 شهر (HR-0001).", monkeypatch)
    spoken = voice.to_spoken(WRITTEN, "m")
    assert "HR-0001" not in spoken
    assert "*" not in spoken


@pytest.mark.parametrize("decline", [_DONT_KNOW_AR, _DONT_KNOW_EN])
def test_a_decline_is_spoken_as_a_fixed_sentence_without_a_model_call(
    decline: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = _fake_llm("should not be used", monkeypatch)
    assert voice.to_spoken(decline, "m") == voice._DONT_KNOW_SPOKEN
    assert seen == []


def test_voice_page_loads_without_a_session() -> None:
    response = TestClient(app).get("/voice")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_voice_endpoints_require_a_session() -> None:
    anonymous = TestClient(app)
    assert anonymous.post("/voice/ask", json={"message": "hi"}).status_code == 401
    recording = anonymous.post("/voice/turn", content=b"x", headers={"Content-Type": "audio/webm"})
    assert recording.status_code == 401


def _fake_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run_turn(
        message: str,
        conversation_id: str | None = None,
        *,
        model: str | None = None,
        user_email: str = "test@t2.sa",
        highlights: bool = True,
    ) -> TurnResult:
        # nothing on the voice page shows highlights, so it must not pay for them
        assert highlights is False
        return TurnResult("c1", "r1", None, WRITTEN, "answer", "a policy question", [], "m")

    monkeypatch.setattr(api, "run_turn", fake_run_turn)
    monkeypatch.setattr(voice, "to_spoken", lambda reply, model: "لك 30 يوم. بعد 12 شهر.")


def _events(body: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in body.splitlines() if line]


def test_a_recording_is_heard_then_answered_in_one_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_turn(monkeypatch)
    calls: list[tuple[bytes, str]] = []

    def fake_transcribe(audio: bytes, filename: str) -> str:
        calls.append((audio, filename))
        return "كم يوم إجازة"

    async def fake_speak(text: str) -> AsyncIterator[tuple[str, voice.Speech | None]]:
        yield "لك 30 يوم.", None

    monkeypatch.setattr(voice, "transcribe", fake_transcribe)
    monkeypatch.setattr(voice, "speak", fake_speak)

    response = client.post(
        "/voice/turn", content=b"abc", headers={"Content-Type": "audio/webm;codecs=opus"}
    )
    assert response.status_code == 200
    events = _events(response.text)
    assert [event["type"] for event in events] == ["heard", "answer", "sentence", "done"]
    assert events[0] == {"type": "heard", "text": "كم يوم إجازة"}
    # the recording's format comes from the Content-Type the browser sent
    assert calls == [(b"abc", "speech.webm")]


def test_a_recording_with_no_words_in_it_is_not_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    def never(*args: object, **kwargs: object) -> TurnResult:
        raise AssertionError("an empty transcript must not reach the assistant")

    monkeypatch.setattr(voice, "transcribe", lambda audio, filename: "")
    monkeypatch.setattr(api, "run_turn", never)

    response = client.post("/voice/turn", content=b"abc", headers={"Content-Type": "audio/webm"})
    assert _events(response.text) == [{"type": "heard", "text": ""}, {"type": "done"}]


def test_voice_turn_rejects_a_bad_recording(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice, "transcribe", lambda audio, filename: "never reached")
    webm = {"Content-Type": "audio/webm"}
    assert client.post("/voice/turn", content=b"", headers=webm).status_code == 422
    text = client.post("/voice/turn", content=b"abc", headers={"Content-Type": "text/plain"})
    assert text.status_code == 415
    monkeypatch.setattr(api, "_MAX_AUDIO_BYTES", 2)
    assert client.post("/voice/turn", content=b"abc", headers=webm).status_code == 413


def test_voice_turn_reports_a_recognition_failure_as_an_error_not_a_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_transcribe(audio: bytes, filename: str) -> str:
        raise APIConnectionError(request=httpx.Request("POST", "https://groq.example"))

    monkeypatch.setattr(voice, "transcribe", failing_transcribe)
    response = client.post("/voice/turn", content=b"abc", headers={"Content-Type": "audio/webm"})
    assert response.status_code == 503
    assert "detail" in response.json()


def test_voice_ask_streams_the_answer_then_each_spoken_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_turn(monkeypatch)

    async def fake_speak(text: str) -> AsyncIterator[tuple[str, voice.Speech | None]]:
        yield (
            "لك 30 يوم.",
            voice.Speech(audio=b"mp3-one", word_starts_ms=[100, 400, 900], end_ms=1300),
        )
        yield "بعد 12 شهر.", None  # the voice failed on this sentence

    monkeypatch.setattr(voice, "speak", fake_speak)

    response = client.post("/voice/ask", json={"message": "كم يوم إجازة؟"})
    assert response.status_code == 200
    answer, first, second, done = _events(response.text)

    assert answer["type"] == "answer"
    assert answer["reply"] == WRITTEN  # the written answer, with its sources, is kept
    assert answer["spoken"] == "لك 30 يوم. بعد 12 شهر."
    assert answer["conversation_id"] == "c1"

    assert first == {
        "type": "sentence",
        "index": 0,
        "text": "لك 30 يوم.",
        "audio": base64.b64encode(b"mp3-one").decode("ascii"),
        "word_starts_ms": [100, 400, 900],
        "speech_end_ms": 1300,
    }
    # a sentence without audio still arrives, so its text is never lost
    assert second == {
        "type": "sentence",
        "index": 1,
        "text": "بعد 12 شهر.",
        "audio": None,
        "word_starts_ms": [],
        "speech_end_ms": 0,
    }
    assert done == {"type": "done"}


def test_voice_ask_reports_a_model_failure_as_an_error_not_a_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_run_turn(*args: object, **kwargs: object) -> TurnResult:
        raise APIConnectionError(request=httpx.Request("POST", "https://groq.example"))

    monkeypatch.setattr(api, "run_turn", failing_run_turn)
    response = client.post("/voice/ask", json={"message": "كم يوم إجازة؟"})
    assert response.status_code == 503
    assert "detail" in response.json()


def test_split_sentences_joins_pieces_too_short_to_speak_alone() -> None:
    text = "نعم. تقدر ترحّل لحد 5 أيام للسنة الجاية؟ والباقي يروح عليك! شكراً."
    assert voice.split_sentences(text) == [
        "نعم. تقدر ترحّل لحد 5 أيام للسنة الجاية؟",
        "والباقي يروح عليك! شكراً.",
    ]
    assert voice.split_sentences("هلا.") == ["هلا."]
    assert voice.split_sentences("  ") == []


def test_speak_keeps_sentence_order_and_survives_one_failed_sentence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = "الجملة الأولى طويلة بما يكفي لتُقرأ لوحدها."
    second = "الجملة الثانية تفشل عند خدمة الصوت دائماً."
    third = "الجملة الثالثة تنجح بعد فشل اللي قبلها."

    async def fake_synthesize(text: str) -> voice.Speech:
        if text == second:
            raise NoAudioReceived("no audio")
        # the first sentence is the slowest, so order cannot come from who finishes first
        await asyncio.sleep(0.05 if text == first else 0)
        return voice.Speech(audio=text.encode(), word_starts_ms=[0], end_ms=500)

    monkeypatch.setattr(voice, "_synthesize", fake_synthesize)

    async def collect() -> list[tuple[str, voice.Speech | None]]:
        return [item async for item in voice.speak(f"{first} {second} {third}")]

    spoken = asyncio.run(collect())
    assert [sentence for sentence, _ in spoken] == [first, second, third]
    assert spoken[0][1] == voice.Speech(audio=first.encode(), word_starts_ms=[0], end_ms=500)
    assert spoken[1][1] is None
    assert spoken[2][1] == voice.Speech(audio=third.encode(), word_starts_ms=[0], end_ms=500)


def test_speak_asks_the_voice_service_for_one_sentence_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Several requests at once make the voice service slow and erratic
    (measured), so sentences are synthesised strictly one after another."""
    sentences = [
        "الجملة الأولى طويلة بما يكفي لتُقرأ لوحدها.",
        "الجملة الثانية كذلك طويلة بما يكفي لوحدها.",
        "الجملة الثالثة هي الأخيرة في هذا الجواب.",
    ]
    in_flight = 0
    most_at_once = 0
    order: list[str] = []

    async def fake_synthesize(text: str) -> voice.Speech:
        nonlocal in_flight, most_at_once
        in_flight += 1
        most_at_once = max(most_at_once, in_flight)
        order.append(text)
        await asyncio.sleep(0.01)
        in_flight -= 1
        return voice.Speech(audio=b"ok", word_starts_ms=[0], end_ms=500)

    monkeypatch.setattr(voice, "_synthesize", fake_synthesize)

    async def collect() -> list[tuple[str, voice.Speech | None]]:
        return [item async for item in voice.speak(" ".join(sentences))]

    asyncio.run(collect())
    assert most_at_once == 1
    assert order == sentences


# ---- tracing: a voice turn is one MLflow trace ------------------------------


class _FakeCommunicate:
    """Stands in for the voice service: a little audio, and one timing per word."""

    def __init__(self, text: str, voice: str, **kwargs: object) -> None:
        self.text = text

    async def stream(self) -> AsyncIterator[dict[str, object]]:
        yield {"type": "audio", "data": b"fake-mp3"}
        for index, word in enumerate(self.text.split()):
            yield {
                "type": "WordBoundary",
                "offset": index * 5_000_000,
                "duration": 4_000_000,
                "text": word,
            }


def _only_outside_services_faked(
    monkeypatch: pytest.MonkeyPatch, *, heard: str, reply: str, rewrite: str
) -> None:
    """Fake only what lives outside this project - speech recognition, the
    agent graph, the dialect model, the voice. This project's own steps then
    run for real, so each one records its own span, as it does in production."""
    recogniser = SimpleNamespace(
        audio=SimpleNamespace(
            transcriptions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(text=heard))
        )
    )
    monkeypatch.setattr(voice, "_groq", lambda: recogniser)

    def invoke(state: dict[str, Any]) -> dict[str, Any]:
        return {
            **state,
            "messages": [*state["messages"], AIMessage(reply)],
            "route": "answer",
            "route_reason": "a policy question",
        }

    monkeypatch.setattr(conversation, "compiled_graph", SimpleNamespace(invoke=invoke))
    _fake_llm(rewrite, monkeypatch)
    monkeypatch.setattr(edge_tts, "Communicate", _FakeCommunicate)


def _trace_of(run_id: str) -> Trace:
    """The trace the saved run points at - the one the page's trace link opens."""
    run = store.get_run(run_id, "test@t2.sa")
    assert run is not None
    assert run.trace_id is not None
    mlflow.flush_trace_async_logging()
    trace: Trace | None = mlflow.get_trace(run.trace_id)
    assert trace is not None
    return trace


_VOICE_STEPS = ("speech_to_text", "chat", "voice_dialect", "text_to_speech")


def test_a_spoken_turn_is_one_trace_with_every_step_inside(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _only_outside_services_faked(
        monkeypatch,
        heard="كم يوم إجازة سنوية لي؟",
        reply=WRITTEN,
        rewrite="لك 30 يوم إجازة سنوية مدفوعة كل سنة. وهذا بعد ما تكمل 12 شهر في الشركة.",
    )
    recording = b"RAW-RECORDING-BYTES"

    response = client.post("/voice/turn", content=recording, headers={"Content-Type": "audio/webm"})
    answer = next(event for event in _events(response.text) if event["type"] == "answer")
    trace = _trace_of(answer["run_id"])
    spans = trace.data.spans

    # one trace, whose top is the voice turn...
    roots = [span for span in spans if span.parent_id is None]
    assert [span.name for span in roots] == ["voice_turn"]
    # ...and every step of the turn sits directly under it. Compared as a
    # sorted list: MLflow does not promise the order spans are listed in, and
    # with faked services two steps can start within the same clock tick.
    steps = [span for span in spans if span.name in _VOICE_STEPS]
    assert sorted(span.name for span in steps if span.parent_id == roots[0].span_id) == [
        "chat",
        "speech_to_text",
        "text_to_speech",  # one per spoken sentence
        "text_to_speech",
        "voice_dialect",
    ]

    tags = trace.info.tags
    assert tags["channel"] == "voice"
    assert tags["voice.input"] == "speech"
    assert tags["voice.spoken_from"] == "rewrite"
    assert trace.info.trace_metadata["mlflow.trace.user"] == "test@t2.sa"

    outputs = roots[0].outputs
    assert outputs["heard"] == "كم يوم إجازة سنوية لي؟"
    assert outputs["reply"] == WRITTEN
    assert outputs["spoken"] == answer["spoken"]
    assert outputs["sentences"] == 2
    assert outputs["sentences_without_audio"] == 0

    # the link the page shows opens this very trace
    assert trace.info.trace_id in answer["trace_url"]
    # the sound itself is never stored - neither the question nor the answer
    stored = json.dumps(trace.to_dict(), default=str)
    assert "RAW-RECORDING-BYTES" not in stored
    assert base64.b64encode(recording).decode("ascii") not in stored
    assert "fake-mp3" not in stored
    assert base64.b64encode(b"fake-mp3").decode("ascii") not in stored


def test_a_typed_voice_turn_is_traced_too_and_shows_a_refused_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # the dialect model changes 30 days to 21: the accuracy check must refuse it
    _only_outside_services_faked(
        monkeypatch,
        heard="not used - the question is typed",
        reply=WRITTEN,
        rewrite="لك 21 يوم إجازة سنوية بعد ما تكمل 12 شهر في الشركة.",
    )

    response = client.post("/voice/ask", json={"message": "كم يوم إجازة؟"})
    answer = next(event for event in _events(response.text) if event["type"] == "answer")
    assert answer["spoken"] == "يستحق الموظف 30 يوم إجازة سنوية بعد إكمال 12 شهراً."

    trace = _trace_of(answer["run_id"])
    names = [span.name for span in trace.data.spans]
    assert [span.name for span in trace.data.spans if span.parent_id is None] == ["voice_turn"]
    assert "speech_to_text" not in names  # nothing was heard: the question was typed
    assert trace.info.tags["voice.input"] == "text"

    # the refusal can be found in MLflow, with the reason
    assert trace.info.tags["voice.spoken_from"] == "written"
    dialect = next(span for span in trace.data.spans if span.name == "voice_dialect")
    assert dialect.attributes["numbers_in_written"] == ["12", "30"]
    assert dialect.attributes["numbers_in_rewrite"] == ["12", "21"]


def test_an_interrupted_turn_says_so_in_its_trace_and_is_not_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _fake_turn(monkeypatch)

    async def never_finishes(text: str) -> AsyncIterator[tuple[str, voice.Speech | None]]:
        yield "لك 30 يوم.", None
        await asyncio.Event().wait()

    monkeypatch.setattr(voice, "speak", never_finishes)

    def interrupted_traces() -> dict[str, Trace]:
        mlflow.flush_trace_async_logging()
        found = mlflow.search_traces(
            filter_string="tags.channel = 'voice'", return_type="list", max_results=500
        )
        return {
            trace.info.trace_id: trace
            for trace in found
            if trace.info.tags.get("voice.interrupted") == "true"
        }

    async def start_then_interrupt() -> None:
        events: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
        turn = asyncio.create_task(
            api._voice_turn(
                events,
                user_email="test@t2.sa",
                conversation_id=None,
                model=None,
                message="كم يوم إجازة؟",
            )
        )
        while (event := await events.get()) is not None and event["type"] != "sentence":
            pass
        turn.cancel()  # what the server does when the listener stops the answer
        with pytest.raises(asyncio.CancelledError):
            await turn

    before = interrupted_traces()
    asyncio.run(start_then_interrupt())
    new = [trace for trace_id, trace in interrupted_traces().items() if trace_id not in before]

    assert len(new) == 1
    assert new[0].info.state == "OK"
    root = next(span for span in new[0].data.spans if span.parent_id is None)
    assert root.name == "voice_turn"
    assert root.outputs["sentences"] == 1  # how far the answer got
