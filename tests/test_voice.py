"""Voice mode: the dialect rewrite's accuracy guard, and the /voice endpoints.

No test here calls Groq or the speech service - the model and both speech
steps are faked, so what is checked is this project's own logic around them.
"""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from edge_tts.exceptions import NoAudioReceived
from fastapi.testclient import TestClient
from groq import APIConnectionError

from t2_assistant import api, store, voice
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
    assert (
        anonymous.post(
            "/voice/transcribe", content=b"x", headers={"Content-Type": "audio/webm"}
        ).status_code
        == 401
    )


def test_transcribe_passes_the_recording_and_its_format(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[bytes, str]] = []

    def fake_transcribe(audio: bytes, filename: str) -> str:
        calls.append((audio, filename))
        return "كم يوم إجازة"

    monkeypatch.setattr(voice, "transcribe", fake_transcribe)
    response = client.post(
        "/voice/transcribe", content=b"abc", headers={"Content-Type": "audio/webm;codecs=opus"}
    )
    assert response.status_code == 200
    assert response.json() == {"text": "كم يوم إجازة"}
    assert calls == [(b"abc", "speech.webm")]


def test_transcribe_rejects_bad_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(voice, "transcribe", lambda audio, filename: "never reached")
    webm = {"Content-Type": "audio/webm"}
    assert client.post("/voice/transcribe", content=b"", headers=webm).status_code == 422
    assert (
        client.post(
            "/voice/transcribe", content=b"abc", headers={"Content-Type": "text/plain"}
        ).status_code
        == 415
    )
    monkeypatch.setattr(api, "_MAX_AUDIO_BYTES", 2)
    assert client.post("/voice/transcribe", content=b"abc", headers=webm).status_code == 413


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
