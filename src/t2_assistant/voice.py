"""Voice mode: a spoken question in, a spoken Saudi-dialect answer out.

Three small steps wrapped around the existing chat turn (conversation.py),
which is left exactly as it is - the facts still come from the same tested,
grounded answer pipeline:

    transcribe()   audio -> text            (Groq Whisper, same API key as the LLM)
    to_spoken()    written reply -> what a Saudi colleague would say out loud
    speak()        spoken text -> its sentences, each with mp3 audio and the
                   moment every word starts (Microsoft's ar-SA neural voice,
                   via edge-tts) - so a page can show each word as it is said

`to_spoken()` is the only step that could damage accuracy: it rewrites an
answer that was already checked against the documents. So it is guarded - if
the rewrite does not carry exactly the same numbers as the written answer, it
is thrown away and the written answer is spoken instead. A slightly formal
answer is better than a Saudi-sounding wrong one.

edge-tts uses the free read-aloud service behind Microsoft Edge. It needs no
key, which suits a prototype, but it is not a licensed production service -
swap `_synthesize()` for Azure Speech (same voices) before a real rollout.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from functools import lru_cache

import aiohttp
import edge_tts
import mlflow
from edge_tts.exceptions import EdgeTTSException
from groq import Groq
from langchain_core.messages import HumanMessage, SystemMessage
from mlflow.entities import SpanType

from t2_assistant.agents.answer import _DOC_ID, _answer_body, _is_decline
from t2_assistant.agents.llm import get_llm
from t2_assistant.config import settings

_DONT_KNOW_SPOKEN = "ما لقيت جواب لهالسؤال في مستندات الشركة الحالية. ممكن توضّح لي سؤالك أكثر؟"

_SPOKEN_SYSTEM = """You turn a written answer from T2's internal company assistant
into what a Saudi colleague would SAY out loud to another employee.

Rewrite the text below in natural spoken Saudi Arabic - the everyday "white"
dialect understood across the Kingdom (for example: وش، أبغى، تقدر، لازم، عشان،
مو، الحين) - whatever language the text is written in. Friendly and
professional, not slang.

Rules:
- Keep every fact exactly as given: the same numbers, durations, amounts,
  conditions, exceptions and names. Write every number in digits, exactly as it
  appears in the text.
- Do not add any fact, advice or opinion that is not in the text, and do not
  drop a condition or an exception.
- It will be read aloud: short sentences, no document codes (like HR-0003), no
  "Sources" line, no lists, no markdown, no brackets or symbols.
- If the text asks the user a question, keep it a question.
- Reply with the spoken text only."""

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_NUMBER = re.compile(r"\d+")
_MARKDOWN = re.compile(r"[*_#`>|]+")
_SPACES = re.compile(r"\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")
_MIN_SENTENCE_CHARS = 25  # shorter pieces ("نعم.") join the next one, or the voice sounds chopped
_TICKS_PER_MS = 10_000  # edge-tts reports time in 100-nanosecond ticks

logger = logging.getLogger("t2_assistant.voice")


@dataclass
class Speech:
    """One sentence read aloud."""

    audio: bytes  # mp3
    word_starts_ms: list[int]  # when each word begins in `audio`, in order
    # when the last word ends: the clip carries close to a second of silence
    # after it, which a player can skip between sentences
    end_ms: int


@lru_cache
def _groq() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def transcribe(audio: bytes, filename: str) -> str:
    """The words in a short recording. `filename` only tells Groq the audio
    format (its extension), so it must match what the browser recorded."""
    result = _groq().audio.transcriptions.create(
        file=(filename, audio),
        model=settings.stt_model,
        language=settings.stt_language,
        temperature=0.0,
    )
    return result.text.strip()


def _speakable(reply: str) -> str:
    """The written reply with everything that should not be read aloud removed:
    the "Sources:" line, document codes, and markdown symbols."""
    text = _DOC_ID.sub("", _answer_body(reply))
    return _SPACES.sub(" ", _MARKDOWN.sub(" ", text)).strip()


def _numbers(text: str) -> set[str]:
    return set(_NUMBER.findall(text.translate(_ARABIC_INDIC_DIGITS)))


@mlflow.trace(name="voice_dialect", span_type=SpanType.LLM)
def to_spoken(reply: str, model: str) -> str:
    """`reply` (a finished written answer) as Saudi-dialect text to read aloud.

    Falls back to the written answer itself, cleaned for speech, whenever the
    dialect rewrite does not carry exactly the same numbers - see the module
    docstring."""
    if _is_decline(reply):
        return _DONT_KNOW_SPOKEN
    written = _speakable(reply)
    if not written:
        return written

    rewrite = get_llm(model).invoke([SystemMessage(_SPOKEN_SYSTEM), HumanMessage(written)])
    spoken = _speakable(str(rewrite.content))
    if not spoken or _numbers(spoken) != _numbers(written):
        return written
    return spoken


def split_sentences(text: str) -> list[str]:
    """`text` cut into the pieces that are read aloud one after another."""
    sentences: list[str] = []
    current = ""
    for piece in _SENTENCE_END.split(text.strip()):
        current = f"{current} {piece}".strip()
        if len(current) >= _MIN_SENTENCE_CHARS:
            sentences.append(current)
            current = ""
    if current:
        sentences.append(current)
    return sentences


async def _synthesize(text: str) -> Speech:
    """`text` read aloud by the configured Saudi voice."""
    audio = bytearray()
    word_starts_ms: list[int] = []
    end_ms = 0
    stream = edge_tts.Communicate(text, settings.tts_voice, boundary="WordBoundary").stream()
    async for chunk in stream:
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
        elif chunk["type"] == "WordBoundary":
            word_starts_ms.append(int(chunk["offset"]) // _TICKS_PER_MS)
            end_ms = (int(chunk["offset"]) + int(chunk["duration"])) // _TICKS_PER_MS
    return Speech(audio=bytes(audio), word_starts_ms=word_starts_ms, end_ms=end_ms)


async def speak(text: str) -> AsyncIterator[tuple[str, Speech | None]]:
    """Each sentence of `text`, in order, with its audio.

    The sentences are synthesised one at a time, running ahead of the
    listener. One at a time on purpose: measured against the voice service, a
    lone request has the first sentence ready in about 0.6s every time, while
    several at once made that anything from 0.9s to 6s. Nothing is lost by
    it - a sentence takes under a second to make and several seconds to play,
    so the next one is ready well before it is needed.

    A sentence the service fails on comes back with None instead of ending
    the whole answer - the caller still has its text."""
    one_at_a_time = asyncio.Lock()

    async def one(sentence: str) -> Speech | None:
        async with one_at_a_time:
            try:
                return await _synthesize(sentence)
            except (EdgeTTSException, aiohttp.ClientError, TimeoutError) as exc:
                logger.warning("text-to-speech call failed: %s", exc)
                return None

    sentences = split_sentences(text)
    # started in order, and the lock is handed on in the order it was asked for
    tasks = [asyncio.create_task(one(sentence)) for sentence in sentences]
    try:
        for sentence, task in zip(sentences, tasks, strict=True):
            yield sentence, await task
    finally:
        # the listener interrupted or left: stop paying for audio nobody will hear
        for task in tasks:
            task.cancel()
