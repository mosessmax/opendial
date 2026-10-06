"""Provider protocols for the caller's brain, voice and ears.

opendial never imports a vendor SDK in its core. Anything that implements
these protocols works. The fakes below are deterministic, need no network,
and are what opendial's own tests run on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Literal, Protocol, TypedDict

import numpy as np

from opendial.audio.frames import SAMPLE_RATE, AudioFrame, to_frames


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLM(Protocol):
    async def complete(self, messages: list[Message]) -> str: ...


class TTS(Protocol):
    async def synthesize(self, text: str, locale: str | None = None) -> list[AudioFrame]: ...


class STT(Protocol):
    async def transcribe(self, frames: Sequence[AudioFrame]) -> str: ...


class ScriptedLLM:
    """Replies with fixed lines in order, or with whatever `reply(messages)` returns."""

    def __init__(self, lines: Iterable[str] | Callable[[list[Message]], str]):
        if callable(lines):
            self._reply = lines
        else:
            remaining = iter(lines)
            self._reply = lambda _messages: next(remaining, "[END]")
        self.calls: list[list[Message]] = []

    async def complete(self, messages: list[Message]) -> str:
        self.calls.append(list(messages))
        return self._reply(messages)


class FakeTTS:
    """Speech-shaped tones, 300 ms per word, with the text attached to the first frame."""

    ms_per_word = 300

    async def synthesize(self, text: str, locale: str | None = None) -> list[AudioFrame]:
        n = SAMPLE_RATE * self.ms_per_word * max(1, len(text.split())) // 1000
        t = np.arange(n) / SAMPLE_RATE
        wave = sum(np.sin(2 * np.pi * hz * t) for hz in (500, 1100, 1700)) / 3
        envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)  # syllable rhythm
        return to_frames((0.3 * 32767 * wave * envelope).astype(np.int16), text=text)


class FakeSTT:
    """Reads back the text attached by FakeTTS."""

    async def transcribe(self, frames: Sequence[AudioFrame]) -> str:
        return " ".join(f.text for f in frames if f.text)
