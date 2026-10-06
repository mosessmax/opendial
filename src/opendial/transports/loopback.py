"""Run an agent in-process. For opendial's own tests and for unit-testing agent logic."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

import numpy as np

from opendial.audio.frames import FRAME_MS, AudioFrame, silence, to_frames
from opendial.audio.vad import EnergyVad, VadEvent
from opendial.providers import STT, TTS, FakeSTT, FakeTTS
from opendial.transports import AgentHangup, AgentHeard, Event, ToolCall


@dataclass
class Reply:
    say: str = ""
    tools: list[ToolCall] = field(default_factory=list)
    hangup: bool = False


Agent = Callable[[str], Awaitable[Reply]]
"""Gets the caller's turn as text (`[DTMF 123#]` for key presses), returns a reply.

Called with "" at the start of the call so the agent can greet.
"""


class LoopbackTransport:
    def __init__(
        self,
        agent: Agent,
        *,
        greet: bool = True,
        latency_ms: int = 400,
        stop_ms: int = 200,
        endpoint_ms: int = 500,
        stt: STT | None = None,
        tts: TTS | None = None,
    ):
        self.agent = agent
        self.greet = greet
        self.latency_ms, self.stop_ms = latency_ms, stop_ms
        self.stt, self.tts = stt or FakeSTT(), tts or FakeTTS()
        self.vad = EnergyVad(end_ms=endpoint_ms)
        self.preroll: deque[AudioFrame] = deque(maxlen=10)  # 200 ms before VAD fires
        self.heard: list[AudioFrame] = []
        self.outbox: deque[AudioFrame] = deque()
        self.events: list[Event] = []
        self.stop_after: int | None = None
        self.hung_up = False

    async def connect(self) -> None:
        if self.greet:
            await self._respond("")

    async def exchange(self, frame: AudioFrame) -> AudioFrame:
        event = self.vad.feed(frame)
        if event is VadEvent.SPEECH_START:
            self.heard = list(self.preroll)
        if self.vad.speaking or event is VadEvent.SPEECH_END:
            self.heard.append(frame)
        else:
            self.preroll.append(frame)
        if event is VadEvent.SPEECH_START and self.outbox and self.stop_after is None:
            self.stop_after = self.stop_ms // FRAME_MS  # barge-in: stop talking soon
        if event is VadEvent.SPEECH_END:
            text = await self.stt.transcribe(self.heard)
            self.heard = []
            self.events.append(AgentHeard(text))
            await self._respond(text)

        if self.stop_after is not None:
            if self.stop_after == 0:
                self.outbox.clear()
                self.stop_after = None
            else:
                self.stop_after -= 1
        if self.outbox:
            return self.outbox.popleft()
        return AudioFrame(np.zeros_like(frame.samples), frame.sample_rate)

    async def send_dtmf(self, digits: str) -> None:
        await self._respond(f"[DTMF {digits}]")

    def drain_events(self) -> list[Event]:
        events, self.events = self.events, []
        return events

    async def close(self) -> None:
        self.outbox.clear()

    async def _respond(self, text: str) -> None:
        if self.hung_up:
            return
        reply = await self.agent(text)
        self.events.extend(reply.tools)
        if reply.say:
            self.outbox.extend(to_frames(silence(self.latency_ms)))
            self.outbox.extend(await self.tts.synthesize(reply.say))
        if reply.hangup:
            self.hung_up = True
            self.events.append(AgentHangup())
