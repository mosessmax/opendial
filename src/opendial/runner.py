"""Place one call: drive the caller and the transport in 20 ms ticks and record everything."""

from __future__ import annotations

from collections import deque

import numpy as np

from opendial.audio.frames import FRAME_MS, AudioFrame, silence
from opendial.audio.vad import Segmenter, VadEvent
from opendial.caller import Caller
from opendial.record import BargeIn, CallRecord, EndReason, Turn
from opendial.transports import AgentHangup, Transport


class _Call:
    def __init__(self, caller: Caller, transport: Transport):
        self.caller, self.transport = caller, transport
        self.limits = caller.scenario.limits
        self.rec = CallRecord(caller.scenario)
        self.ears = Segmenter()
        self.now = 0.0
        self.outbox: deque[AudioFrame] = deque()
        self.caller_turn: Turn | None = None
        self.waiting_since: float | None = 0.0  # when we started waiting for the agent
        self.agent_start = 0.0
        self.agent_turns = 0
        self.barge: BargeIn | None = None
        self.barge_due: tuple[float, str] | None = None
        self.agent_hung_up = False
        self.caller_audio: list[np.ndarray] = []
        self.agent_audio: list[np.ndarray] = []

    async def say(self, text: str) -> None:
        self.outbox.extend(await self.caller.speak(text))
        self.caller_turn = Turn("caller", text, self.now, self.now)
        self.rec.turns.append(self.caller_turn)

    async def run(self) -> CallRecord:
        await self.transport.connect()
        if self.caller.scenario.opening == "caller_first":
            text = await self.caller.reply("")
            if text:
                await self.say(text)
        try:
            self.rec.ended_by = await self.loop()
        finally:
            await self.transport.close()
        self.rec.caller_finished = self.caller.finished
        self.rec.turns.sort(key=lambda t: t.start_ms)
        self.rec.duration_ms = self.now
        self.rec.caller_audio = np.concatenate(self.caller_audio or [silence(0)])
        self.rec.agent_audio = np.concatenate(self.agent_audio or [silence(0)])
        return self.rec

    async def loop(self) -> EndReason:
        while self.now < self.limits.max_duration_s * 1000:
            out = self.outbox.popleft() if self.outbox else AudioFrame(silence(FRAME_MS))
            heard = await self.transport.exchange(out)
            self.now += FRAME_MS
            self.caller_audio.append(out.samples)
            self.agent_audio.append(heard.samples)

            if self.caller_turn and not self.outbox:
                self.caller_turn.end_ms = self.now
                self.caller_turn = None
                self.waiting_since = self.now

            for event in self.transport.drain_events():
                self.rec.events.append((self.now, event))
                self.agent_hung_up |= isinstance(event, AgentHangup)

            event, utterance = self.ears.feed(heard)

            if event is VadEvent.SPEECH_START:
                self.agent_start = self.now - self.ears.vad.start_ms
                if self.rec.first_speech_ms is None:
                    self.rec.first_speech_ms = self.agent_start
                elif self.waiting_since is not None:
                    self.rec.latencies_ms.append(self.agent_start - self.waiting_since)
                self.waiting_since = None
                planned = self.caller.interruption_for(self.agent_turns)
                if planned:
                    self.barge_due = (self.agent_start + planned[0], planned[1])

            if self.barge_due and self.ears.speaking and self.now >= self.barge_due[0]:
                self.barge = BargeIn(self.agent_turns, self.now)
                self.rec.barge_ins.append(self.barge)
                await self.say(self.barge_due[1])
                self.barge_due = None

            if event is VadEvent.SPEECH_END:
                end = self.now - self.ears.vad.end_ms
                interrupted = self.barge is not None and self.barge.agent_turn == self.agent_turns
                if interrupted and self.barge:
                    self.barge.agent_stopped_ms = end
                text = await self.caller.hear(utterance)
                self.rec.turns.append(Turn("agent", text, self.agent_start, end, interrupted))
                self.agent_turns += 1
                self.barge_due = None
                if self.agent_hung_up:
                    return "agent"
                if self.agent_turns >= self.limits.max_turns:
                    return "max_turns"
                if interrupted:
                    continue  # the caller already spoke over it
                digits = self.caller.dtmf_for(text)
                if digits:
                    await self.transport.send_dtmf(digits)
                    self.waiting_since = self.now
                    continue
                reply = await self.caller.reply(text)
                if reply is None:
                    return "caller"
                await self.say(reply)

            timeout = self.limits.response_timeout_s * 1000
            if self.waiting_since is not None and self.now - self.waiting_since > timeout:
                return "agent" if self.agent_hung_up else "timeout"
        return "max_duration"


async def place_call(caller: Caller, transport: Transport) -> CallRecord:
    return await _Call(caller, transport).run()
