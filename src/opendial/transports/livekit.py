"""LiveKit transport (stub): join a room as the caller and talk to an agent in it.

Requires `pip install "opendial[livekit]"`. Point opendial at `from_env` and set
LIVEKIT_URL, LIVEKIT_API_KEY and LIVEKIT_API_SECRET. Set OPENDIAL_AGENT_NAME to
dispatch a named LiveKit agent into each test room.

Audio is paced in real time. Events travel as JSON on the `opendial` data
topic; agents publish them with `event_payload` (see below), for example:

    await room.local_participant.publish_data(
        event_payload(ToolCall("reschedule", {"day": "sat"})), topic=TOPIC
    )

Not yet covered: reading LiveKit Agents' built-in transcription stream for
`AgentHeard`, reconnects, and multiple agent participants.
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from typing import Any

import numpy as np

from opendial.audio.frames import FRAME_MS, SAMPLE_RATE, AudioFrame, silence
from opendial.transports import AgentHangup, AgentHeard, Event, ToolCall

TOPIC = "opendial"
_DTMF_CODES = {
    **{str(d): d for d in range(10)},
    "*": 10,
    "#": 11,
    "A": 12,
    "B": 13,
    "C": 14,
    "D": 15,
}


def event_payload(event: Event) -> bytes:
    if isinstance(event, ToolCall):
        data: dict[str, Any] = {"type": "tool_call", "name": event.name, "args": event.args}
    elif isinstance(event, AgentHeard):
        data = {"type": "heard", "text": event.text}
    else:
        data = {"type": "hangup"}
    return json.dumps(data).encode()


def parse_event(payload: bytes) -> Event | None:
    try:
        data = json.loads(payload)
        kind = data["type"]
    except (ValueError, KeyError, TypeError):
        return None
    if kind == "tool_call":
        return ToolCall(str(data["name"]), dict(data.get("args") or {}))
    if kind == "heard":
        return AgentHeard(str(data["text"]))
    if kind == "hangup":
        return AgentHangup()
    return None


class FrameBuffer:
    """Re-chunks incoming audio of any frame size into 20 ms frames."""

    def __init__(self, sample_rate: int = SAMPLE_RATE, frame_ms: int = FRAME_MS):
        self.size = sample_rate * frame_ms // 1000
        self.sample_rate = sample_rate
        self.pending = np.zeros(0, dtype=np.int16)

    def push(self, samples: np.ndarray) -> None:
        self.pending = np.concatenate([self.pending, samples.astype(np.int16)])

    def pop(self) -> AudioFrame:
        if len(self.pending) < self.size:
            return AudioFrame(silence(FRAME_MS, self.sample_rate), self.sample_rate)
        out, self.pending = self.pending[: self.size], self.pending[self.size :]
        return AudioFrame(out, self.sample_rate)


class LiveKitTransport:
    def __init__(
        self,
        url: str,
        api_key: str,
        api_secret: str,
        *,
        room: str | None = None,
        agent_name: str | None = None,
        identity: str = "opendial-caller",
    ):
        try:
            from livekit import api, rtc
        except ImportError as e:  # pragma: no cover
            raise ImportError('LiveKit support needs: pip install "opendial[livekit]"') from e
        self._api, self._rtc = api, rtc
        self.url, self.api_key, self.api_secret = url, api_key, api_secret
        self.room_name = room or f"opendial-{uuid.uuid4().hex[:8]}"
        self.agent_name, self.identity = agent_name, identity
        self.incoming = FrameBuffer()
        self.events: list[Event] = []
        self._tasks: set[asyncio.Task[None]] = set()
        self._next_tick = 0.0

    async def connect(self) -> None:
        api, rtc = self._api, self._rtc
        grants = api.VideoGrants(room_join=True, room=self.room_name)
        token = (
            api.AccessToken(self.api_key, self.api_secret)
            .with_identity(self.identity)
            .with_grants(grants)
            .to_jwt()
        )
        self.room = rtc.Room()
        self.room.on("track_subscribed", self._on_track)
        self.room.on("data_received", self._on_data)
        self.room.on("participant_disconnected", lambda _p: self.events.append(AgentHangup()))
        await self.room.connect(self.url, token)

        self.source = rtc.AudioSource(SAMPLE_RATE, 1)
        track = rtc.LocalAudioTrack.create_audio_track("opendial-caller", self.source)
        options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
        await self.room.local_participant.publish_track(track, options)

        if self.agent_name:
            async with api.LiveKitAPI(self.url, self.api_key, self.api_secret) as lk:
                request = api.CreateAgentDispatchRequest(
                    agent_name=self.agent_name, room=self.room_name
                )
                await lk.agent_dispatch.create_dispatch(request)
        self._next_tick = asyncio.get_running_loop().time()

    def _on_track(self, track: Any, _publication: Any, _participant: Any) -> None:
        if track.kind == self._rtc.TrackKind.KIND_AUDIO:
            task = asyncio.ensure_future(self._read(track))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _read(self, track: Any) -> None:
        stream = self._rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=1)
        async for event in stream:
            self.incoming.push(np.frombuffer(event.frame.data, dtype=np.int16))

    def _on_data(self, packet: Any) -> None:
        if packet.topic == TOPIC and (event := parse_event(packet.data)):
            self.events.append(event)

    async def exchange(self, frame: AudioFrame) -> AudioFrame:
        rtc = self._rtc
        data = frame.samples.tobytes()
        await self.source.capture_frame(
            rtc.AudioFrame(data, frame.sample_rate, 1, len(frame.samples))
        )
        loop = asyncio.get_running_loop()
        self._next_tick += FRAME_MS / 1000
        await asyncio.sleep(max(0.0, self._next_tick - loop.time()))
        return self.incoming.pop()

    async def send_dtmf(self, digits: str) -> None:
        for digit in digits.upper():
            await self.room.local_participant.publish_dtmf(code=_DTMF_CODES[digit], digit=digit)

    def drain_events(self) -> list[Event]:
        events, self.events = self.events, []
        return events

    async def close(self) -> None:
        for task in self._tasks:
            task.cancel()
        await self.room.disconnect()


def from_env() -> LiveKitTransport:
    return LiveKitTransport(
        os.environ["LIVEKIT_URL"],
        os.environ["LIVEKIT_API_KEY"],
        os.environ["LIVEKIT_API_SECRET"],
        agent_name=os.environ.get("OPENDIAL_AGENT_NAME"),
    )
