"""How opendial reaches the agent under test.

The runner exchanges audio in lockstep: every tick it sends one caller frame
and receives one agent frame (silence when the agent is quiet). Live
transports pace this in real time; the loopback transport runs as fast as
the CPU allows, which keeps tests quick and deterministic.

Anything the agent reports besides audio (tool calls, what its own STT
heard, hanging up) comes back as events.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from opendial.audio.frames import AudioFrame


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AgentHeard:
    """What the agent's STT transcribed from the caller's last turn."""

    text: str


@dataclass(frozen=True)
class AgentHangup:
    pass


Event = ToolCall | AgentHeard | AgentHangup


class Transport(Protocol):
    async def connect(self) -> None: ...

    async def exchange(self, frame: AudioFrame) -> AudioFrame:
        """Send one caller frame, return the agent frame for the same tick."""
        ...

    async def send_dtmf(self, digits: str) -> None: ...

    def drain_events(self) -> list[Event]: ...

    async def close(self) -> None: ...
