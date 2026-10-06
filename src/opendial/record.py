"""CallRecord: everything that happened on one call. Assertions and reports read only this."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from opendial.audio.frames import PCM, SAMPLE_RATE
from opendial.scenario import Scenario
from opendial.transports import AgentHeard, Event, ToolCall

EndReason = Literal["caller", "agent", "timeout", "max_turns", "max_duration"]


@dataclass
class Turn:
    speaker: Literal["caller", "agent"]
    text: str
    start_ms: float
    end_ms: float
    interrupted: bool = False


@dataclass
class BargeIn:
    agent_turn: int
    at_ms: float
    agent_stopped_ms: float | None = None

    @property
    def stop_ms(self) -> float | None:
        return None if self.agent_stopped_ms is None else self.agent_stopped_ms - self.at_ms


@dataclass
class CallRecord:
    scenario: Scenario
    turns: list[Turn] = field(default_factory=list)
    events: list[tuple[float, Event]] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)
    """Caller stops talking -> agent starts talking, per response."""
    first_speech_ms: float | None = None
    """Call connected -> agent's first speech."""
    barge_ins: list[BargeIn] = field(default_factory=list)
    ended_by: EndReason = "caller"
    caller_finished: bool = False
    """Whether the caller had said everything it meant to when the call ended."""
    duration_ms: float = 0.0
    caller_audio: PCM = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    agent_audio: PCM = field(default_factory=lambda: np.zeros(0, dtype=np.int16))
    sample_rate: int = SAMPLE_RATE

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [e for _, e in self.events if isinstance(e, ToolCall)]

    @property
    def agent_heard(self) -> list[str]:
        return [e.text for _, e in self.events if isinstance(e, AgentHeard)]

    def transcript(self) -> str:
        return "\n".join(f"{t.speaker.upper()}: {t.text}" for t in self.turns)
