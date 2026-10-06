"""Scenario schema: one voice-agent test case, written as data.

A scenario says who is calling (persona), why (goal), over what kind of line
(audio), what disruptions happen (interruptions, DTMF), and what counts as a
pass (success). Usually written in YAML and loaded with `load_scenario`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Persona(_Model):
    name: str = "Caller"
    description: str = "A typical customer."
    language: str = "en"
    accent: str | None = None
    """Accent pack id, e.g. `ng-pidgin`."""
    traits: list[str] = []


class Noise(_Model):
    kind: Literal["white", "pink", "babble", "market", "traffic"] = "pink"
    snr_db: float = 20.0


class PacketLoss(_Model):
    rate: float = Field(0.0, ge=0.0, lt=1.0)
    burst: float = Field(1.0, ge=1.0)
    """Mean burst length in frames."""


class Audio(_Model):
    line: str | None = None
    """Line pack id, e.g. `ng-mobile-3g`. Fields set below override it."""
    narrowband: bool | None = None
    codec: Literal["none", "mulaw", "gsm-like"] | None = None
    noise: Noise | None = None
    packet_loss: PacketLoss | None = None


class Interruption(_Model):
    turn: int = Field(ge=0)
    """Agent turn to barge in on (0 is the greeting)."""
    after_ms: int = Field(800, ge=0)
    say: str


class Dtmf(_Model):
    when: str
    """Case-insensitive regex matched against what the agent says."""
    digits: str = Field(pattern=r"^[0-9*#A-D]+$")


class Limits(_Model):
    max_turns: int = Field(20, ge=1)
    max_duration_s: float = Field(300.0, gt=0)
    response_timeout_s: float = Field(10.0, gt=0)


# Success criteria. Each one has a `type` tag so YAML can list them freely.


class TaskCompleted(_Model):
    type: Literal["task_completed"]
    judge: str
    """Rubric an LLM judge checks the call against."""


class TimeToFirstSpeech(_Model):
    type: Literal["time_to_first_speech"]
    max_ms: float = Field(gt=0)
    stat: Literal["p50", "p90", "p95", "max"] = "p95"


class InterruptionHandled(_Model):
    type: Literal["interruption_handled"]
    max_stop_ms: float = Field(800, gt=0)


class ToolCalled(_Model):
    type: Literal["tool_called"]
    name: str
    args: dict[str, Any] = {}
    """Subset match on the call's arguments."""
    min_times: int = Field(1, ge=0)
    max_times: int | None = None


class TranscriptionAccuracy(_Model):
    type: Literal["transcription_accuracy"]
    max_wer: float = Field(ge=0.0)


class NoAgentHangup(_Model):
    type: Literal["no_agent_hangup"]


Criterion = Annotated[
    TaskCompleted
    | TimeToFirstSpeech
    | InterruptionHandled
    | ToolCalled
    | TranscriptionAccuracy
    | NoAgentHangup,
    Field(discriminator="type"),
]


class Scenario(_Model):
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    description: str = ""
    tags: list[str] = []
    persona: Persona = Persona()
    goal: str
    context: dict[str, Any] = {}
    """Facts the caller knows and may reveal (name, order id, ...)."""
    script: list[str] | None = None
    """Fixed caller lines instead of an LLM caller."""
    opening: Literal["agent_first", "caller_first"] = "agent_first"
    audio: Audio = Audio()
    interruptions: list[Interruption] = []
    dtmf: list[Dtmf] = []
    limits: Limits = Limits()
    success: list[Criterion] = []


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario from YAML. `name` defaults to the file's stem."""
    path = Path(path)
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a mapping at the top level")
    data.setdefault("name", path.name.split(".")[0])
    return Scenario.model_validate(data)
