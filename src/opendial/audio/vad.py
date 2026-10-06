"""Energy VAD used to split the agent's audio into turns.

Agent audio is TTS over a clean link, so an energy gate with hangover is
enough. A model-based VAD can replace it by implementing `feed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from opendial.audio.frames import AudioFrame, rms_dbfs


class VadEvent(Enum):
    SPEECH_START = "speech_start"
    SPEECH_END = "speech_end"


@dataclass
class EnergyVad:
    threshold_dbfs: float = -45.0
    start_ms: float = 60
    """Voiced audio needed before speech starts (ignores clicks)."""
    end_ms: float = 600
    """Silence needed before speech ends (the endpointing window)."""

    speaking: bool = False
    _voiced: float = 0.0
    _silent: float = 0.0

    def feed(self, frame: AudioFrame) -> VadEvent | None:
        if rms_dbfs(frame.samples) > self.threshold_dbfs:
            self._voiced += frame.duration_ms
            self._silent = 0.0
            if not self.speaking and self._voiced >= self.start_ms:
                self.speaking = True
                return VadEvent.SPEECH_START
        else:
            self._silent += frame.duration_ms
            if not self.speaking:
                self._voiced = 0.0
            elif self._silent >= self.end_ms:
                self.speaking, self._voiced = False, 0.0
                return VadEvent.SPEECH_END
        return None
