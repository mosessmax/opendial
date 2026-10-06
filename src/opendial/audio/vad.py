"""Energy VAD used to split the agent's audio into turns.

Agent audio is TTS over a clean link, so an energy gate with hangover is
enough. A model-based VAD can replace it by implementing `feed`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
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


@dataclass
class Segmenter:
    """Cuts a frame stream into utterances, keeping a little audio from before the VAD fired."""

    vad: EnergyVad = field(default_factory=EnergyVad)
    preroll_frames: int = 10
    _preroll: deque[AudioFrame] = field(init=False)
    _current: list[AudioFrame] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._preroll = deque(maxlen=self.preroll_frames)

    @property
    def speaking(self) -> bool:
        return self.vad.speaking

    def feed(self, frame: AudioFrame) -> tuple[VadEvent | None, list[AudioFrame]]:
        """Returns the VAD event, plus the whole utterance when one just ended."""
        event = self.vad.feed(frame)
        if event is VadEvent.SPEECH_START:
            self._current = [*self._preroll, frame]
        elif self.vad.speaking or event is VadEvent.SPEECH_END:
            self._current.append(frame)
        else:
            self._preroll.append(frame)
        if event is VadEvent.SPEECH_END:
            done, self._current = self._current, []
            self._preroll.clear()
            return event, done
        return event, []
