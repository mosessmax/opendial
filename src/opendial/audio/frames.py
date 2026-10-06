"""Audio frames. Mono 16-bit PCM throughout."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

PCM = npt.NDArray[np.int16]

SAMPLE_RATE = 16_000
FRAME_MS = 20


@dataclass(frozen=True)
class AudioFrame:
    samples: PCM
    sample_rate: int = SAMPLE_RATE
    text: str | None = None
    """Ground-truth text riding along with synthetic speech.

    Only the fake TTS/STT providers use it, so the whole pipeline can be
    tested without real speech models. Real providers ignore it.
    """

    @property
    def duration_ms(self) -> float:
        return len(self.samples) * 1000 / self.sample_rate

    def with_samples(self, samples: PCM) -> AudioFrame:
        return replace(self, samples=samples)


def silence(ms: float, sample_rate: int = SAMPLE_RATE) -> PCM:
    return np.zeros(round(ms * sample_rate / 1000), dtype=np.int16)


def to_frames(
    pcm: PCM, sample_rate: int = SAMPLE_RATE, frame_ms: int = FRAME_MS, text: str | None = None
) -> list[AudioFrame]:
    """Split PCM into fixed-size frames, zero-padding the last. `text` goes on the first."""
    n = sample_rate * frame_ms // 1000
    pcm = np.concatenate([pcm, np.zeros(-len(pcm) % n, dtype=np.int16)])
    frames = [AudioFrame(pcm[i : i + n], sample_rate) for i in range(0, len(pcm), n)]
    if frames and text is not None:
        frames[0] = replace(frames[0], text=text)
    return frames


def rms_dbfs(pcm: PCM) -> float:
    if len(pcm) == 0:
        return -120.0
    rms = np.sqrt(np.mean(pcm.astype(np.float64) ** 2))
    return float(20 * np.log10(max(rms, 1e-9) / 32768))
