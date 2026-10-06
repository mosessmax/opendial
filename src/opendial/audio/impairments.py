"""Line impairments applied to the caller's audio: noise, band limit, codec, loss.

Each impairment takes an utterance (a list of frames) and returns one of the
same shape. All randomness is seeded so a failing call replays exactly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import numpy.typing as npt

from opendial.audio.frames import AudioFrame
from opendial.scenario import Audio, Noise, PacketLoss

Signal = npt.NDArray[np.float64]


class Impairment(Protocol):
    def apply(self, frames: Sequence[AudioFrame]) -> list[AudioFrame]: ...


def _join(frames: Sequence[AudioFrame]) -> Signal:
    return np.concatenate([f.samples for f in frames]).astype(np.float64) / 32768


def _split(frames: Sequence[AudioFrame], x: Signal) -> list[AudioFrame]:
    pcm = np.clip(np.round(x * 32768), -32768, 32767).astype(np.int16)
    bounds = np.cumsum([0] + [len(f.samples) for f in frames])
    return [f.with_samples(pcm[a:b]) for f, a, b in zip(frames, bounds, bounds[1:], strict=False)]


def _bandpass(x: Signal, sample_rate: int, lo: float, hi: float) -> Signal:
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x), 1 / sample_rate)
    spec[(freqs < lo) | (freqs > hi)] = 0
    return np.fft.irfft(spec, n=len(x))


def _unit(x: Signal) -> Signal:
    return x / (np.std(x) + 1e-12)


def _colored(rng: np.random.Generator, n: int, exponent: float) -> Signal:
    """Noise with power ~ 1/f^exponent: 0 white, 1 pink, 2 brown."""
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n)
    f[0] = f[1] if n > 1 else 1.0
    return _unit(np.fft.irfft(spec / f ** (exponent / 2), n=n))


def _babble(rng: np.random.Generator, n: int, sr: int, talkers: int = 6) -> Signal:
    """Crowd noise: speech-band noise with syllable-rate amplitude modulation."""
    t = np.arange(n) / sr
    out = np.zeros(n)
    for _ in range(talkers):
        voice = _bandpass(rng.standard_normal(n), sr, 200, 3000)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(3, 6) * t + rng.uniform(0, 6.3))
        out += voice * envelope**2
    return _unit(out)


def _market(rng: np.random.Generator, n: int, sr: int) -> Signal:
    """Open-air market: dense babble, low rumble, a sharp clatter about once a second."""
    out = _babble(rng, n, sr, talkers=10) + 0.4 * _colored(rng, n, 2)
    hit = sr // 20
    decay = np.exp(-np.linspace(0, 8, hit))
    for start in rng.integers(0, max(1, n - hit), size=max(1, n // sr)):
        out[start : start + hit] += 4 * rng.standard_normal(hit) * decay
    return _unit(out)


def _traffic(rng: np.random.Generator, n: int, sr: int) -> Signal:
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * np.arange(n) / sr)
    return _unit(_colored(rng, n, 2) * swell)


@dataclass
class AddNoise:
    noise: Noise
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def _generate(self, n: int, sr: int) -> Signal:
        kind = self.noise.kind
        if kind == "white":
            return _colored(self.rng, n, 0)
        if kind == "pink":
            return _colored(self.rng, n, 1)
        return {"babble": _babble, "market": _market, "traffic": _traffic}[kind](self.rng, n, sr)

    def apply(self, frames: Sequence[AudioFrame]) -> list[AudioFrame]:
        if not frames:
            return []
        x = _join(frames)
        voiced = x[np.abs(x) > 1e-4]
        level = np.sqrt(np.mean(voiced**2)) if len(voiced) else 0.03
        gain = level / 10 ** (self.noise.snr_db / 20)
        return _split(frames, x + gain * self._generate(len(x), frames[0].sample_rate))


@dataclass
class Narrowband:
    """Telephone band, 300-3400 Hz."""

    lo: float = 300
    hi: float = 3400

    def apply(self, frames: Sequence[AudioFrame]) -> list[AudioFrame]:
        if not frames:
            return []
        return _split(frames, _bandpass(_join(frames), frames[0].sample_rate, self.lo, self.hi))


@dataclass
class MuLaw:
    """mu-law round trip. Fewer `bits` approximates harsher mobile codecs."""

    bits: int = 8
    mu: float = 255

    def apply(self, frames: Sequence[AudioFrame]) -> list[AudioFrame]:
        if not frames:
            return []
        x = np.clip(_join(frames), -1, 1)
        y = np.sign(x) * np.log1p(self.mu * np.abs(x)) / np.log1p(self.mu)
        levels = 2 ** (self.bits - 1)
        y = np.round(y * levels) / levels
        return _split(frames, np.sign(y) * np.expm1(np.abs(y) * np.log1p(self.mu)) / self.mu)


@dataclass
class DropFrames:
    """Bursty frame loss (Gilbert-Elliott). Lost frames become silence, no concealment."""

    loss: PacketLoss
    seed: int = 0
    rng: np.random.Generator = field(init=False, repr=False)
    bad: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def apply(self, frames: Sequence[AudioFrame]) -> list[AudioFrame]:
        if self.loss.rate <= 0:
            return list(frames)
        recover = 1 / self.loss.burst
        fail = self.loss.rate * recover / (1 - self.loss.rate)
        out = []
        for f in frames:
            self.bad = self.rng.random() >= recover if self.bad else self.rng.random() < fail
            out.append(f.with_samples(np.zeros_like(f.samples)) if self.bad else f)
        return out


@dataclass
class Chain:
    steps: list[Impairment] = field(default_factory=list)

    def apply(self, frames: Sequence[AudioFrame]) -> list[AudioFrame]:
        out = list(frames)
        for step in self.steps:
            out = step.apply(out)
        return out


def build_chain(audio: Audio, seed: int = 0) -> Chain:
    """Room noise reaches the mic, the phone band-limits and encodes, the network drops."""
    steps: list[Impairment] = []
    if audio.noise:
        steps.append(AddNoise(audio.noise, seed))
    if audio.narrowband or audio.codec == "gsm-like":
        steps.append(Narrowband())
    if audio.codec in ("mulaw", "gsm-like"):
        steps.append(MuLaw(bits=8 if audio.codec == "mulaw" else 6))
    if audio.packet_loss and audio.packet_loss.rate > 0:
        steps.append(DropFrames(audio.packet_loss, seed + 1))
    return Chain(steps)
