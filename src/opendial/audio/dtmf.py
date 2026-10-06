"""In-band DTMF tones, for transports without out-of-band DTMF."""

from __future__ import annotations

import numpy as np

from opendial.audio.frames import PCM, SAMPLE_RATE, silence

_ROWS = (697, 770, 852, 941)
_COLS = (1209, 1336, 1477, 1633)
_KEYS = ("123A", "456B", "789C", "*0#D")
FREQS = {k: (_ROWS[r], _COLS[c]) for r, row in enumerate(_KEYS) for c, k in enumerate(row)}


def dtmf_tones(
    digits: str, sample_rate: int = SAMPLE_RATE, tone_ms: int = 100, gap_ms: int = 80
) -> PCM:
    t = np.arange(sample_rate * tone_ms // 1000) / sample_rate
    parts = []
    for d in digits.upper():
        if d not in FREQS:
            raise ValueError(f"not a DTMF key: {d!r}")
        lo, hi = FREQS[d]
        tone = 0.25 * (np.sin(2 * np.pi * lo * t) + np.sin(2 * np.pi * hi * t)) / 2
        parts += [(tone * 32767).astype(np.int16), silence(gap_ms, sample_rate)]
    return np.concatenate(parts) if parts else silence(0, sample_rate)
