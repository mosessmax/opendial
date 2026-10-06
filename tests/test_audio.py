import numpy as np
import pytest

from opendial.audio import EnergyVad, VadEvent, dtmf_tones, silence, to_frames
from opendial.audio.dtmf import FREQS
from opendial.audio.frames import SAMPLE_RATE


def tone(ms, hz=440, level=0.3):
    t = np.arange(SAMPLE_RATE * ms // 1000) / SAMPLE_RATE
    return (level * 32767 * np.sin(2 * np.pi * hz * t)).astype(np.int16)


def test_to_frames_pads_and_tags_first_frame():
    frames = to_frames(tone(50), text="hello")
    assert len(frames) == 3
    assert all(len(f.samples) == 320 for f in frames)
    assert frames[0].text == "hello" and frames[1].text is None


def test_vad_start_and_end():
    vad = EnergyVad(end_ms=200)
    pcm = np.concatenate([silence(100), tone(300), silence(300)])
    events = [(i, e) for i, f in enumerate(to_frames(pcm)) if (e := vad.feed(f))]
    assert [e for _, e in events] == [VadEvent.SPEECH_START, VadEvent.SPEECH_END]
    start, end = (i * 20 for i, _ in events)
    assert 140 <= start <= 180  # 100ms lead + 60ms start window
    assert 580 <= end <= 620  # 400ms + 200ms hangover


def test_vad_ignores_clicks():
    vad = EnergyVad()
    pcm = np.concatenate([tone(20), silence(200)])
    assert not any(vad.feed(f) for f in to_frames(pcm))


@pytest.mark.parametrize("key", ["5", "#"])
def test_dtmf_frequencies(key):
    pcm = dtmf_tones(key)[: SAMPLE_RATE // 10].astype(float)
    spectrum = np.abs(np.fft.rfft(pcm))
    freqs = np.fft.rfftfreq(len(pcm), 1 / SAMPLE_RATE)
    peaks = sorted(freqs[np.argsort(spectrum)[-2:]])
    assert np.allclose(peaks, FREQS[key], atol=10)


def test_dtmf_rejects_bad_key():
    with pytest.raises(ValueError):
        dtmf_tones("1x")
