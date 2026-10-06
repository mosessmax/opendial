import numpy as np
import pytest

from opendial.audio import to_frames
from opendial.audio.frames import SAMPLE_RATE
from opendial.audio.impairments import AddNoise, DropFrames, MuLaw, Narrowband, build_chain
from opendial.scenario import Audio, Noise, PacketLoss


def speechish(seconds=2.0):
    t = np.arange(int(SAMPLE_RATE * seconds)) / SAMPLE_RATE
    x = 0.2 * np.sin(2 * np.pi * 1000 * t) + 0.1 * np.sin(2 * np.pi * 5000 * t)
    return to_frames((x * 32767).astype(np.int16), text="hi")


def joined(frames):
    return np.concatenate([f.samples for f in frames]).astype(float)


def power_at(x, hz):
    spec = np.abs(np.fft.rfft(x))
    return spec[np.argmin(np.abs(np.fft.rfftfreq(len(x), 1 / SAMPLE_RATE) - hz))]


@pytest.mark.parametrize("kind", ["white", "pink", "babble", "market", "traffic"])
def test_noise_hits_target_snr(kind):
    clean = speechish()
    noisy = AddNoise(Noise(kind=kind, snr_db=10)).apply(clean)
    signal, noise = joined(clean), joined(noisy) - joined(clean)
    snr = 10 * np.log10(np.mean(signal**2) / np.mean(noise**2))
    assert snr == pytest.approx(10, abs=1)


def test_seeded_noise_is_deterministic():
    a = AddNoise(Noise(kind="market"), seed=7).apply(speechish())
    b = AddNoise(Noise(kind="market"), seed=7).apply(speechish())
    assert np.array_equal(joined(a), joined(b))


def test_shape_and_text_are_kept():
    clean = speechish()
    out = build_chain(Audio(noise=Noise(), codec="gsm-like")).apply(clean)
    assert [len(f.samples) for f in out] == [len(f.samples) for f in clean]
    assert out[0].text == "hi"


def test_narrowband_removes_high_frequencies():
    x = joined(Narrowband().apply(speechish()))
    assert power_at(x, 5000) < 0.01 * power_at(x, 1000)


def test_mulaw_is_close_but_lossy():
    clean = joined(speechish())
    x = joined(MuLaw().apply(speechish()))
    assert not np.array_equal(x, clean)
    assert np.mean((x - clean) ** 2) < 0.01 * np.mean(clean**2)


def test_drop_frames_matches_rate_and_bursts():
    frames = speechish(seconds=60)
    out = DropFrames(PacketLoss(rate=0.1, burst=4)).apply(frames)
    dropped = np.array([not f.samples.any() for f in out])
    assert dropped.mean() == pytest.approx(0.1, abs=0.03)
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, dropped, 0])))[::2]
    assert runs.mean() == pytest.approx(4, abs=1)


def test_chain_order():
    chain = build_chain(Audio(noise=Noise(), codec="gsm-like", packet_loss=PacketLoss(rate=0.05)))
    names = [type(s).__name__ for s in chain.steps]
    assert names == ["AddNoise", "Narrowband", "MuLaw", "DropFrames"]
