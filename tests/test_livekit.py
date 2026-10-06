import numpy as np
import pytest

from opendial.transports import AgentHangup, AgentHeard, ToolCall
from opendial.transports.livekit import FrameBuffer, event_payload, parse_event


@pytest.mark.parametrize(
    "event", [ToolCall("reschedule", {"day": "sat"}), AgentHeard("abeg"), AgentHangup()]
)
def test_event_round_trip(event):
    assert parse_event(event_payload(event)) == event


@pytest.mark.parametrize("payload", [b"not json", b"{}", b'{"type": "dance"}', b"[1]"])
def test_bad_payloads_are_ignored(payload):
    assert parse_event(payload) is None


def test_frame_buffer_rechunks_and_pads_with_silence():
    buf = FrameBuffer()
    buf.push(np.arange(480, dtype=np.int16))  # one 30 ms chunk
    first, second = buf.pop(), buf.pop()
    assert first.samples.tolist() == list(range(320))
    assert len(second.samples) == 320 and not second.samples.any()
    buf.push(np.arange(160, dtype=np.int16))
    assert buf.pop().samples[:160].tolist() == list(range(320, 480))


def test_livekit_sdk_names_exist():
    rtc = pytest.importorskip("livekit.rtc")
    for name in ("Room", "AudioSource", "LocalAudioTrack", "AudioStream", "TrackPublishOptions"):
        assert hasattr(rtc, name)
