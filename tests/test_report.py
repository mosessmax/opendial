import json
import wave
import xml.etree.ElementTree as ET

import numpy as np

from opendial.assertions import Result
from opendial.record import CallRecord, Turn
from opendial.report import Outcome, write_report
from opendial.scenario import Scenario


def outcome(name, status):
    rec = CallRecord(
        Scenario(name=name, goal="Move <delivery>"),
        turns=[Turn("agent", "Hello & welcome", 0, 1000), Turn("caller", "hi", 1200, 1500)],
        duration_ms=2000,
        caller_audio=np.ones(32000, dtype=np.int16),
        agent_audio=np.full(16000, 2, dtype=np.int16),
    )
    return Outcome(rec, [Result("tool_called", status, "lookup matched 0 time(s)")])


def test_write_report(tmp_path):
    index = write_report([outcome("good", "passed"), outcome("bad", "failed")], tmp_path)

    page = index.read_text()
    assert "1/2 calls passed" in page
    assert "Hello &amp; welcome" in page and "Move &lt;delivery&gt;" in page

    suite = ET.parse(tmp_path / "junit.xml").getroot()
    assert suite.get("tests") == "2" and suite.get("failures") == "1"
    assert suite.find("testcase[@name='bad']/failure") is not None
    assert suite.find("testcase[@name='good']/failure") is None

    data = json.loads((tmp_path / "calls" / "bad.json").read_text())
    assert data["passed"] is False and data["turns"][0]["speaker"] == "agent"

    with wave.open(str(tmp_path / "calls" / "good.wav")) as w:
        assert (w.getnchannels(), w.getnframes()) == (2, 32000)
        frames = np.frombuffer(w.readframes(2), dtype=np.int16)
        assert frames.tolist() == [1, 2, 1, 2]
