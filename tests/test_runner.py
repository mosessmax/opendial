import asyncio

import pytest

from opendial.caller import Caller
from opendial.providers import FakeSTT, FakeTTS, ScriptedLLM
from opendial.runner import place_call
from opendial.scenario import Scenario
from opendial.transports import ToolCall
from opendial.transports.loopback import LoopbackTransport, Reply


async def agent(text):
    if not text:
        return Reply("Welcome to Kuro Express. How can I help?")
    if text.startswith("[DTMF"):
        return Reply("Thanks, I found your order.", tools=[ToolCall("lookup", {"id": "4417"})])
    if "saturday" in text.lower():
        return Reply("Done, moved to Saturday.", tools=[ToolCall("reschedule", {"day": "sat"})])
    if "bye" in text.lower():
        return Reply("Goodbye!", hangup=True)
    return Reply("Please enter your order number.")


def call(scenario_fields, llm=None, **transport):
    scenario = Scenario(name="t", goal="Move delivery to Saturday", **scenario_fields)
    caller = Caller(scenario, llm or ScriptedLLM([]), FakeTTS(), FakeSTT())
    return asyncio.run(place_call(caller, LoopbackTransport(agent, **transport)))


def test_full_scripted_call():
    rec = call(
        {
            "script": ["I want to change my delivery", "Saturday please", "Ok bye"],
            "dtmf": [{"when": "order number", "digits": "4417#"}],
        },
        latency_ms=400,
    )
    assert rec.ended_by == "agent"
    speakers = [t.speaker[0] for t in rec.turns]
    assert "".join(speakers) == "acaacaca"
    assert rec.turns[0].text.startswith("Welcome")
    assert [c.name for c in rec.tool_calls] == ["lookup", "reschedule"]
    assert rec.agent_heard[0] == "I want to change my delivery"
    assert rec.first_speech_ms == pytest.approx(400, abs=40)
    # spoken turns: 500 ms endpointing + 400 ms latency; the DTMF reply skips endpointing
    assert rec.latencies_ms == pytest.approx([900, 400, 900, 900], abs=40)
    assert len(rec.caller_audio) == len(rec.agent_audio) == rec.duration_ms * 16


def test_caller_hangs_up():
    rec = call({"script": ["Never mind"]})
    assert rec.ended_by == "caller"


def test_timeout_when_agent_is_silent():
    rec = call({"script": ["hello"], "limits": {"response_timeout_s": 2}}, greet=False)
    assert rec.ended_by == "timeout"
    assert rec.first_speech_ms is None


def test_barge_in_is_measured():
    rec = call(
        {
            "script": ["I want to change my delivery", "bye"],
            "interruptions": [{"turn": 0, "after_ms": 600, "say": "Saturday"}],
        },
        stop_ms=200,
    )
    barge = rec.barge_ins[0]
    assert rec.turns[0].interrupted
    assert barge.stop_ms is not None and barge.stop_ms < 500


def test_max_turns():
    rec = call({"script": ["hi"] * 10, "limits": {"max_turns": 2}})
    assert rec.ended_by == "max_turns"
