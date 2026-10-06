import asyncio

from opendial.audio import EnergyVad, VadEvent
from opendial.providers import FakeSTT, FakeTTS, ScriptedLLM


def run(coro):
    return asyncio.run(coro)


def test_scripted_llm_in_order_then_ends():
    llm = ScriptedLLM(["hello", "bye"])
    assert [run(llm.complete([])) for _ in range(3)] == ["hello", "bye", "[END]"]
    assert len(llm.calls) == 3


def test_scripted_llm_with_function():
    llm = ScriptedLLM(lambda messages: f"seen {len(messages)}")
    assert run(llm.complete([{"role": "user", "content": "hi"}])) == "seen 1"


def test_fake_tts_round_trips_through_fake_stt():
    frames = run(FakeTTS().synthesize("abeg where my package dey"))
    assert sum(f.duration_ms for f in frames) >= 1500
    assert run(FakeSTT().transcribe(frames)) == "abeg where my package dey"


def test_fake_speech_triggers_vad():
    vad = EnergyVad()
    frames = run(FakeTTS().synthesize("hello there"))
    assert VadEvent.SPEECH_START in [vad.feed(f) for f in frames]
