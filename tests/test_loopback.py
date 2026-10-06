import asyncio

import numpy as np

from opendial.audio import silence, to_frames
from opendial.providers import FakeTTS
from opendial.transports import AgentHangup, AgentHeard, ToolCall
from opendial.transports.loopback import LoopbackTransport, Reply


def run(coro):
    return asyncio.run(coro)


async def echo_agent(text):
    if not text:
        return Reply("Hello, how can I help you today?")
    if text.startswith("[DTMF"):
        return Reply("Got your keys.", tools=[ToolCall("lookup", {"keys": text[6:-1]})])
    if "bye" in text:
        return Reply("Goodbye.", hangup=True)
    return Reply(f"You said {text}.")


def voiced(frame):
    return bool(np.abs(frame.samples).max() > 1000)


async def pump(transport, frames):
    return [await transport.exchange(f) for f in frames]


def quiet(ms):
    return to_frames(silence(ms))


def test_greeting_after_latency():
    async def go():
        t = LoopbackTransport(echo_agent, latency_ms=400)
        await t.connect()
        return [voiced(f) for f in await pump(t, quiet(1000))]

    out = run(go())
    assert out.index(True) * 20 == 400


def test_caller_turn_is_heard_and_answered():
    async def go():
        t = LoopbackTransport(echo_agent, greet=False, endpoint_ms=300)
        await t.connect()
        speech = await FakeTTS().synthesize("where is my parcel")
        out = await pump(t, speech + quiet(3000))
        return t.drain_events(), out

    events, out = run(go())
    assert events == [AgentHeard("where is my parcel")]
    assert any(voiced(f) and f.text == "You said where is my parcel." for f in out)


def test_barge_in_stops_agent():
    async def go():
        t = LoopbackTransport(echo_agent, latency_ms=0, stop_ms=200)
        await t.connect()
        speech = await FakeTTS().synthesize("wait wait wait")
        out = await pump(t, quiet(300) + speech)
        return [voiced(f) for f in out]

    out = run(go())
    talking = out[15:]  # agent was mid-greeting when the caller cut in at 300ms
    assert any(talking[:10]) and not any(talking[20:])


def test_dtmf_tool_calls_and_hangup():
    async def go():
        t = LoopbackTransport(echo_agent, greet=False, endpoint_ms=300)
        await t.connect()
        await t.send_dtmf("4417#")
        await pump(t, quiet(2000))
        bye = await FakeTTS().synthesize("ok bye")
        await pump(t, bye + quiet(2000))
        return t.drain_events()

    events = run(go())
    assert ToolCall("lookup", {"keys": "4417#"}) in events
    assert events[-1] == AgentHangup()
