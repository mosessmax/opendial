import asyncio

from opendial.caller import END, Caller, system_prompt
from opendial.providers import FakeSTT, FakeTTS, ScriptedLLM
from opendial.scenario import Scenario


def run(coro):
    return asyncio.run(coro)


def make(llm=None, **fields):
    scenario = Scenario(name="t", goal="Move my delivery to Saturday", **fields)
    return Caller(scenario, llm or ScriptedLLM([]), FakeTTS(), FakeSTT())


def test_prompt_carries_goal_context_and_accent():
    scenario = Scenario(
        name="t",
        goal="Move my delivery to Saturday",
        persona={"name": "Chidi", "accent": "ng-pidgin", "traits": ["impatient"]},
        context={"order_id": "4417"},
    )
    prompt = system_prompt(scenario)
    for expected in ("Chidi", "Saturday", "impatient", "order_id: 4417", "Pidgin", END):
        assert expected in prompt


def test_llm_caller_keeps_history():
    llm = ScriptedLLM(["Hi, I need to move a delivery."])
    caller = make(llm)
    assert run(caller.reply("Hello, how can I help?")) == "Hi, I need to move a delivery."
    roles = [m["role"] for m in llm.calls[0]]
    assert roles == ["system", "user"]
    assert caller.history[-1]["role"] == "assistant"


def test_end_marker_says_goodbye_then_hangs_up():
    caller = make(ScriptedLLM([f"Thanks, bye. {END}"]))
    assert run(caller.reply("Done!")) == "Thanks, bye."
    assert caller.finished
    assert run(caller.reply("Anything else?")) is None


def test_bare_end_hangs_up_now():
    assert run(make(ScriptedLLM([END])).reply("hi")) is None


def test_script_mode_ignores_llm():
    llm = ScriptedLLM(["should not be used"])
    caller = make(llm, script=["one", "two"])
    assert run(caller.reply("x")) == "one" and not caller.finished
    assert run(caller.reply("x")) == "two" and caller.finished
    assert run(caller.reply("x")) is None
    assert llm.calls == []


def test_speak_goes_through_the_line():
    clean = run(FakeTTS().synthesize("hello there"))
    caller = make(audio={"line": "ng-market-call"})
    noisy = run(caller.speak("hello there"))
    assert len(noisy) == len(clean)
    assert any((a.samples != b.samples).any() for a, b in zip(noisy, clean, strict=True))
    assert run(caller.hear(noisy)) == "hello there"


def test_dtmf_and_interruptions():
    caller = make(
        dtmf=[{"when": "order number", "digits": "4417#"}],
        interruptions=[{"turn": 1, "after_ms": 500, "say": "Abeg wait"}],
    )
    assert caller.dtmf_for("Please key in your ORDER NUMBER") == "4417#"
    assert caller.dtmf_for("Hello") is None
    assert caller.interruption_for(1) == (500, "Abeg wait")
    assert caller.interruption_for(0) is None
