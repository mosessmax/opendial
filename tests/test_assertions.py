import asyncio

import pytest

from opendial.assertions import evaluate, wer
from opendial.providers import ScriptedLLM
from opendial.record import BargeIn, CallRecord, Turn
from opendial.scenario import Scenario
from opendial.transports import AgentHeard, ToolCall


def record(success, accent=None, **fields):
    scenario = Scenario(name="t", goal="Move delivery", persona={"accent": accent}, success=success)
    return CallRecord(scenario, **fields)


def check(rec, judge=None):
    return [(r.criterion, r.status) for r in asyncio.run(evaluate(rec, judge))]


def test_wer():
    assert wer("a b c d".split(), "a b c d".split()) == 0
    assert wer("a b c d".split(), "a x c".split()) == 0.5
    assert wer([], []) == 0


def test_time_to_first_speech_uses_percentile():
    rec = record(
        [{"type": "time_to_first_speech", "max_ms": 1000, "stat": "max"}],
        first_speech_ms=400,
        latencies_ms=[800, 1200],
    )
    assert check(rec) == [("time_to_first_speech", "failed")]


def test_tool_called_matches_args_subset_and_counts():
    calls = [(1, ToolCall("reschedule", {"day": "sat", "id": 7})), (2, ToolCall("lookup"))]
    ok = record(
        [{"type": "tool_called", "name": "reschedule", "args": {"day": "sat"}}], events=calls
    )
    too_many = record([{"type": "tool_called", "name": "lookup", "max_times": 0}], events=calls)
    assert check(ok) == [("tool_called", "passed")]
    assert check(too_many) == [("tool_called", "failed")]


def test_interruption_handled():
    crit = [{"type": "interruption_handled", "max_stop_ms": 500}]
    assert check(record(crit)) == [("interruption_handled", "failed")]
    fast = record(crit, barge_ins=[BargeIn(0, 1000, 1300)])
    slow = record(crit, barge_ins=[BargeIn(0, 1000, 1900)])
    assert check(fast) == [("interruption_handled", "passed")]
    assert check(slow) == [("interruption_handled", "failed")]


def test_transcription_accuracy_forgives_accent_spellings():
    rec = record(
        [{"type": "transcription_accuracy", "max_wer": 0.0}],
        accent="ng-pidgin",
        turns=[Turn("caller", "Abeg, where my package dey?", 0, 1)],
        events=[(1, AgentHeard("a beg where my package dey"))],
    )
    assert check(rec) == [("transcription_accuracy", "passed")]
    assert check(record([{"type": "transcription_accuracy", "max_wer": 0.1}])) == [
        ("transcription_accuracy", "skipped")
    ]


@pytest.mark.parametrize(("ended_by", "finished", "status"), [
    ("agent", False, "failed"),
    ("agent", True, "passed"),
    ("caller", False, "passed"),
])  # fmt: skip
def test_no_agent_hangup(ended_by, finished, status):
    rec = record([{"type": "no_agent_hangup"}], ended_by=ended_by, caller_finished=finished)
    assert check(rec) == [("no_agent_hangup", status)]


def test_task_completed_with_judge():
    crit = [{"type": "task_completed", "judge": "Delivery moved to Saturday"}]
    yes = ScriptedLLM(['Sure: {"passed": true, "reason": "moved"}'])
    garbage = ScriptedLLM(["I think so?"])
    assert check(record(crit)) == [("task_completed", "skipped")]
    assert check(record(crit), yes) == [("task_completed", "passed")]
    assert check(record(crit), garbage) == [("task_completed", "failed")]
    assert "Delivery moved to Saturday" in yes.calls[0][0]["content"]
