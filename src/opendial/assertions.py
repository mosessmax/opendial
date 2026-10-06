"""Success criteria, evaluated against a CallRecord.

Every check except `task_completed` is a pure function of the record, so a
saved call can be re-scored against new criteria without placing it again.
`task_completed` asks an LLM judge and is skipped when no judge is given.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np

from opendial import packs
from opendial.providers import LLM
from opendial.record import CallRecord
from opendial.scenario import (
    Criterion,
    InterruptionHandled,
    NoAgentHangup,
    TaskCompleted,
    TimeToFirstSpeech,
    ToolCalled,
    TranscriptionAccuracy,
)


@dataclass(frozen=True)
class Result:
    criterion: str
    status: Literal["passed", "failed", "skipped"]
    detail: str

    @property
    def ok(self) -> bool:
        return self.status != "failed"


def _verdict(criterion: Criterion, passed: bool, detail: str) -> Result:
    return Result(criterion.type, "passed" if passed else "failed", detail)


def time_to_first_speech(c: TimeToFirstSpeech, rec: CallRecord) -> Result:
    samples = [rec.first_speech_ms] if rec.first_speech_ms is not None else []
    samples += rec.latencies_ms
    if not samples:
        return _verdict(c, False, "the agent never spoke")
    q = {"p50": 50, "p90": 90, "p95": 95, "max": 100}[c.stat]
    value = float(np.percentile(samples, q))
    return _verdict(c, value <= c.max_ms, f"{c.stat} {value:.0f} ms (limit {c.max_ms:.0f} ms)")


def interruption_handled(c: InterruptionHandled, rec: CallRecord) -> Result:
    if not rec.barge_ins:
        return _verdict(c, False, "no interruption happened; add one to the scenario")
    stops = [b.stop_ms for b in rec.barge_ins]
    if None in stops:
        return _verdict(c, False, "the agent kept talking through an interruption")
    worst = max(s for s in stops if s is not None)
    return _verdict(c, worst <= c.max_stop_ms, f"stopped within {worst:.0f} ms")


def tool_called(c: ToolCalled, rec: CallRecord) -> Result:
    matches = [
        t for t in rec.tool_calls
        if t.name == c.name and all(t.args.get(k) == v for k, v in c.args.items())
    ]  # fmt: skip
    n = len(matches)
    ok = n >= c.min_times and (c.max_times is None or n <= c.max_times)
    seen = ", ".join(t.name for t in rec.tool_calls) or "none"
    return _verdict(c, ok, f"{c.name} matched {n} time(s); calls seen: {seen}")


def _words(text: str, normalize: dict[str, str]) -> list[str]:
    text = re.sub(r"[^\w\s']", " ", text.lower())
    for variant, canonical in normalize.items():
        text = re.sub(rf"\b{re.escape(variant)}\b", canonical, text)
    return text.split()


def wer(reference: list[str], hypothesis: list[str]) -> float:
    """Word error rate: word-level edit distance over reference length."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    row = list(range(len(hypothesis) + 1))
    for i, ref in enumerate(reference, 1):
        prev, row[0] = row[0], i
        for j, hyp in enumerate(hypothesis, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (ref != hyp))
    return row[-1] / len(reference)


def transcription_accuracy(c: TranscriptionAccuracy, rec: CallRecord) -> Result:
    if not rec.agent_heard:
        return Result(c.type, "skipped", "the transport did not report the agent's transcripts")
    accent = rec.scenario.persona.accent
    normalize = packs.accent(accent).normalize if accent else {}
    said = " ".join(t.text for t in rec.turns if t.speaker == "caller")
    heard = " ".join(rec.agent_heard)
    value = wer(_words(said, normalize), _words(heard, normalize))
    return _verdict(c, value <= c.max_wer, f"WER {value:.2f} (limit {c.max_wer:.2f})")


def no_agent_hangup(c: NoAgentHangup, rec: CallRecord) -> Result:
    early = rec.ended_by == "agent" and not rec.caller_finished
    return _verdict(c, not early, "agent hung up early" if early else f"ended by {rec.ended_by}")


_JUDGE = """You are grading a phone call between a caller and a voice agent.

The caller's goal: {goal}
Pass the call only if this is true: {rubric}

Transcript:
{transcript}

Tool calls made by the agent: {tools}

Answer with JSON only: {{"passed": true or false, "reason": "one sentence"}}"""


async def task_completed(c: TaskCompleted, rec: CallRecord, judge: LLM | None) -> Result:
    if judge is None:
        return Result(c.type, "skipped", "no judge LLM configured")
    tools = ", ".join(f"{t.name}({json.dumps(t.args)})" for t in rec.tool_calls) or "none"
    prompt = _JUDGE.format(
        goal=rec.scenario.goal, rubric=c.judge, transcript=rec.transcript(), tools=tools
    )
    raw = await judge.complete([{"role": "user", "content": prompt}])
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    try:
        verdict = json.loads(match.group(0) if match else raw)
        return _verdict(c, bool(verdict["passed"]), str(verdict.get("reason", "")))
    except (ValueError, KeyError, TypeError):
        return _verdict(c, False, f"judge returned unreadable output: {raw[:200]!r}")


async def evaluate(rec: CallRecord, judge: LLM | None = None) -> list[Result]:
    results = []
    for c in rec.scenario.success:
        if isinstance(c, TaskCompleted):
            results.append(await task_completed(c, rec, judge))
        elif isinstance(c, TimeToFirstSpeech):
            results.append(time_to_first_speech(c, rec))
        elif isinstance(c, InterruptionHandled):
            results.append(interruption_handled(c, rec))
        elif isinstance(c, ToolCalled):
            results.append(tool_called(c, rec))
        elif isinstance(c, TranscriptionAccuracy):
            results.append(transcription_accuracy(c, rec))
        else:
            results.append(no_agent_hangup(c, rec))
    return results
