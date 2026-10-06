"""Write call outcomes as JUnit XML (for CI), JSON and WAV (for replay) and HTML (for people)."""

from __future__ import annotations

import html
import json
import wave
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opendial.assertions import Result
from opendial.record import CallRecord


@dataclass
class Outcome:
    record: CallRecord
    results: list[Result]

    @property
    def name(self) -> str:
        return self.record.scenario.name

    @property
    def passed(self) -> bool:
        return all(r.ok for r in self.results)


def write_wav(rec: CallRecord, path: Path) -> None:
    """Stereo WAV: caller on the left, agent on the right."""
    n = max(len(rec.caller_audio), len(rec.agent_audio))
    stereo = np.zeros((n, 2), dtype=np.int16)
    stereo[: len(rec.caller_audio), 0] = rec.caller_audio
    stereo[: len(rec.agent_audio), 1] = rec.agent_audio
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(rec.sample_rate)
        w.writeframes(stereo.tobytes())


def to_dict(o: Outcome) -> dict[str, Any]:
    rec = o.record
    return {
        "scenario": rec.scenario.model_dump(mode="json"),
        "passed": o.passed,
        "results": [r.__dict__ for r in o.results],
        "ended_by": rec.ended_by,
        "caller_finished": rec.caller_finished,
        "duration_ms": rec.duration_ms,
        "first_speech_ms": rec.first_speech_ms,
        "latencies_ms": rec.latencies_ms,
        "turns": [t.__dict__ for t in rec.turns],
        "tool_calls": [{"name": t.name, "args": t.args} for t in rec.tool_calls],
        "agent_heard": rec.agent_heard,
        "barge_ins": [{**b.__dict__, "stop_ms": b.stop_ms} for b in rec.barge_ins],
    }


def write_junit(outcomes: Sequence[Outcome], path: Path) -> None:
    failures = sum(not o.passed for o in outcomes)
    suite = ET.Element(
        "testsuite", name="opendial", tests=str(len(outcomes)), failures=str(failures)
    )
    for o in outcomes:
        case = ET.SubElement(
            suite, "testcase", classname="opendial", name=o.name,
            time=f"{o.record.duration_ms / 1000:.2f}",
        )  # fmt: skip
        failed = [r for r in o.results if not r.ok]
        if failed:
            message = "; ".join(f"{r.criterion}: {r.detail}" for r in failed)
            ET.SubElement(case, "failure", message=message).text = o.record.transcript()
        ET.SubElement(case, "system-out").text = o.record.transcript()
    ET.ElementTree(suite).write(path, encoding="utf-8", xml_declaration=True)


_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>opendial report</title>
<style>
  body {{ font: 15px/1.5 system-ui, sans-serif; max-width: 860px; margin: 2rem auto;
         padding: 0 1rem; color: #1a1a1a; background: #fff; }}
  @media (prefers-color-scheme: dark) {{ body {{ color: #e8e8e8; background: #141414; }}
    .call {{ border-color: #333; }} .turn.agent {{ background: #1f2933; }} }}
  h1 {{ font-size: 1.4rem; }} .call {{ border: 1px solid #ddd; border-radius: 8px;
         padding: 1rem 1.25rem; margin: 1rem 0; }}
  .pass {{ color: #1a7f37; }} .fail {{ color: #cf222e; }} .skip {{ color: #9a6700; }}
  .turn {{ padding: .25rem .5rem; border-radius: 4px; margin: .2rem 0; }}
  .turn.agent {{ background: #f0f4f8; }} .t {{ color: #888; font-size: .8em; margin-right: .5em; }}
  audio {{ width: 100%; margin: .5rem 0; }} ul {{ padding-left: 1.2rem; }}
</style></head><body>
<h1>opendial: {passed}/{total} calls passed</h1>
{calls}
</body></html>
"""

_STATUS = {"passed": "pass", "failed": "fail", "skipped": "skip"}


def _call_html(o: Outcome, wav: str) -> str:
    e = html.escape
    rec = o.record
    results = "".join(
        f'<li class="{_STATUS[r.status]}">{e(r.criterion)}: {r.status}. {e(r.detail)}</li>'
        for r in o.results
    )
    turns = "".join(
        f'<div class="turn {t.speaker}"><span class="t">{t.start_ms / 1000:.1f}s</span>'
        f"<b>{t.speaker}</b>{' (interrupted)' if t.interrupted else ''}: {e(t.text)}</div>"
        for t in rec.turns
    )
    verdict = '<span class="pass">passed</span>' if o.passed else '<span class="fail">failed</span>'
    return (
        f'<section class="call"><h2>{e(o.name)}: {verdict}</h2>'
        f"<p>{e(rec.scenario.goal)}<br>Ended by {rec.ended_by} after "
        f"{rec.duration_ms / 1000:.1f}s. Audio: caller left, agent right.</p>"
        f'<audio controls preload="none" src="{e(wav)}"></audio>'
        f"<ul>{results}</ul>{turns}</section>"
    )


def write_report(outcomes: Sequence[Outcome], out_dir: Path) -> Path:
    """Write calls/<name>.json and .wav, junit.xml and index.html. Returns the HTML path."""
    calls = out_dir / "calls"
    calls.mkdir(parents=True, exist_ok=True)
    sections = []
    for o in outcomes:
        write_wav(o.record, calls / f"{o.name}.wav")
        (calls / f"{o.name}.json").write_text(json.dumps(to_dict(o), indent=2))
        sections.append(_call_html(o, f"calls/{o.name}.wav"))
    write_junit(outcomes, out_dir / "junit.xml")
    page = _PAGE.format(
        passed=sum(o.passed for o in outcomes), total=len(outcomes), calls="".join(sections)
    )
    index = out_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index
