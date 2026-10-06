"""Shared plumbing for the CLI and the pytest plugin: find scenarios, build providers, run calls.

Targets and providers are given as `module:attribute` import paths, resolved
from the current directory:

* target: an async `agent(text) -> Reply` function (run in-process over the
  loopback transport) or a zero-argument factory returning a Transport.
* llm, tts, stt, judge: zero-argument factories returning a provider.
  TTS and STT default to the fakes, which only make sense with loopback.
"""

from __future__ import annotations

import importlib
import inspect
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opendial.assertions import evaluate
from opendial.caller import Caller
from opendial.providers import FakeSTT, FakeTTS, ScriptedLLM
from opendial.report import Outcome
from opendial.runner import place_call
from opendial.scenario import Scenario, load_scenario
from opendial.transports import Transport
from opendial.transports.loopback import LoopbackTransport

PATTERNS = ("*.scenario.yaml", "*.scenario.yml")


def find_scenarios(paths: Iterable[str | Path]) -> list[Path]:
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found += sorted(f for pattern in PATTERNS for f in p.rglob(pattern))
        elif p.exists():
            found.append(p)
        else:
            raise FileNotFoundError(p)
    return found


def load_object(path: str) -> Any:
    module, _, attr = path.partition(":")
    if not attr:
        raise ValueError(f"expected module:attribute, got {path!r}")
    if "" not in sys.path:
        sys.path.insert(0, "")
    obj: Any = importlib.import_module(module)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


@dataclass
class Config:
    target: str | None = None
    llm: str | None = None
    tts: str | None = None
    stt: str | None = None
    judge: str | None = None
    seed: int = 0

    def transport(self) -> Transport:
        if not self.target:
            raise ValueError("no target: pass --target module:attribute")
        obj = load_object(self.target)
        if inspect.iscoroutinefunction(obj):
            return LoopbackTransport(obj)
        transport: Transport = obj()
        return transport

    def _make(self, path: str | None, default: Any = None) -> Any:
        return load_object(path)() if path else default

    def caller(self, scenario: Scenario) -> Caller:
        llm = self._make(self.llm)
        if llm is None:
            if scenario.script is None:
                raise ValueError(f"{scenario.name}: no script and no caller LLM (--llm)")
            llm = ScriptedLLM([])
        tts = self._make(self.tts, FakeTTS())
        stt = self._make(self.stt, FakeSTT())
        return Caller(scenario, llm, tts, stt, seed=self.seed)


async def run_scenario(scenario: Scenario, config: Config) -> Outcome:
    record = await place_call(config.caller(scenario), config.transport())
    return Outcome(record, await evaluate(record, config._make(config.judge)))


def load_all(paths: Iterable[Path]) -> list[Scenario]:
    return [load_scenario(p) for p in paths]
