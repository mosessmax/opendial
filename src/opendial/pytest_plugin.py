"""pytest plugin: every *.scenario.yaml becomes a test, and `opendial_call` runs one from Python.

Configure with command-line flags or in pyproject.toml:

    [tool.pytest.ini_options]
    opendial_target = "my_agent:agent"
    opendial_llm = "my_providers:caller_llm"
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from opendial.report import Outcome, write_report
from opendial.scenario import Scenario, load_scenario
from opendial.session import PATTERNS, Config, run_scenario

_OPTIONS = {
    "target": "agent or transport factory, module:attribute",
    "llm": "caller LLM factory",
    "tts": "caller TTS factory (default: fake)",
    "stt": "caller STT factory (default: fake)",
    "judge": "judge LLM factory for task_completed",
    "report": "write an HTML/JUnit report to this directory",
}
_OUTCOMES = pytest.StashKey[list[Outcome]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("opendial")
    for name, help_text in _OPTIONS.items():
        group.addoption(f"--opendial-{name}", dest=f"opendial_{name}", help=help_text)
        parser.addini(f"opendial_{name}", help_text)
    group.addoption("--opendial-seed", dest="opendial_seed", type=int, default=0)


def _option(config: pytest.Config, name: str) -> str | None:
    value = config.getoption(f"opendial_{name}") or config.getini(f"opendial_{name}")
    return str(value) if value else None


def _config(config: pytest.Config) -> Config:
    return Config(
        target=_option(config, "target"),
        llm=_option(config, "llm"),
        tts=_option(config, "tts"),
        stt=_option(config, "stt"),
        judge=_option(config, "judge"),
        seed=config.getoption("opendial_seed"),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.stash[_OUTCOMES] = []
    config.addinivalue_line("markers", "opendial: a voice-agent call")


def pytest_sessionfinish(session: pytest.Session) -> None:
    out = _option(session.config, "report")
    outcomes = session.config.stash[_OUTCOMES]
    if out and outcomes:
        write_report(outcomes, Path(out))


def pytest_collect_file(parent: pytest.Collector, file_path: Path) -> pytest.Collector | None:
    if any(file_path.match(p) for p in PATTERNS):
        return ScenarioFile.from_parent(parent, path=file_path)
    return None


class CallFailed(Exception):
    def __init__(self, outcome: Outcome):
        self.outcome = outcome


def _run(config: pytest.Config, scenario: Scenario) -> Outcome:
    outcome = asyncio.run(run_scenario(scenario, _config(config)))
    config.stash[_OUTCOMES].append(outcome)
    return outcome


class ScenarioFile(pytest.File):
    def collect(self) -> Iterator[pytest.Item]:
        scenario = load_scenario(self.path)
        item = ScenarioItem.from_parent(self, name=scenario.name, scenario=scenario)
        item.add_marker("opendial")
        for tag in scenario.tags:
            item.add_marker(tag)
        yield item


class ScenarioItem(pytest.Item):
    def __init__(self, *, scenario: Scenario, **kwargs: Any):
        super().__init__(**kwargs)
        self.scenario = scenario

    def runtest(self) -> None:
        outcome = _run(self.config, self.scenario)
        if not outcome.passed:
            raise CallFailed(outcome)

    def repr_failure(self, excinfo: pytest.ExceptionInfo[BaseException], style: Any = None) -> str:
        if not isinstance(excinfo.value, CallFailed):
            return str(super().repr_failure(excinfo))
        o = excinfo.value.outcome
        lines = [f"{r.status:<7} {r.criterion}: {r.detail}" for r in o.results]
        lines += ["", f"ended by {o.record.ended_by}", o.record.transcript()]
        return "\n".join(lines)

    def reportinfo(self) -> tuple[Path, int, str]:
        return self.path, 0, f"scenario: {self.name}"


@pytest.fixture
def opendial_call(request: pytest.FixtureRequest) -> Callable[..., Outcome]:
    """Run a scenario (object or YAML path) and return its Outcome. Does not assert."""

    def call(scenario: Scenario | str | Path) -> Outcome:
        if not isinstance(scenario, Scenario):
            scenario = load_scenario(scenario)
        return _run(request.config, scenario)

    return call
