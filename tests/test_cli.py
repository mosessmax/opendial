import json

import pytest

from opendial.cli import main

AGENT = """
from opendial.transports import ToolCall
from opendial.transports.loopback import Reply


async def agent(text):
    if not text:
        return Reply("Hello, how can I help?")
    if "saturday" in text.lower():
        return Reply("Moved to Saturday.", tools=[ToolCall("reschedule", {"day": "sat"})])
    return Reply("Goodbye.", hangup=True)
"""

SCENARIO = """
goal: Move my delivery to Saturday
tags: [{tag}]
script: {script}
success:
  - {{type: tool_called, name: reschedule}}
  - {{type: no_agent_hangup}}
"""


@pytest.fixture
def project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    (tmp_path / "my_agent.py").write_text(AGENT)
    scenarios = tmp_path / "scenarios"
    scenarios.mkdir()
    good = '["Saturday please", "Thanks bye"]'
    bad = '["Saturday please", "One more thing", "My address changed"]'  # agent hangs up early
    (scenarios / "good.scenario.yaml").write_text(SCENARIO.format(tag="smoke", script=good))
    (scenarios / "bad.scenario.yaml").write_text(SCENARIO.format(tag="slow", script=bad))
    return tmp_path


def test_run_reports_and_exit_code(project, capsys):
    code = main(["run", "scenarios", "--target", "my_agent:agent", "--out", "out"])
    out = capsys.readouterr().out
    assert code == 1
    assert "PASS  good" in out and "FAIL  bad" in out
    assert (project / "out" / "index.html").exists()
    assert (project / "out" / "junit.xml").exists()


def test_run_filters_by_tag(project):
    assert main(["run", "scenarios", "--target", "my_agent:agent", "--tag", "smoke"]) == 0


def test_validate(project, capsys):
    (project / "scenarios" / "broken.scenario.yaml").write_text("goal: x\npersona: {accent: nope}")
    assert main(["validate", "scenarios"]) == 1
    assert "error" in capsys.readouterr().out


def test_packs_and_schema(capsys):
    assert main(["packs"]) == 0
    assert "ng-pidgin" in capsys.readouterr().out
    assert main(["schema"]) == 0
    assert json.loads(capsys.readouterr().out)["title"] == "Scenario"
