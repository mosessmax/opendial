import pytest
from pydantic import ValidationError

from opendial.scenario import Scenario, ToolCalled, load_scenario

YAML = """
goal: Reschedule a delivery to Saturday
persona:
  name: Chidi
  accent: ng-pidgin
audio:
  line: ng-mobile-3g
  noise: {kind: market, snr_db: 8}
interruptions:
  - {turn: 1, say: "Abeg wait"}
dtmf:
  - {when: "order number", digits: "4417#"}
success:
  - {type: task_completed, judge: Delivery moved to Saturday}
  - {type: tool_called, name: reschedule, args: {day: saturday}}
  - {type: time_to_first_speech, max_ms: 1200}
"""


def test_load_yaml(tmp_path):
    path = tmp_path / "reschedule.scenario.yaml"
    path.write_text(YAML)
    s = load_scenario(path)
    assert s.name == "reschedule"
    assert s.persona.accent == "ng-pidgin"
    assert s.audio.noise.kind == "market"
    assert isinstance(s.success[1], ToolCalled)
    assert s.success[1].args == {"day": "saturday"}


def test_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        Scenario(name="x", goal="y", persona={"nmae": "typo"})


def test_rejects_unknown_criterion():
    with pytest.raises(ValidationError):
        Scenario(name="x", goal="y", success=[{"type": "vibes_ok"}])


def test_rejects_bad_dtmf():
    with pytest.raises(ValidationError):
        Scenario(name="x", goal="y", dtmf=[{"when": "pin", "digits": "12x"}])
