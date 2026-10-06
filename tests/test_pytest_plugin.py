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

GOOD = """
goal: Move my delivery to Saturday
tags: [smoke]
script: ["Saturday please", "Thanks bye"]
success: [{type: tool_called, name: reschedule}]
"""

BAD = """
goal: Cancel my order
script: ["Cancel it please"]
success: [{type: tool_called, name: cancel}]
"""


def setup(pytester):
    pytester.makepyfile(my_agent=AGENT)
    pytester.makefile(".scenario.yaml", good=GOOD, bad=BAD)
    pytester.makeini("[pytest]\nopendial_target = my_agent:agent\n")
    pytester.syspathinsert()


def test_scenarios_are_collected_and_run(pytester):
    setup(pytester)
    result = pytester.runpytest("--opendial-report", "report")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*failed  tool_called: cancel matched 0 time(s)*", "*CALLER: *"])
    assert (pytester.path / "report" / "junit.xml").exists()


def test_tags_become_markers(pytester):
    setup(pytester)
    pytester.runpytest("-m", "smoke").assert_outcomes(passed=1)


def test_opendial_call_fixture(pytester):
    setup(pytester)
    pytester.makepyfile(
        test_python="""
from opendial.scenario import Scenario

def test_call(opendial_call):
    outcome = opendial_call(Scenario(name="py", goal="g", script=["Saturday"]))
    assert outcome.record.tool_calls[0].name == "reschedule"
"""
    )
    pytester.runpytest("test_python.py").assert_outcomes(passed=1)
