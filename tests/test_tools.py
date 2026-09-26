# uv run python -m pytest

from agent_lab.models import Action
from agent_lab.tools import execute_tool


def test_run_assay_returns_score():
    action = Action(tool="run_assay", arguments={"compound": "B"})
    observation = execute_tool(action)
    assert observation.success == True
    assert observation.result == 0.81

def test_unknown_compound_returns_failed_observation():
    action = Action(tool="run_assay", arguments={"compound": "D"})
    observation = execute_tool(action)
    assert observation.success == False
    assert observation.error == "Unknown compound: D"

def test_unknown_tool_returns_failed_observation():
    action = Action(tool="unknown_tool", arguments={"compound": "A"})
    observation = execute_tool(action)
    assert observation.success == False
    assert observation.error == "Unknown tool: unknown_tool"