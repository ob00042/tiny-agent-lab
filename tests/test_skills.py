from agent_lab.skills import investigate_compound
from agent_lab.tools import CALL_COUNTS
from agent_lab.state import AgentState
from agent_lab.memory import Memory
from agent_lab.trajectory import Trajectory

CALL_COUNTS.clear()


def test_investigate_compound_retries_transient_failure():
    state = AgentState()
    memory = Memory()
    trajectory = Trajectory()

    observation = investigate_compound(
        "C",
        state,
        memory,
        trajectory,
    )
    
    assert observation.success == True
    assert state.tested_compounds["C"] == 0.63
    assert state.retry_counts["C"] == 1
    assert CALL_COUNTS["C"] == 2
    assert trajectory.steps[0].observation.success is False
    assert trajectory.steps[1].observation.success is True


def test_investigate_compound_no_retries():
    state = AgentState()
    memory = Memory()
    trajectory = Trajectory()

    observation = investigate_compound(
        "B",
        state,
        memory,
        trajectory,
    )

    assert observation.success == True
    assert state.tested_compounds["B"] == 0.81
    assert "B" not in state.retry_counts
    assert CALL_COUNTS["B"] == 1
    assert trajectory.steps[0].observation.success is True
