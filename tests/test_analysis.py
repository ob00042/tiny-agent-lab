from agent_lab.analysis import count_tool_failures, count_tool_calls, find_repeated_actions, classify_failure
from agent_lab.models import Action, Observation
from agent_lab.trajectory import TrajectoryStep, Trajectory

test_trajectory = Trajectory(steps=[
    TrajectoryStep(
        index=0,
        action=Action(tool="run_assay", arguments={"compound": "B"}),
        observation=Observation(success=True, result=0.81)
    ),
    TrajectoryStep(
        index=1,
        action=Action(tool="run_assay", arguments={"compound": "B"}),
        observation=Observation(success=True, result=0.81)
    ),
    TrajectoryStep(
        index=2,
        action=Action(tool="run_assay", arguments={"compound": "A"}),
        observation=Observation(success=True, result=0.42)
    ),
    TrajectoryStep(
        index=3,
        action=Action(tool="run_assay", arguments={"compound": "C"}),
        observation=Observation(success=False, error="instrument timeout")
    )
])


def test_count_tool_failure():
    assert count_tool_failures(test_trajectory) == 1

def test_count_tool_calls():
    assert count_tool_calls(trajectory=test_trajectory, tool_name="run_assay") == 4

def test_find_repeated_actions():
    assert find_repeated_actions(test_trajectory) == [1]

def test_timeout_is_classified():
    step = TrajectoryStep(
        index=0,
        action=Action(tool="run_assay", arguments={"compound": "C"}),
        observation=Observation(success=False, error="instrument timeout")
    )

    classification = classify_failure(step)

    assert classification == "tool_timeout"
