from agent_lab.models import Action, Observation
from agent_lab.trajectory import TrajectoryStep, Trajectory
from agent_lab.replay import ReplayExecutor, ReplayDivergence
import pytest


test_trajectory = Trajectory(steps=[
    TrajectoryStep(
        index=0,
        action=Action(tool="run_assay", arguments={"compound": "A"}),
        observation=Observation(success=True, result=0.42)
    )])

def test_matching_action():
    executor = ReplayExecutor(trajectory=test_trajectory)
    observation = executor.execute(Action(tool="run_assay", arguments={"compound": "A"}))
    assert observation == Observation(success=True, result=0.42)


def test_divergence():
    executor = ReplayExecutor(trajectory=test_trajectory)
    with pytest.raises(ReplayDivergence):
        executor.execute(Action(tool="run_assay", arguments={"compound": "B"}))
