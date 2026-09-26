from agent_lab.models import Action, Observation
from agent_lab.trajectory import TrajectoryStep, Trajectory, save_trajectory, load_trajectory

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

def test_save_and_load_trajectory(tmp_path):
    trajectory_path = tmp_path / "trajectory.json"
    save_trajectory(trajectory=test_trajectory, path=trajectory_path)

    assert trajectory_path.exists()

    loaded_trajectory = load_trajectory(path=trajectory_path)

    assert loaded_trajectory == test_trajectory
