from agent_lab.models import Action, Observation
from agent_lab.trajectory import Trajectory


class ReplayDivergence(Exception):
    pass


class ReplayExecutor:
    def __init__(
        self,
        trajectory: Trajectory,
    ) -> None:
        self.trajectory = trajectory
        self.position = 0

    def execute(
        self,
        action: Action,
    ) -> Observation:
        if self.position >= len(self.trajectory.steps):
            raise ReplayDivergence(
                "Replay contains no more recorded steps"
            )

        expected_step = self.trajectory.steps[self.position]

        if action != expected_step.action:
            raise ReplayDivergence(
                f"Expected {expected_step.action}, "
                f"got {action}"
            )

        self.position += 1
        return expected_step.observation
