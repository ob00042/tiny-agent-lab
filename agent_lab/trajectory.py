from dataclasses import dataclass, field

from agent_lab.models import Action, Observation


@dataclass
class TrajectoryStep:
    index: int
    action: Action
    observation: Observation


@dataclass
class Trajectory:
    steps: list[TrajectoryStep] = field(default_factory=list)

    def record(
        self,
        action: Action,
        observation: Observation,
    ) -> None:
        self.steps.append(
            TrajectoryStep(
                index=len(self.steps),
                action=action,
                observation=observation,
            )
        )
