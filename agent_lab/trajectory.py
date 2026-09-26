from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
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


def save_trajectory(
    trajectory: Trajectory,
    path: str | Path,
) -> None:
    with open(path, "w") as file:
        json.dump(
            asdict(trajectory),
            file,
            indent=2,
        )


def load_trajectory(
    path: str | Path,
) -> Trajectory:
    with open(path) as file:
        data = json.load(file)

    steps = []

    for item in data["steps"]:
        action = Action(
            **item["action"],
        )

        observation = Observation(
            **item["observation"],
        )

        steps.append(
            TrajectoryStep(
                index=item["index"],
                action=action,
                observation=observation,
            )
        )

    return Trajectory(
        steps=steps,
    )
