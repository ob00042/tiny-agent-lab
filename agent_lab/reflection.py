from dataclasses import dataclass
from agent_lab.analysis import count_tool_failures
from agent_lab.trajectory import Trajectory
from agent_lab.memory import Memory


@dataclass
class Reflection:
    lesson: str


def reflect(
    trajectory: Trajectory,
    memory: Memory,
) -> Reflection | None:
    failures = count_tool_failures(
        trajectory
    )

    if failures >= 2:
        reflection = Reflection(
            lesson=(
                "Multiple tool failures occurred "
                "during this run."
            )
        )
        memory.add(kind="reflection", content=reflection.lesson)
        return reflection

    return None
