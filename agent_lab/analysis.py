from agent_lab.trajectory import Trajectory, TrajectoryStep

def count_tool_failures(trajectory: Trajectory) -> int:
    return sum(1 for step in trajectory.steps if not step.observation.success)

def count_tool_calls(trajectory: Trajectory, tool_name: str) -> int:
    return sum(1 for step in trajectory.steps if step.action.tool == tool_name)

def find_repeated_actions(trajectory: Trajectory) -> list[int]:
    repeated_indexes = []

    for i in range(1, len(trajectory.steps)):
        if trajectory.steps[i].action == trajectory.steps[i-1].action:
            repeated_indexes.append(i)

    return repeated_indexes

def classify_failure(step: TrajectoryStep) -> str|None:
    if step.observation.success:
        return None

    error = step.observation.error or ""

    if "timeout" in error.lower():
        return "tool_timeout"

    return "unknown_tool_failure"
