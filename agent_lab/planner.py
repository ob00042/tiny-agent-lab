from dataclasses import dataclass
from agent_lab.models import Observation


@dataclass
class Plan:
    steps: list[str]


class Planner:
    def create_plan(self) -> Plan:
        return Plan(
            steps=[
                "inspect_previous_results",
                "test_candidates",
                "select_best_candidate",
            ]
        )

def should_replan(
    observation: Observation,
) -> bool:
    return not observation.success
