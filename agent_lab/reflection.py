from dataclasses import dataclass
from agent_lab.analysis import count_tool_failures
from agent_lab.trajectory import Trajectory
from agent_lab.memory import Memory
from agent_lab.llm import LocalLLMClient


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


REFLECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "failure_type": {
            "type": "string",
        },
        "lesson": {
            "type": "string",
        },
    },
    "required": [
        "failure_type",
        "lesson",
    ],
    "additionalProperties": False,
}


REFLECTION_SYSTEM_PROMPTS = {
    "v1": """Analyze only the recorded evidence.

Do not invent causes that are not present in the trajectory.

Separate operational tool failures from scientific negative results."""
}


class LLMReflector:
    def __init__(self, 
                client: LocalLLMClient, 
                prompt_version: str = "v1") -> None:
        self.client = client
        self.prompt_version = prompt_version

    @property
    def system_prompt(self):
        return REFLECTION_SYSTEM_PROMPTS[self.prompt_version]


    def _build_reflection_prompt(
        self,
        trajectory: Trajectory,
    ) -> str:
        lines = []

        for step in trajectory.steps:
            lines.append(
                f"""
                Step {step.index}
                Tool: {step.action.tool}
                Arguments: {step.action.arguments}
                Success: {step.observation.success}
                Result: {step.observation.result}
                Error: {step.observation.error}
                """.strip()
            )

        trajectory_text = "\n\n".join(lines)

        return f"""
            Analyze the following agent trajectory.

            Trajectory:
            {trajectory_text}

            Identify the main failure type and one concise lesson that is directly supported by the recorded evidence.
            """
    

    def reflect(self, trajectory: Trajectory, memory: Memory) -> Reflection | None:
        
        response = self.client.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": self.system_prompt,
                },
                {
                    "role": "user",
                    "content": self._build_reflection_prompt(trajectory=trajectory),
                },
            ],
            schema=REFLECTION_SCHEMA,
        )

        reflection = Reflection(lesson=response.lesson)
        memory.add(kind="reflection", content=reflection.lesson)
        
        return reflection
