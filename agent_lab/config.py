from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    model: str = "Qwen3-0.6B-Q4_0"
    proposer_prompt_version: str = "v1"
    temperature: float = 0.0
    max_proposal_attempts: int = 3
    reflection_prompt_version: str = "v1"
