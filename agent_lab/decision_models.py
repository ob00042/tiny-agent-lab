from dataclasses import dataclass
from agent_lab.models import Action


@dataclass
class Proposal:
    action: Action
    reason: str


@dataclass
class Critique:
    accepted: bool
    reason: str
