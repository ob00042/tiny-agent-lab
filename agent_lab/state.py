from dataclasses import dataclass, field


@dataclass
class AgentState:
    tested_compounds: dict[str, float] = field(default_factory=dict) # mutable defaults should not be shared between instances
    current_compound: str | None = None
    finished: bool = False
    selected_compound: str | None = None
    retry_counts: dict[str, int] = field(default_factory=dict)
