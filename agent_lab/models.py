from dataclasses import dataclass
from typing import Any


@dataclass
class Action:
    tool: str
    arguments: dict[str, Any]


@dataclass
class Observation:
    success: bool
    result: Any = None
    error: str | None = None
