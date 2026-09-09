from dataclasses import dataclass, field
from typing import Any, ClassVar


@dataclass(frozen=True, slots=True)
class HistoryConfig:
    migration: bool = False
    pre_iteration: bool = False
    new_particle: bool = False
    error: bool = False
    iteration: bool = False
    best: bool = False

    EVENT: ClassVar[dict[str, str]] = {
        "migration": "0 - migration",
        "pre_iteration": "1 - pre_iteration",
        "new_particle": "2 - new_particle",
        "error": "3 - error",
        "iteration": "4 - iteration",
        "best": "5 - best",
    }

    @classmethod
    def get_event(cls, event_type: str) -> str:
        return cls.EVENT[event_type]


@dataclass
class ProcessorResult:
    result: dict[str, Any] | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
