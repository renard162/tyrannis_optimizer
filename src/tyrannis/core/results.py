import json
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self


@dataclass(frozen=True, slots=True)
class HistoryConfig:
    migration: bool = False
    pre_iteration: bool = False
    new_particle: bool = False
    error: bool = False
    iteration: bool = False
    best: bool = False
    status: bool = False

    EVENT: ClassVar[dict[str, str]] = {
        "migration": "0 - migration",
        "pre_iteration": "1 - pre_iteration",
        "new_particle": "2 - new_particle",
        "error": "3 - error",
        "iteration": "4 - iteration",
        "local_best": "5 - local_best",
        "iter_best": "5 - iter_best",
        "iter_worst": "5 - iter_worst",
    }

    @classmethod
    def get_event(cls, event_type: str) -> str:
        return cls.EVENT[event_type]

    @property
    def history_enabled(self) -> bool:
        return (
            self.migration
            or self.pre_iteration
            or self.new_particle
            or self.error
            or self.iteration
            or self.best
        )

    @property
    def registered_events(self) -> list[str]:
        return list(self.EVENT.values())


@dataclass
class ProcessorResult:
    result: dict[str, Any] | None = None
    history: list[str] = field(default_factory=list)
