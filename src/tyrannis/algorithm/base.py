import json
from abc import ABC
from collections.abc import Callable


class ParticleBase(ABC):
    """Base class for optimization particles."""

    def __init__(
        self,
        identifier: str,
        fitness_function: Callable[[dict], float],
        arguments: dict,
        fitness: float | None = None,
    ) -> None:
        self._identifier = identifier
        self._arguments = arguments
        self._fitness_function = fitness_function

        self._candidate_arguments = None
        self._candidate_fitness = None

        if fitness is None:
            self._fitness = fitness_function(arguments)
        else:
            self._fitness = fitness

    @property
    def arguments(self) -> dict:
        return self._arguments

    @property
    def fitness(self) -> float | None:
        return self._fitness

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str) -> None:
        self._identifier = identifier

    def __call__(self) -> dict:
        return {
            "identifier": self._identifier,
            "arguments": self._arguments,
            "fitness": self._fitness,
        }

    def dump(self) -> str:
        return json.dumps(
            {
                "arguments": self._arguments,
                "fitness": self._fitness,
            }
        )

    def update(self, arguments: dict) -> float:
        if arguments is None:
            raise ValueError("Arguments cannot be None.")

        self._candidate_arguments = None
        self._candidate_fitness = None

        candidate_fitness = self._fitness_function(arguments)

        self._candidate_arguments = arguments
        self._candidate_fitness = candidate_fitness

        return self._candidate_fitness

    def consolidate(self) -> None:
        if self._candidate_arguments is None:
            raise RuntimeError("No candidate solution available for consolidation.")

        self._arguments = self._candidate_arguments
        self._fitness = self._candidate_fitness

        self._candidate_arguments = None
        self._candidate_fitness = None
