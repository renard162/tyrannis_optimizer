import json
from abc import ABC, abstractmethod
from collections.abc import Callable


class ParticleBase(ABC):
    """Base class for optimization particles."""

    def __init__(
        self,
        identifier: str,
        fitness_function: Callable[[dict[str, float]], float],
        arguments: dict[str, float],
        fitness: float | None = None,
    ) -> None:
        self._identifier = identifier
        self._fitness_function = fitness_function
        self._arguments = arguments
        self._fitness = fitness

        self._candidate_arguments = None
        self._candidate_fitness = None

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str) -> None:
        self._identifier = identifier

    @property
    def arguments(self) -> dict[str, float]:
        return self._arguments

    @property
    def fitness(self) -> float | None:
        return self._fitness

    @property
    def candidate_fitness(self) -> float | None:
        return self._candidate_fitness

    def __call__(self) -> dict[str, str | dict[str, float] | float | None]:
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

    def update(self, arguments: dict[str, float]) -> None:
        if arguments is None:
            raise ValueError("Arguments cannot be None.")

        self._candidate_arguments = None
        self._candidate_fitness = None

        candidate_fitness = self._fitness_function(arguments)

        self._candidate_arguments = arguments
        self._candidate_fitness = candidate_fitness

    def consolidate(self, new: bool) -> None:
        if self._candidate_arguments is None:
            raise RuntimeError("No candidate solution available for consolidation.")

        if new or (self._fitness is None):
            self._arguments = self._candidate_arguments
            self._fitness = self._candidate_fitness

        self._candidate_arguments = None
        self._candidate_fitness = None


class AlgorithmBase(ABC):
    """Base class for optimization algorithm."""

    def __init__(
        self,
        identifier: str,
        fitness_function: Callable[[dict[str, float]], float],
        boundaries: dict[str, tuple[float, float]],
    ) -> None:
        self._identifier = identifier
        self._fitness_function = fitness_function
        self._boundaries = boundaries
        self._population: dict[str, ParticleBase] = {}
        self._lbest: ParticleBase | None = None
        self._lworst: ParticleBase | None = None

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str) -> None:
        self._identifier = identifier

    @property
    def population(self) -> dict[str, ParticleBase]:
        return self._population

    @property
    def lbest(self) -> ParticleBase | None:
        return self._lbest

    @property
    def lworst(self) -> ParticleBase | None:
        return self._lworst

    @abstractmethod
    def create_particle(
        self,
        identifier: str | None,
        arguments: dict[str, float] | None,
        fitness: float | None,
    ) -> None:
        """
        Create and add a particle to the population.

        When no arguments are provided, the particle must be created
        according to the algorithm's initial particle generation rule.

        When particle data is provided, the arguments may be used to
        create a particle with the specified state. This allows dynamic
        particle creation during algorithm execution, such as migration
        between populations or the generation of new individuals in
        evolutionary algorithms.
        """

    @abstractmethod
    def pre_iteration(self) -> None:
        """
        Prepare the algorithm state before updating the particles.

        This method may modify the algorithm object and any objects contained
        by it.
        """

    @abstractmethod
    def update_particle(self, identifier: str) -> ParticleBase:
        """
        Return the updated particle identified by `identifier`.

        This method must not modify the algorithm object or any shared state.
        The returned particle represents the candidate state resulting
        from the particle update.
        """

    def update_population(self, population: list[ParticleBase]) -> None:
        self._population = {particle.identifier: particle for particle in population}

    @abstractmethod
    def post_iteration(self) -> None:
        """
        Process the results of the particle iteration.

        This method may modify the algorithm object and any objects contained
        by it, including consolidating particle states and updating
        population-level results.
        """
