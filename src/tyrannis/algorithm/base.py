import json
from abc import ABC, abstractmethod
from collections.abc import Callable


class ParticleBase(ABC):
    """Base class for optimization particles."""

    def __init__(
        self,
        identifier: str,
        fitness_function: Callable[[dict[str, float]], float],
        variables: dict[str, float],
        fitness: float | None = None,
    ) -> None:
        self._identifier = identifier
        self._fitness_function = fitness_function
        self._variables = variables
        self._fitness = fitness

        self._candidate_variables = None
        self._candidate_fitness = None

    @property
    def identifier(self) -> str:
        return self._identifier

    @identifier.setter
    def identifier(self, identifier: str) -> None:
        self._identifier = identifier

    @property
    def variables(self) -> dict[str, float]:
        return self._variables

    @property
    def fitness(self) -> float | None:
        return self._fitness

    @property
    def candidate_fitness(self) -> float | None:
        return self._candidate_fitness

    def __call__(self) -> dict[str, str | dict[str, float] | float | None]:
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
        }

    def dump(self) -> str:
        return json.dumps(self())

    def update(self, variables: dict[str, float]) -> None:
        if variables is None:
            raise ValueError("Variables cannot be None.")

        self._candidate_variables = None
        self._candidate_fitness = None

        candidate_fitness = self._fitness_function(variables)

        self._candidate_variables = variables
        self._candidate_fitness = candidate_fitness

    def consolidate(self, new: bool) -> None:
        if self._candidate_variables is None:
            raise RuntimeError("No candidate solution available for consolidation.")

        if new or (self._fitness is None):
            self._variables = self._candidate_variables
            self._fitness = self._candidate_fitness

        self._candidate_variables = None
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
        self._local_best: ParticleBase | None = None
        self._local_worst: ParticleBase | None = None

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
    def local_best(self) -> ParticleBase | None:
        return self._local_best

    @local_best.setter
    def local_best(self, new_particle: ParticleBase) -> None:
        self._local_best = new_particle

    @property
    def local_worst(self) -> ParticleBase | None:
        return self._local_worst

    @local_worst.setter
    def local_worst(self, new_particle: ParticleBase) -> None:
        self._local_worst = new_particle

    @abstractmethod
    def create_particle(
        self,
        identifier: str | None,
        variables: dict[str, float] | None,
        fitness: float | None,
    ) -> None:
        """
        Create and add a particle to the population.

        When no variables are provided, the particle must be created
        according to the algorithm's initial particle generation rule.

        When particle data is provided, the arguments may be used to
        create a particle with the specified state. This allows dynamic
        particle creation during algorithm execution, such as migration
        between populations or the generation of new individuals in
        evolutionary algorithms.
        """

    @abstractmethod
    def delete_particle(self, identifier: str | None) -> None:
        """
        Delete a particle from the population.

        When an identifier is provided, the particle identified by it must
        be removed according to the algorithm's deletion rules.

        When no identifier is provided, the algorithm may select a particle
        according to its own deletion rules. This allows dynamic population
        management during algorithm execution, such as removing individuals
        in evolutionary algorithms or removing particles during migration
        between populations.
        """

    @abstractmethod
    def pre_iteration(self, actual_iter: int) -> None:
        """
        Prepare the algorithm state before updating the particles.

        The current iteration number is provided through `actual_iter`.
        Iteration 0 represents the initial population setup, in which the
        initial states of the particles and the algorithm are established.
        It does not represent the first actual optimization iteration.

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

    def update_population(self, new_population: list[ParticleBase]) -> None:
        self._population.update(
            {particle.identifier: particle for particle in new_population}
        )

    @abstractmethod
    def post_iteration(self, actual_iter: int) -> None:
        """
        Process the results of the particle iteration.

        The current iteration number is provided through `actual_iter`.
        Iteration 0 represents the initial population setup, in which the
        initial states of the particles and the algorithm are established.
        It does not represent the first actual optimization iteration.

        This method may modify the algorithm object and any objects contained
        by it, including consolidating particle states and updating
        population-level results.
        """
