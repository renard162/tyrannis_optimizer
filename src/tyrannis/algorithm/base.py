import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.random import SeedSequence


class ParticleBase(ABC):
    """Base class for optimization particles."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: float | None = None,
    ) -> None:
        self._identifier = identifier
        self._variables = variables
        self._fitness = fitness

        self._new_particle = True
        self._candidate_variables = None
        self._candidate_fitness = None

    def __call__(self) -> dict[str, str | dict[str, float] | float | None]:
        """
        Return the particle state as a dictionary.

        The returned dictionary must always contain the arguments required by
        the particle's constructor. The keys must correspond to the constructor
        parameter names, and their values must represent the current state of
        the particle.

        This representation allows the particle state to be serialized,
        transferred, or used to recreate an equivalent particle instance.
        """
        return {
            "identifier": self._identifier,
            "variables": self._variables,
            "fitness": self._fitness,
        }

    def recreate(
        self,
        variables: dict[str, float],
        fitness: float | None,
    ) -> None:
        """
        Reset the particle state using the values provided to its constructor.

        The provided arguments must correspond to the particle constructor
        parameters that define its state, excluding the `identifier`. The internal
        state of the existing particle is reset using the provided values, while
        its `identifier` is preserved.

        This method allows a particle to be treated as a new particle without
        creating a new instance, reducing the complexity and overhead of replacing
        an existing particle object.
        """
        self._variables = variables
        self._fitness = fitness
        self._new_particle = True
        self._candidate_variables = None
        self._candidate_fitness = None

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def new_particle(self) -> bool:
        return self._new_particle

    @property
    def variables(self) -> dict[str, float]:
        return self._variables

    @property
    def fitness(self) -> float | None:
        return self._fitness

    @property
    def candidate_variables(self) -> dict[str, float] | None:
        return self._candidate_variables

    @candidate_variables.setter
    def candidate_variables(self) -> dict[str, float] | None:
        return self._candidate_variables

    @property
    def candidate_fitness(self) -> float | None:
        return self._candidate_fitness

    @candidate_fitness.setter
    def candidate_fitness(self, new_value: float | None) -> None:
        self._candidate_fitness = new_value

    def dump(self) -> str:
        return json.dumps(self())

    def update(
        self,
        variables: dict[str, float],
        fitness_function: Callable[[dict[str, float]], float],
    ) -> None:
        if variables is None:
            raise ValueError("Variables cannot be None.")

        self._candidate_variables = None
        self._candidate_fitness = None
        self._candidate_variables = variables
        self._candidate_fitness = fitness_function(variables)

    def consolidate(self, consolidate_new: bool) -> None:
        if self._candidate_variables is None:
            raise RuntimeError("No candidate solution available for consolidation.")

        self._new_particle = False

        if consolidate_new or (self._fitness is None):
            self._variables = self._candidate_variables
            self._fitness = self._candidate_fitness

        self._candidate_variables = None
        self._candidate_fitness = None


class AlgorithmBase(ABC):
    """Base class for optimization algorithm."""

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the parameters and execution state specific to the optimization algorithm.

        This method must define and initialize all parameters and internal variables
        required for the execution of the optimization algorithm. Each algorithm
        implementation is responsible for initializing its own algorithm-specific
        configuration parameters and execution state. Parameters and variables that
        are common to the optimization process as a whole should not be initialized
        here.

        Examples of algorithm-specific parameters include inertia and cognitive and
        social coefficients in Particle Swarm Optimization.
        """

    def initialize_context(
        self,
        fitness_function: Callable[[dict[str, float]], float],
        boundaries: dict[str, tuple[float, float]],
    ) -> None:
        self._fitness_function = fitness_function
        self._boundaries = boundaries
        self._population: dict[str, ParticleBase] = {}
        self._local_best: ParticleBase | None = None
        self._iter_best: str | None = None
        self._iter_worst: str | None = None

    def configure(
        self,
        identifier: str,
        seed: int | SeedSequence | None,
    ) -> None:
        self._identifier = identifier
        self._rng = np.random.default_rng(seed)

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def population(self) -> dict[str, ParticleBase]:
        return self._population

    @property
    def local_best(self) -> ParticleBase | None:
        return self._local_best

    @property
    def iter_best(self) -> ParticleBase | None:
        if self._iter_best is None:
            return None
        return self._population[self._iter_best]

    @property
    def iter_worst(self) -> ParticleBase | None:
        if self._iter_worst is None:
            return None
        return self.population[self._iter_worst]

    @property
    def new_particles_id(self) -> list[str]:
        return [
            particle_id
            for particle_id, particle in self._population.items()
            if particle.new_particle
        ]

    @abstractmethod
    def create_particle(
        self,
        identifier: str | None,
        variables: dict[str, float] | None,
        fitness: float | None,
    ) -> None:
        """
        Create and add a particle to the population.

        This method must only create and add the particle to the population. It
        must never evaluate the fitness function. When the particle is created
        without a fitness value, its fitness is initialized by a dedicated method
        invoked after ``pre_iteration``.

        When no variables are provided, the particle must be created according
        to the algorithm's initial particle generation rule.

        When particle data is provided, the arguments may be used to create a
        particle with the specified state. This allows dynamic particle creation
        during algorithm execution, such as migration between populations or the
        generation of new individuals in evolutionary algorithms.
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

        The current iteration number is provided through ``actual_iter``.
        Iteration 0 represents the initial population setup, in which the
        initial states of the particles and the algorithm are established.
        It does not represent the first actual optimization iteration.

        All operations that create or remove particles as part of the algorithm's
        iteration process must be performed during this method. Fitness
        initialization for newly created particles is performed immediately
        after ``pre_iteration`` by dedicated methods, allowing those particles
        to participate in the current iteration without requiring an additional
        iteration solely for their initial fitness evaluation.

        This method may modify the algorithm object and any objects contained
        by it. It must not evaluate the fitness of newly created particles
        directly.
        """

    @abstractmethod
    def initialize_particle(self, identifier: str) -> ParticleBase:
        """
        Initialize the fitness of a newly created particle.

        This method is intended exclusively for evaluating particles that are
        newly added to the population. Unlike the initial population setup,
        newly created particles must have their fitness initialized immediately
        so that they can participate in the current iteration without requiring
        an additional iteration solely for their first fitness evaluation.

        The particle's initial fitness must be consolidated with
        ``consolidate(consolidate_new=True)`` before this method returns, ensuring that the
        newly evaluated fitness becomes the particle's current fitness before it
        participates in subsequent iterations.

        This method does not alter the iteration semantics of the algorithm.
        Iteration 0 remains the initialization iteration of the algorithm and is
        not affected by the use of this method.

        The method may modify the specified particle and, consequently, the
        algorithm's population. Changes made to the population are preserved
        after the particle initialization process.

        No state of the algorithm other than the population may be modified by
        this method. Any changes made to other algorithm state are considered
        volatile and will be discarded after the particle initialization process.

        The returned particle represents the initialized state of the particle
        and may be used to replace its corresponding entry in the population.
        """

    @abstractmethod
    def update_particle(self, identifier: str) -> ParticleBase:
        """
        Update and return the particle identified by `identifier`.

        This method may modify the specified particle and, consequently, the
        algorithm's population. Changes made to the population are preserved after
        the particle update process and become part of the algorithm's subsequent
        state.

        No state of the algorithm other than the population may be modified by
        this method. Any changes made to other algorithm state are considered
        volatile and will be discarded after the particle update process.

        The returned particle represents the updated state of the particle and
        may be used to replace its corresponding entry in the population.
        """

    def get_unmodified_particle(self, identifier: str) -> ParticleBase:
        particle = self._population[identifier]
        particle.candidate_variables = particle.variables
        particle.candidate_fitness = particle.fitness
        return particle

    def update_population(self, new_population: Iterable[ParticleBase]) -> None:
        self._population.update(
            {particle.identifier: particle for particle in new_population}
        )

    @abstractmethod
    def post_iteration(self, actual_iter: int) -> None:
        """
        Process the results of the particle iteration.

        The current iteration number is provided through ``actual_iter``.
        Iteration 0 represents the initial population setup, in which the
        initial states of the particles and the algorithm are established.
        It does not represent the first actual optimization iteration.

        Particle states resulting from the current iteration may be consolidated
        by this method. However, newly created particles must not be initialized
        or consolidated here. Their fitness initialization and consolidation are
        performed automatically by ``initialize_particle`` before the particle
        participates in the iteration.

        This method may modify the algorithm object and any objects contained
        by it, including consolidating the results of particle updates and
        updating population-level results.
        """

    def update_solution_state(self) -> None:
        iter_best = min(
            self._population.values(),
            key=self.get_fitness,
        )
        iter_worst = max(
            self._population.values(),
            key=self.get_fitness,
        )

        self._iter_best = iter_best.identifier
        self._iter_worst = iter_worst.identifier

        if self._local_best is None or (
            self.get_fitness(iter_best) < self.get_fitness(self._local_best)
        ):
            self._local_best = deepcopy(iter_best)

    @staticmethod
    def get_fitness(particle: ParticleBase) -> float:
        if particle.fitness is None:
            raise RuntimeError(
                f"Particle '{particle.identifier}' does not have a fitness."
            )
        return particle.fitness
