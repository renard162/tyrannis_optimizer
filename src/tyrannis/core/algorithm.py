import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any

import numpy as np
from numpy.random import SeedSequence


class CostFunctionWrapperBase(ABC):
    """Neutral wrapper for a cost function."""

    def __init__(self, function: Callable[..., float]) -> None:
        self._function = function

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        function = self._function
        return function(*args, **kwargs)


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
        self._random_values_cache = []

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
    def candidate_variables(
        self, new_variables: dict[str, float]
    ) -> dict[str, float] | None:
        self._candidate_variables = new_variables

    @property
    def candidate_fitness(self) -> float | None:
        return self._candidate_fitness

    @candidate_fitness.setter
    def candidate_fitness(self, new_value: float | None) -> None:
        self._candidate_fitness = new_value

    @property
    def random_cache(self) -> list[Any]:
        return self._random_values_cache

    @random_cache.setter
    def random_cache(self, new_cache: list[Any]) -> None:
        self._random_values_cache = new_cache

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
        if (self._candidate_variables is None) or (self._candidate_fitness is None):
            if self._new_particle and self._fitness is not None:
                self._new_particle = False
                return

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
        raise NotImplementedError

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
        cost_function_wrapper: type[CostFunctionWrapperBase],
        seed: int | SeedSequence | None,
    ) -> None:
        self._identifier = identifier
        self._fitness_function = cost_function_wrapper(self._fitness_function)
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
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Create and add a particle to the population.

        The signature of this method must match the signature of the particle
        constructor (`__init__`) implemented by the algorithm, including all of its
        arguments and their respective types. The only required argument of this
        method must be `identifier`. All other arguments must have `None` as their
        default value.

        When an argument other than `identifier` is not provided, the method must
        determine its value according to the particle creation rules defined by the
        algorithm. The method must not evaluate the fitness function during particle
        creation, even when the fitness value is not provided.

        If `fitness` is not provided, the particle must be created with an undefined
        fitness. Its fitness will be evaluated subsequently by `initialize_particle`,
        which is responsible for initializing the fitness of newly created particles
        before they participate in the algorithm's execution.

        Having `identifier` as the only required argument is fundamental to the
        operation of Tyrannis, as particles may be created generically by the
        framework without knowledge of the algorithm-specific parameters required by
        their implementation.

        When complete particle state is provided, including algorithm-specific
        arguments, the method must use the provided values to recreate that state
        rather than generating new values for those arguments.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @abstractmethod
    def pre_iteration(self, actual_iter: int) -> None:
        """
        Prepare the algorithm state before the particle processing phases.

        The current iteration number is provided through ``actual_iter``.
        Iteration 0 represents the initial population setup, in which the
        initial states of the particles and the algorithm are established.
        It does not represent the first actual optimization iteration.

        This method is responsible for performing all algorithm-specific
        operations that must occur before particles are initialized or updated
        during the current iteration. Such operations may include modifying
        algorithm state, creating particles, removing particles, updating
        population-level parameters, or performing other preparations required
        by the algorithm.

        Particles created during this method are considered new particles and
        are initialized immediately after this method returns. Their fitness
        must not be evaluated directly by ``pre_iteration``. The processor is
        responsible for subsequently identifying new particles and invoking
        ``initialize_particle`` for them.

        Random values required by particle initialization or update must not be
        generated directly by this method solely for the purpose of making them
        available to ``initialize_particle`` or ``update_particle``. Random
        values required during particle processing are prepared separately by
        ``create_random_cache`` immediately before the corresponding
        particle-processing phase.

        This separation is important because particle processing may be
        executed in parallel. Random values must be generated in the execution
        context that owns the algorithm's random-number generator and before
        particle processing is distributed to worker processes or threads.

        The method may modify the algorithm object and any objects contained by
        it, according to the lifecycle requirements of the concrete algorithm.

        This method must not evaluate the fitness of newly created particles
        directly.
        """
        raise NotImplementedError

    @abstractmethod
    def create_random_cache(self, particle_ids: list[str]) -> None:
        """
        Generate and store the random values required by the specified particles.

        ``particle_ids`` contains the identifiers of the particles that will be
        processed in the immediately following particle-processing phase. The
        concrete algorithm must generate every random value that its
        ``initialize_particle`` or ``update_particle`` implementation requires
        and store those values in each particle's ``random_cache``.

        The random cache exists to separate random-number generation from
        particle processing. This is particularly important when particle
        processing is executed in parallel, because worker processes may inherit
        equivalent or otherwise correlated states of a deterministic random
        number generator. If random values were generated independently inside
        ``initialize_particle`` or ``update_particle``, different workers could
        generate equal or undesirably correlated values for different particles.

        Therefore, random values used by particle processing must be generated
        by this method, before the particles are submitted to the workers. The
        values are generated using the random-number generator owned by the
        algorithm and are then transferred to the particle through its
        ``random_cache``.

        The amount and type of random values required are specific to each
        algorithm. Implementations must generate exactly the values required by
        the corresponding particle-processing logic and must not assume that all
        algorithms require the same number or type of random values.

        For example, an Artificial Bee Colony implementation may require a
        random selection of a single variable to modify, while a Genetic
        Algorithm may require several random values to determine mutation
        operations. Particle Swarm Optimization may require multiple floating
        point values for each variable. Other algorithms may require integers,
        floating point values, booleans, categorical values, or other
        algorithm-specific random choices.

        The values must be appended to the cache in the exact order in which
        they will be consumed by ``initialize_particle`` or ``update_particle``.
        The cache therefore represents an ordered sequence of typed values, not
        an unordered collection of random parameters.

        Implementations must ensure that every particle receives its own
        independently generated sequence of random values. Random values must
        not be reused between particles unless such reuse is explicitly part of
        the algorithm's mathematical definition.

        ``initialize_particle`` and ``update_particle`` must consume the values
        from the cache rather than generating new random values themselves.
        """
        raise NotImplementedError

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
        raise NotImplementedError

    @abstractmethod
    def update_particle(self, identifier: str) -> ParticleBase:
        """
        Update and return the particle identified by ``identifier``.

        The update must calculate the candidate state of the specified particle
        according to the optimization algorithm and return the resulting
        particle. The returned particle may subsequently be used by the
        processor to replace the corresponding particle in the population.

        Any random values required by the update must be obtained from the
        particle's ``random_cache``. Random values must not normally be
        generated directly by this method. The cache is populated by
        ``create_random_cache`` immediately before the particle-processing
        phase, while the algorithm is still executing in the context that owns
        its random-number generator.

        The values in ``random_cache`` must be consumed in the same order in
        which they were generated. Implementations should therefore consume
        values using ``pop(0)`` rather than selecting values by position or
        generating additional random values during the update.

        Preserving this order is important for both correctness and type
        safety. Different algorithms require different quantities and types of
        random values. For example, an algorithm may require, in sequence,
        three floating point values, one integer, and one categorical variable.
        Its cache would consequently contain values equivalent to:

            [float, float, float, int, str]

        Consuming the cache from the beginning in the same order guarantees
        that each operation receives the type and value intended for it. If
        values are consumed out of order, an operation may receive a value of
        the wrong type or a value intended for a different operation, producing
        incorrect algorithm behavior or runtime errors.

        The concrete implementation is responsible for knowing exactly how
        many cached values it requires and in which order they were generated.
        It must consume the corresponding values completely during the particle
        update. It must not assume that the cache has a fixed size or fixed
        value types shared by all algorithms.

        The random cache is temporary state associated with the current
        particle-processing phase. It must not be treated as persistent
        algorithm state.

        This method may modify the specified particle and, consequently, the
        algorithm's population. Changes made to the population are preserved
        after the particle update process and become part of the algorithm's
        subsequent state.

        No state of the algorithm other than the population may be modified by
        this method. Any changes made to other algorithm state are considered
        volatile and may be discarded after the particle update process.

        The returned particle represents the updated state of the particle and
        may be used to replace its corresponding entry in the population.
        """
        raise NotImplementedError

    def get_unmodified_particle(self, identifier: str) -> ParticleBase:
        particle = self._population[identifier]
        particle.candidate_variables = particle.variables
        particle.candidate_fitness = particle.fitness
        return particle

    def update_population(self, new_population: Iterable[ParticleBase]) -> None:
        self._population.update(
            {particle.identifier: particle for particle in new_population}
        )

        for particle_id in self._population:
            self._population[particle_id].random_cache = []

    @abstractmethod
    def post_iteration(self, actual_iter: int) -> None:
        """
        Finalize the current iteration after all particle processing is complete.

        The current iteration number is provided through ``actual_iter``.
        Iteration 0 represents the initial population setup of the algorithm
        and does not represent the first actual optimization iteration.

        This method is executed after the initialization of newly created
        particles and the update of the particles participating in the current
        iteration. It is therefore the appropriate lifecycle stage for
        consolidating candidate particle states, updating population-level
        algorithm state, calculating iteration-level information, and performing
        any other operations that depend on the results of the current
        particle-processing phase.

        Newly created particles must not be initialized here. Their initial
        fitness is evaluated and consolidated by ``initialize_particle`` before
        they participate in the current iteration.

        This method may modify the algorithm object and any objects contained
        by it, including consolidating particle candidates and updating
        population-level results.

        The implementation must preserve the iteration semantics established
        by the processor. Iteration 0 remains the initialization iteration and
        must not be interpreted as the first optimization iteration merely
        because particle-processing methods may be invoked during it.
        """
        raise NotImplementedError

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
