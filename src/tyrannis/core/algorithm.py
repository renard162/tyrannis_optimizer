import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from typing import Any, Generic, TypeAlias, TypeVar

import numpy as np
from numpy.random import SeedSequence

Serializable: TypeAlias = (
    str
    | int
    | float
    | np.float64
    | bool
    | list["Serializable"]
    | Mapping[str, "Serializable"]
)


class CostFunctionWrapperBase(ABC):
    """Neutral wrapper for a cost function."""

    def __init__(self, function: Callable[..., float]) -> None:
        self._function = function

    def __call__(self, *args: Any, **kwargs: Any) -> float:
        function = self._function
        return function(*args, **kwargs)


class ParticleBase(ABC):
    """
    Base class for optimization particles.

    A particle represents an isolated candidate solution and must contain all
    state and behavior that is specific to that candidate. When implementing a
    new optimization algorithm, a dedicated particle class must be created by
    inheriting from `ParticleBase`, even when no additional behavior or state is
    required.

    A particle must remain independent from all other particles and from the
    algorithm itself. Anything that concerns only the particle belongs in its
    object, while logic that requires information about other particles,
    population-wide state, or the algorithm must remain outside the particle
    and be handled by `AlgorithmBase`. Particle objects must therefore not
    maintain dependencies or references between one another.

    During execution, the particle holds its current state and may temporarily
    hold a candidate state produced by the algorithm. The algorithm is
    responsible for determining how particles interact and for deciding when
    candidate states are consolidated into the current state.
    """

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: float = np.inf,
    ) -> None:
        self._identifier = identifier
        self._variables = variables
        self._fitness = fitness

        self._new_particle = True
        self._candidate_variables = None
        self._candidate_fitness = None
        self._random_values_cache = []

        if not np.isinf(fitness):
            self._candidate_variables = self._variables
            self._candidate_fitness = self._fitness

    def __call__(self) -> dict[str, Serializable]:
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
    def fitness(self) -> float:
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
            raise RuntimeError("No candidate solution available for consolidation.")

        if self._new_particle and (not np.isinf(self._fitness)):
            self._new_particle = False
            return

        self._new_particle = False

        if consolidate_new or np.isinf(self._fitness):
            self._variables = self._candidate_variables
            self._fitness = self._candidate_fitness

        self._candidate_variables = None
        self._candidate_fitness = None


ParticleType = TypeVar("ParticleType", bound=ParticleBase)


class AlgorithmBase(ABC, Generic[ParticleType]):
    """
    Base class for optimization algorithms.

    This class defines the execution contract of an optimization algorithm and
    is responsible for all interactions between particles. Concrete algorithms
    implement the particle creation, initialization, update, and consolidation
    rules, while `AlgorithmBase` manages the population and coordinates their
    execution.

    The execution starts by creating the initial particles and then repeatedly
    follows this lifecycle:

        create_particle
        -> pre_iteration
        -> create_random_cache (new particles)
        -> initialize_particle (new particles)
        -> create_random_cache (entire population)
        -> update_particle (entire population)
        -> pos_iteration

    `pre_iteration` prepares the particle states and algorithm state for the
    iteration, including, but not limited to, creating or removing particles
    from the population.

    `create_random_cache` generates the random values required by
    `initialize_particle` and `update_particle`.

    `initialize_particle` performs the algorithm-specific initialization of
    newly created particles, establishing their initial state.

    `update_particle` computes the transition of particles from state k to
    state k+1. The resulting candidate state is not necessarily consolidated
    at this stage, since consolidation may require comparisons between
    particles or other population-level decisions.

    `pos_iteration` consolidates the new particle states and prepares the
    algorithm and its particles for the next execution cycle.
    """

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
        fitness: float = np.inf,
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
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        """
        Generate and store the random values required by the specified particles.

        ``particle_ids`` contains the identifiers of the particles that will be
        processed in the immediately following particle-processing phase. The
        concrete algorithm must generate every random value that its
        ``initialize_particle`` or ``update_particle`` implementation requires
        and store those values in each particle's ``random_cache``.

        ``initialize`` indicates whether the particles are about to be
        initialized. When ``True``, the algorithm may use an initialization-specific
        random-generation strategy. This is important because the random values
        required to establish the initial state of a particle may be generated
        differently from the random values required to update an already existing
        particle. When ``False``, the cache must contain the random values required
        by the normal particle-update phase.

        The random cache exists to separate random-number generation from particle
        processing. This is particularly important when particle processing is
        executed in parallel, because worker processes may inherit equivalent or
        otherwise correlated states of a deterministic random number generator. If
        random values were generated independently inside ``initialize_particle``
        or ``update_particle``, different workers could generate equal or
        undesirably correlated values for different particles.

        Therefore, random values used by particle processing must be generated by
        this method, before the particles are submitted to the workers. The values
        are generated using the random-number generator owned by the algorithm and
        are then transferred to the particle through its ``random_cache``.

        The amount and type of random values required are specific to each
        algorithm. Implementations must generate exactly the values required by
        the corresponding particle-processing logic and must not assume that all
        algorithms require the same number or type of random values.

        The value of ``initialize`` may therefore affect both the amount and the
        type of random values generated. An algorithm may require one sampling
        strategy when establishing the initial state of a particle and another
        strategy when transitioning an existing particle from state ``k`` to
        state ``k+1``.

        For example, an Artificial Bee Colony implementation may require a random
        selection of a single variable to modify during an update, while its
        initialization may require random values for establishing all variables of
        a new particle. A Genetic Algorithm may require several random values to
        determine mutation operations during updates, while initialization may
        require a different sampling procedure for constructing the initial
        population. Particle Swarm Optimization may require multiple floating
        point values for each variable during updates, while initialization may
        require values generated according to the initial-position strategy of the
        algorithm.

        The values must be appended to the cache in the exact order in which they
        will be consumed by ``initialize_particle`` or ``update_particle``. The
        cache therefore represents an ordered sequence of typed values, not an
        unordered collection of random parameters.

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
        Fully initialize a newly created particle.

        This method is intended exclusively for particles that have been newly
        added to the population. It must perform all algorithm-specific operations
        required to initialize the particle before it participates in the current
        iteration, including the evaluation of its initial fitness when necessary.

        Newly created particles must have their fitness evaluated during this
        process so that they can participate in the current iteration without
        requiring an additional iteration solely for their first fitness
        evaluation.

        This method must not consolidate the particle. Consolidation is performed
        subsequently by ``consolidate_new_particles``.

        Any information required to consolidate the particle that is not already
        part of the particle's state must be stored in the particle before this
        method returns. ``consolidate_new_particles`` has no access to the
        algorithm instance or any other algorithm state and must therefore be able
        to complete the consolidation using only the state contained in the
        particle.

        This method does not alter the iteration semantics of the algorithm.
        Iteration 0 remains the initialization iteration of the algorithm and is
        not affected by the use of this method.

        The method may modify the specified particle and, consequently, the
        algorithm's population. Changes made to the population are preserved
        after the particle initialization process.

        No state of the algorithm other than the population may be modified by
        this method. Any changes made to other algorithm state are considered
        volatile and will be discarded after the particle initialization process.

        The returned particle represents the fully initialized, but not yet
        consolidated, state of the particle and may be used to replace its
        corresponding entry in the population.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def consolidate_new_particles(particle: ParticleType) -> ParticleType:
        """
        Consolidate a newly initialized particle.

        This method performs the complete consolidation process for a particle
        initialized by ``initialize_particle``. The method is executed once for
        each newly initialized particle, with individual particle consolidations
        performed in parallel.

        Consolidation must be performed using only the state contained in the
        particle itself. The method must not access the algorithm instance, the
        population, algorithm configuration, the fitness function, or any other
        external state. Any information required for consolidation that is not
        inherently part of the particle must have been stored in the particle
        during ``initialize_particle``.

        After this method returns, the particle must represent its fully
        consolidated state and be ready to participate in the current iteration.

        This method must not perform particle initialization or evaluate the
        fitness function. Those operations are the responsibility of
        ``initialize_particle``.
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
            key=lambda particle: particle.fitness,
        )
        iter_worst = max(
            self._population.values(),
            key=lambda particle: particle.fitness,
        )

        self._iter_best = iter_best.identifier
        self._iter_worst = iter_worst.identifier

        if self._local_best is None or (iter_best.fitness < self._local_best.fitness):
            self._local_best = deepcopy(iter_best)
