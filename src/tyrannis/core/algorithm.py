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
    | None
    | list["Serializable"]
    | Mapping[str, "Serializable"]
)

FITNESS_UNDEFINED: np.float64 = np.float64(np.inf)


class CostFunctionWrapperBase(ABC):
    """Neutral wrapper for a cost function."""

    def __init__(self, function: Callable[..., np.float64]) -> None:
        self._function = function

    def __call__(self, *args: Any, **kwargs: Any) -> np.float64:
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
        fitness: np.float64 = FITNESS_UNDEFINED,
    ) -> None:
        self._identifier = identifier
        self._variables = variables
        self._fitness = fitness

        self._new_particle = True
        self._candidate_variables = None
        self._candidate_fitness = None
        self._random_values_cache = {}
        self._error_fitness = None

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
    def fitness(self) -> np.float64:
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
    def candidate_fitness(self) -> np.float64 | None:
        return self._candidate_fitness

    @candidate_fitness.setter
    def candidate_fitness(self, new_value: np.float64 | None) -> None:
        self._candidate_fitness = new_value

    @property
    def random_cache(self) -> dict[str, Any]:
        return self._random_values_cache

    @random_cache.setter
    def random_cache(self, new_cache: dict[str, Any]) -> None:
        self._random_values_cache = new_cache

    @property
    def error_fitness(self) -> np.float64 | None:
        return self._error_fitness

    @error_fitness.setter
    def error_fitness(self, new_value: np.float64) -> None:
        self._error_fitness = new_value

    def dump(self) -> str:
        return json.dumps(self())

    def update(
        self,
        variables: dict[str, float],
        fitness_function: Callable[[dict[str, float]], np.float64],
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

        self._error_fitness = None

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

        -> create_particle
        -> pre_iteration
        -> create_random_cache (new particles)
        -> initialize new particles
        -> consolidate new particles
        -> update population
        -> [if actual_iter > 0]
            -> create_random_cache
            -> update particles
            -> update population
            -> [if double_particle_check]
                -> inter_iteration
                -> create_random_cache
                -> second_update_particle
                -> update population
        -> post_iteration

    `pre_iteration` prepares the particle states and algorithm state for the
    iteration, including, but not limited to, creating or removing particles
    from the population.

    `create_random_cache` generates the random values required by
    `initialize_particle`, `update_particle`, and `second_update_particle`.

    `initialize_particle` performs the algorithm-specific initialization of
    newly created particles, establishing their initial state.

    `update_particle` computes the transition of particles from state k to
    state k+1. The resulting candidate state is not necessarily consolidated
    at this stage, since consolidation may require comparisons between
    particles or other population-level decisions.

    When `double_particle_check` is enabled, `inter_iteration` updates the
    algorithm state between the two particle-processing phases and
    `second_update_particle` performs the second particle update using that
    intermediate state.

    `post_iteration` consolidates the new particle states and prepares the
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
        fitness_function: Callable[[dict[str, float]], np.float64],
        boundaries: dict[str, tuple[float, float]],
        n_iter: int,
        n_particles: int,
    ) -> None:
        self._fitness_function = fitness_function
        self._boundaries = boundaries
        self._population: dict[str, ParticleType] = {}
        self._local_best: ParticleType | None = None
        self._iter_best: str | None = None
        self._iter_worst: str | None = None
        self._double_particle_check: bool = False
        self._max_iterations: int = n_iter
        # Start value of n_particles to initialize algorithm.
        # Thre real value is dinamically set after migration step
        self._n_particles: int = n_particles

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
    def population(self) -> dict[str, ParticleType]:
        return self._population

    @property
    def local_best(self) -> ParticleType | None:
        return self._local_best

    @property
    def iter_best(self) -> ParticleType | None:
        if self._iter_best is None:
            return None
        return self._population[self._iter_best]

    @property
    def iter_worst(self) -> ParticleType | None:
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

    @property
    def double_particle_check(self) -> bool:
        return self._double_particle_check

    def update_n_particles(self) -> None:
        self._n_particles = len(self._population)

    @abstractmethod
    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Create and add a particle to the population.

        The signature of this method must match the signature of the particle
        constructor (`__init__`) implemented by the algorithm, including all of its
        arguments and their respective types. The only required argument of this
        method must be `identifier`. All other arguments must be optional and use
        the appropriate undefined value for their respective type. In particular,
        fitness-related arguments must use `FITNESS_UNDEFINED` as their default
        value.

        When an argument other than `identifier` is not provided, the method must
        determine its value according to the particle creation rules defined by the
        algorithm. The method must not evaluate the fitness function during particle
        creation, even when the fitness value is not provided.

        If `fitness` is not provided, the particle must be created with
        `FITNESS_UNDEFINED`. Its fitness will be evaluated subsequently by
        `initialize_particle`, which is responsible for initializing the fitness of
        newly created particles before they participate in the algorithm's execution.

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
        management during execution, such as removing individuals
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

        If the algorithm requires two cost-function checks for each particle
        during an optimization iteration, ``self._double_particle_check`` must
        be set to ``True`` by this method when ``actual_iter == 0``. This enables
        the intermediate algorithm-state update and the second particle-update
        phase for the subsequent optimization iterations.

        Particles created during this method are considered new particles and
        are initialized immediately after this method returns. Their fitness
        must not be evaluated directly by ``pre_iteration``. The processor is
        responsible for subsequently identifying new particles and invoking
        ``initialize_particle`` for them.

        Random values required for particle initialization or update must not be
        generated directly by this method solely for the purpose of making them
        available to ``initialize_particle``, ``update_particle``, or
        ``second_update_particle``. Random values required during particle
        processing are prepared separately by
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
        ``initialize_particle``, ``update_particle``, or
        ``second_update_particle`` implementation requires and store those
        values in each particle's ``random_cache``.

        ``initialize`` indicates whether the particles are about to be
        initialized. When ``True``, the algorithm may use an initialization-specific
        random-generation strategy. When ``False``, it must generate the random
        values required by the particle-update phase.

        If the algorithm uses two cost-function checks during the same
        optimization iteration, every call with ``initialize=False`` must
        generate the complete set of random values required by both
        ``update_particle`` and ``second_update_particle``. The same cache
        generation strategy is therefore used before either update phase, even
        though each phase may use only part of the generated values.

        The random cache separates random-number generation from particle
        processing. This is particularly important when particle processing is
        executed in parallel, because worker processes may inherit equivalent or
        otherwise correlated states of a deterministic random-number generator.
        Random values used during particle processing are therefore generated by
        this method before the particles are submitted to workers.

        The generated values are stored in the particle's ``random_cache`` as
        named entries. Each entry must have a key that uniquely identifies the
        random value within the particle and the corresponding particle-processing
        operation. Algorithms may define their own key convention, but keys must
        be sufficiently specific to allow the consuming method to retrieve the
        intended value without relying on insertion order.

        The cache is intentionally represented as a flat dictionary rather than
        as a nested structure. This keeps access simple while allowing different
        algorithms to define keys appropriate to their own random-value
        requirements.

        The amount and type of random values required are specific to each
        algorithm. Implementations must generate the values required by the
        corresponding particle-processing logic and must not assume that all
        algorithms require the same number or type of random values. For
        algorithms that use two particle checks, every update cache must include
        the values required by both update phases.

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

        Implementations must ensure that every particle receives its own
        independently generated set of random values. Random values must not be
        reused between particles unless such reuse is explicitly part of the
        algorithm's mathematical definition.

        ``initialize_particle``, ``update_particle``, and
        ``second_update_particle`` must obtain the values from the cache rather
        than generating new random values themselves. Cached values must not be
        removed during particle processing. The cache is cleared by the framework
        when the processed population is updated.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_particle(self, identifier: str) -> ParticleType:
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
    def update_particle(self, identifier: str) -> ParticleType:
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

        Random values in ``random_cache`` are identified by keys defined by the
        concrete algorithm. The update implementation must retrieve each value
        using its corresponding key without removing it from the cache. It must
        not rely on dictionary insertion order or generate additional random
        values during the update.

        Key-based access is important because different algorithms require
        different quantities and types of random values. A cache may therefore
        contain entries for different operations, variables, or stages of the
        update. For example, an algorithm may use keys such as
        ``"x1-selection"`` and ``"x1-mutation"`` to distinguish two random values
        associated with the same variable.

        The concrete implementation is responsible for knowing exactly which
        cached values it requires and which keys identify them. It must not
        assume that the cache has a fixed size or fixed value types shared by all
        algorithms.

        The random cache is temporary state associated with the current
        particle-processing phase. It must not be treated as persistent
        algorithm state.

        This method may modify the specified particle and, consequently, the
        algorithm's population. Changes made to the population are preserved
        after the particle update process and become part of the algorithm's
        subsequent state.

        No state of the algorithm other than the population may be modified by
        this method. Changes made to other algorithm state are considered
        volatile and may be discarded after the particle update process.

        The returned particle represents the updated state of the particle and
        may be used to replace its corresponding entry in the population.
        """
        raise NotImplementedError

    def inter_iteration(self, actual_iter: int) -> None:
        """
        Update the algorithm state between two particle-processing phases.

        The current iteration number is provided through ``actual_iter``. This
        method is used only by algorithms that require two cost-function checks
        for each particle during the same optimization iteration and have enabled
        ``double_particle_check``.

        This method is executed after the results of ``update_particle`` have
        been incorporated into the population and before
        ``second_update_particle`` is executed. It is responsible for modifying
        the algorithm state according to the results of the first particle
        update when that intermediate state is required by the second update.

        Such operations may include consolidating intermediate particle states,
        updating population-level parameters, calculating selection information,
        or performing other preparations required by the algorithm before the
        second particle-processing phase.

        Random values required for particle processing must not be generated
        directly by this method solely for the purpose of making them available
        to ``second_update_particle``. Random values are prepared separately by
        ``create_random_cache`` immediately after this method returns.

        The method may modify the algorithm state and the state of existing
        particles according to the lifecycle requirements of the concrete
        algorithm. It must not create or remove particles from the population.

        Algorithms that do not require two particle checks may leave this method
        unchanged.
        """
        return

    def second_update_particle(self, identifier: str) -> ParticleType:
        """
        Perform and return the second update of the identified particle.

        This method is used only by algorithms that require two cost-function
        checks for each particle during the same optimization iteration and have
        enabled ``double_particle_check``. It is executed after
        ``inter_iteration`` has modified the intermediate algorithm state.

        The second update must calculate the candidate state of the specified
        particle according to the optimization algorithm and return the resulting
        particle. The returned particle may subsequently be used by the processor
        to replace the corresponding particle in the population.

        Any random values required by the update must be obtained from the
        particle's ``random_cache``. Random values must not normally be generated
        directly by this method. The cache is populated by
        ``create_random_cache`` immediately before the particle-processing phase,
        while the algorithm is still executing in the context that owns its
        random-number generator.

        Random values in ``random_cache`` are identified by keys defined by the
        concrete algorithm. The update implementation must retrieve each value
        using its corresponding key without removing it from the cache. It must
        not rely on dictionary insertion order or generate additional random
        values during the update.

        The concrete implementation is responsible for knowing exactly which
        cached values it requires and which keys identify them. It must not assume
        that the cache has a fixed size or fixed value types shared by all
        algorithms.

        The random cache is temporary state associated with the current
        particle-processing phase. It must not be treated as persistent algorithm
        state.

        This method may modify the specified particle and, consequently, the
        algorithm's population. Changes made to the population are preserved
        after the particle update process and become part of the algorithm's
        subsequent state.

        No state of the algorithm other than the population may be modified by
        this method. Changes made to other algorithm state are considered
        volatile and may be discarded after the particle update process. State
        required by the second update must therefore be prepared through
        ``inter_iteration`` before particle processing begins.

        The returned particle represents the updated state of the particle and
        may be used to replace its corresponding entry in the population.

        Algorithms that do not require two particle checks may use the default
        implementation, which returns the particle unchanged.
        """
        return self._population[identifier]

    def update_population(self, new_population: Iterable[ParticleType]) -> None:
        self._population.update(
            {particle.identifier: particle for particle in new_population}
        )

        for particle_id in self._population:
            self._population[particle_id].random_cache = {}

    @abstractmethod
    def post_iteration(self, actual_iter: int) -> None:
        """
        Finalize the current iteration after all particle processing is complete.

        The current iteration number is provided through ``actual_iter``.
        Iteration 0 represents the initial population setup of the algorithm
        and does not represent the first actual optimization iteration.

        This method is executed after the initialization of newly created
        particles and all particle update phases participating in the current
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
        by the processor. Iteration 0 remains the initialization iteration of the
        algorithm and must not be interpreted as the first optimization iteration
        merely because particle-processing methods may be invoked during it.
        """
        raise NotImplementedError

    def update_solution_state(self) -> None:
        iter_best = min(
            self._population.values(), key=lambda particle: particle.fitness
        )
        iter_worst = max(
            self._population.values(), key=lambda particle: particle.fitness
        )

        self._iter_best = iter_best.identifier
        self._iter_worst = iter_worst.identifier

        if self._local_best is None or (iter_best.fitness < self._local_best.fitness):
            self._local_best = deepcopy(iter_best)
