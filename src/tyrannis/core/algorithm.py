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

    Fitness values follow minimization semantics. ``-np.inf`` is therefore a
    valid evaluated fitness representing the best possible value, while
    ``+np.inf`` represents the worst possible value and is also used by
    ``FITNESS_UNDEFINED`` as the initial value for unevaluated particles. Code
    that distinguishes evaluated state must not use ``np.isinf``, because that
    would incorrectly classify ``-np.inf`` as undefined.
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

        if fitness != FITNESS_UNDEFINED:
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
    def candidate_variables(self, new_variables: dict[str, float]) -> None:
        self._candidate_variables = new_variables

    @property
    def candidate_fitness(self) -> np.float64 | None:
        return self._candidate_fitness

    @candidate_fitness.setter
    def candidate_fitness(self, new_value: np.float64 | None) -> None:
        self._candidate_fitness = new_value

    @property
    def random_cache(self) -> dict[str, Any]:
        """Return the temporary random state prepared for this particle.

        The cache contract is ``dict[str, Any]``. Values normally contain
        materialized random values prepared before particle processing. When the
        required number of draws is dynamic or pre-materialization would be
        unnecessarily excessive, a particle-specific ``np.random.Generator`` may
        instead be stored as a named cache value by
        ``AlgorithmBase.create_random_cache``.

        A cached generator is part of the cache value space; it does not replace
        the dictionary itself. Its state must have been derived from fresh state of
        the algorithm's persistent random-number generator rather than by
        recreating a generator from the original algorithm seed for each iteration.
        """
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
        """
        Evaluate and store a candidate solution.

        This method is the exclusive entry point for cost-function evaluations
        performed by optimization algorithms. Every fitness evaluation, including
        evaluations performed during ``initialize_particle``, ``update_particle``,
        and ``second_update_particle``, must be executed through
        ``ParticleBase.update`` or the corresponding inherited implementation.

        Algorithm implementations must never invoke the fitness function directly.
        This requirement also applies when an algorithm evaluates multiple
        candidate solutions for the same particle during one iteration, performs
        greedy selection between intermediate candidates, or uses more than one
        particle-processing phase. Each candidate whose fitness is required must
        first be passed through this method, and its resulting
        ``candidate_fitness`` must be used by the algorithm.

        Centralizing evaluation in the particle keeps the evaluated fitness
        associated with the corresponding candidate state and provides a single
        evaluation contract for all optimization algorithms. The supplied
        ``variables`` are stored as ``candidate_variables`` and the result of
        ``fitness_function`` is stored as ``candidate_fitness``.

        This method does not consolidate the candidate into the particle's current
        state. Consolidation remains the responsibility of the algorithm lifecycle.
        """
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

        if self._new_particle:
            self._new_particle = False

            if self._fitness != FITNESS_UNDEFINED:
                self._candidate_variables = None
                self._candidate_fitness = None
                return

            self._variables = self._candidate_variables
            self._fitness = self._candidate_fitness
        elif consolidate_new:
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
                -> [if double_check_ids]
                    -> create_random_cache (double_check_ids)
                    -> second_update_particle (double_check_ids)
                    -> update population
        -> post_iteration

    `pre_iteration` prepares the particle states and algorithm state for the
    iteration, including, but not limited to, creating or removing particles
    from the population.

    `create_random_cache` prepares the random state required by
    `initialize_particle`, `update_particle`, and `second_update_particle`.
    Materialized random values are preferred. When a particle requires a
    dynamic number of draws, or pre-materializing all possible values would
    be unnecessarily excessive, the cache may instead contain an independent
    particle-specific ``np.random.Generator`` as one of its named values.

    `initialize_particle` performs the algorithm-specific initialization of
    newly created particles, establishing their initial state.

    `update_particle` computes the transition of particles from state k to
    state k+1. The resulting candidate state is not necessarily consolidated
    at this stage, since consolidation may require comparisons between
    particles or other population-level decisions.

    When `double_particle_check` is enabled, `inter_iteration` updates the
    algorithm state between the two particle-processing phases and defines
    `double_check_ids`, which identifies the particles that require the second
    update. Only those particles are passed to `create_random_cache` and
    `second_update_particle`. If `double_check_ids` is empty, the second
    particle-processing phase is skipped.

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
        self._double_check_ids: list[str] = []
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
        """Return whether the algorithm uses a second particle-processing phase."""
        return self._double_particle_check

    @property
    def double_check_ids(self) -> list[str]:
        """Return the particle identifiers selected for the second update phase."""
        return self._double_check_ids

    @double_check_ids.setter
    def double_check_ids(self, particle_ids: list[str]) -> None:
        self._double_check_ids = particle_ids

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

        The parameters required to represent the complete particle state must
        correspond to the parameters of the particle constructor (`__init__`)
        implemented by the algorithm, using the same names and corresponding
        types. Unlike the particle constructor, the only required argument of this
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

        If the algorithm requires a second cost-function check for one or more
        particles during an optimization iteration,
        ``self._double_particle_check`` must be set to ``True`` by this method
        when ``actual_iter == 0``. This enables the intermediate algorithm-state
        update and the optional second particle-update phase for subsequent
        optimization iterations. The particles that actually participate in
        that second phase are defined by ``double_check_ids`` after
        ``inter_iteration``.

        Particles created during this method are considered new particles and
        are initialized immediately after this method returns. Their fitness
        must not be evaluated directly by ``pre_iteration``. The processor is
        responsible for subsequently identifying new particles and invoking
        ``initialize_particle`` for them.

        Random state required for particle initialization or update must not be
        prepared directly by this method solely for the purpose of making it
        available to ``initialize_particle``, ``update_particle``, or
        ``second_update_particle``. Random state required during particle
        processing is prepared separately by ``create_random_cache`` immediately
        before the corresponding particle-processing phase.

        This separation is important because particle processing may be
        executed in parallel. Materialized random values, or independent
        particle-specific random-number generators when their use is justified,
        must be prepared in the execution context that owns the algorithm's
        persistent random-number generator and before particle processing is
        distributed to worker processes or threads.

        The method may modify the algorithm object and any objects contained by
        it, according to the lifecycle requirements of the concrete algorithm.

        This method must not evaluate the fitness of newly created particles
        directly.
        """
        raise NotImplementedError

    @abstractmethod
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        """
        Prepare and store the random state required by the specified particles.

        ``particle_ids`` contains the identifiers of the particles that will be
        processed in the immediately following particle-processing phase. The
        concrete algorithm must prepare every source of randomness required by
        its ``initialize_particle``, ``update_particle``, or
        ``second_update_particle`` implementation and store it as named entries
        in each particle's ``random_cache``. The cache contract remains
        ``dict[str, Any]``.

        Random values should normally be materialized in advance and stored
        directly in the cache. This is the preferred strategy whenever the
        required number of draws is known and reasonably sized, because it makes
        the randomness consumed by particle processing explicit and independent
        from the worker execution context.

        When the number of draws required by a particle is only known while that
        particle is being processed, or when pre-materializing all possible draws
        would be unnecessarily excessive, the cache may instead contain an
        independent particle-specific ``np.random.Generator`` as one of its named
        values. For example, a cache may contain
        ``{"random_generator": generator}``, and the consuming method may retrieve
        that entry and draw the dynamically required values from it. The generator
        is therefore a value stored inside ``random_cache``; it does not replace
        the cache or alter its dictionary contract.

        A cached generator is an exceptional strategy. Materialized values remain
        preferred, and a generator should be used only when dynamic draw counts or
        excessive pre-materialization make it strictly advantageous. It should not
        be used merely to replace ordinary materialized values with one generator
        per variable or operation.

        Cached generators must preserve the reproducibility and progression of the
        algorithm's random stream. The algorithm seed initializes ``self._rng``
        only once, during algorithm configuration. A cached generator created in a
        later call to ``create_random_cache`` must derive its seed or entropy from
        fresh state produced by that persistent, advancing ``self._rng``. The
        original algorithm seed must not be reused to construct a new
        ``SeedSequence`` or ``Generator`` for each particle, phase, or iteration,
        because doing so would restart the same random stream and could make
        particle updates deterministically repeat across iterations.

        For example, an implementation may draw fresh integer entropy from
        ``self._rng`` and use that entropy to initialize an independent
        ``np.random.SeedSequence`` and then an ``np.random.Generator``. The exact
        derivation strategy is algorithm-specific, but it must consume fresh state
        from ``self._rng`` each time a new cached generator is required. The
        persistent ``self._rng`` itself must not be placed in the particle cache.

        Each cached generator must be independent and exclusive to the
        corresponding particle and particle-processing phase. Generator objects or
        their initial states must not be reused across particles or phases unless
        such reuse is explicitly part of the algorithm's mathematical definition.
        A subsequent execution started with the same algorithm seed should recreate
        the same complete sequence of derived generators, while successive
        iterations within that execution must continue advancing through distinct
        random state.

        ``initialize`` indicates whether the particles are about to be
        initialized. When ``True``, the algorithm may use an
        initialization-specific random-generation strategy. When ``False``, it
        must prepare the random state required by the immediately following
        particle-update phase.

        If the algorithm uses a second cost-function check during the same
        optimization iteration, ``create_random_cache`` is called separately
        before ``update_particle`` and, when required, before
        ``second_update_particle``. Each call with ``initialize=False`` must
        therefore prepare only the random state required by the immediately
        following update phase. The second call occurs after ``inter_iteration``
        and receives only the identifiers in ``double_check_ids``, allowing its
        random-generation strategy to depend on algorithm state established
        during the intermediate step. If ``double_check_ids`` is empty, the
        second cache creation is skipped.

        The random cache separates ownership of random-number generation from
        particle processing. This is particularly important when particle
        processing is executed in parallel, because worker processes may inherit
        equivalent or otherwise correlated states of a deterministic random-number
        generator. Randomness used during particle processing must therefore
        originate from state prepared by this method before the particles are
        submitted to workers.

        Entries in ``random_cache`` are identified by keys defined by the concrete
        algorithm. Each key must uniquely identify the corresponding random value,
        random object, or particle-processing operation. Algorithms may define
        their own key convention, but keys must be sufficiently specific to allow
        the consuming method to retrieve the intended value without relying on
        insertion order.

        The cache is intentionally represented as a flat dictionary rather than
        as a nested structure. This keeps access simple while allowing different
        algorithms to define keys appropriate to their own randomness requirements.
        Values may therefore be scalars, arrays, selections, or, when justified by
        the dynamic-randomness case described above, a particle-specific
        ``np.random.Generator``.

        The amount and type of random state required are specific to each
        algorithm. Implementations must prepare the values or generator state
        required by the corresponding particle-processing logic and must not
        assume that all algorithms require the same number or type of cache
        entries. For algorithms that use two particle-processing phases, the
        caches for the first and second update phases may therefore contain
        different entries according to the requirements of each phase. The
        second-phase cache is created only for the particles identified by
        ``double_check_ids``.

        The value of ``initialize`` may therefore affect both the amount and the
        type of random state prepared. An algorithm may require one sampling
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
        algorithm. An algorithm whose number of random draws depends dynamically
        on decisions made inside a particle update may instead store an independent
        generator under a key such as ``"random_generator"``.

        Implementations must ensure that every particle receives independent random
        state. Materialized random values must not be reused between particles, and
        cached generators must not share mutable generator state, unless such reuse
        is explicitly part of the algorithm's mathematical definition.

        ``initialize_particle``, ``update_particle``, and
        ``second_update_particle`` must obtain all randomness from the cache. They
        must not access the algorithm's ``self._rng`` or another untracked random
        source while particle processing is underway. When the cache contains a
        generator, the consuming method may draw the required random values from
        that cached generator. Cache entries must be accessed by key and must not
        be removed from the dictionary with ``pop`` or equivalent destructive
        operations. Drawing from a cached generator may advance that generator's
        internal state, but the generator object itself remains stored in the
        cache. The cache is cleared by the framework when the processed population
        is updated.
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

        Newly created particles whose fitness is `FITNESS_UNDEFINED` must have
        their fitness evaluated during this process so that they can participate
        in the current iteration without requiring an additional iteration solely
        for their first fitness evaluation. Particles recreated with an already
        evaluated fitness must preserve the provided fitness and do not require
        re-evaluation solely because they are new to the population.

        Any randomness required during initialization must be obtained from the
        particle's ``random_cache``. Materialized values are preferred. If
        ``create_random_cache`` has instead stored an independent
        ``np.random.Generator`` because the required number of draws is dynamic
        or pre-materialization would be excessive, this method may retrieve that
        generator by key and draw from it locally. It must not access
        ``self._rng`` directly, recreate a generator from the original algorithm
        seed, or remove cache entries with ``pop`` or equivalent destructive
        operations.

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

        Any randomness required by the update must be obtained from the
        particle's ``random_cache``. Materialized random values are preferred and
        should be used whenever the required draws are known and reasonably sized.
        The cache is populated by ``create_random_cache`` immediately before the
        particle-processing phase, while the algorithm is still executing in the
        context that owns its persistent random-number generator.

        Random state in ``random_cache`` is identified by keys defined by the
        concrete algorithm. The update implementation must retrieve each entry
        using its corresponding key without removing it from the cache and must
        not rely on dictionary insertion order.

        If a cache entry contains an independent ``np.random.Generator`` because
        the required number of draws is dynamic or pre-materialization would be
        excessive, this method may retrieve that generator by key and draw from it
        locally. Such a generator must already have been derived by
        ``create_random_cache`` from fresh state of the persistent ``self._rng``.
        This method must not access ``self._rng`` directly, recreate a generator
        from the original algorithm seed, or use another untracked random source.

        Key-based access is important because different algorithms require
        different quantities and types of random state. A cache may therefore
        contain entries for different operations, variables, or stages of the
        update. For example, an algorithm may use keys such as
        ``"x1-selection"`` and ``"x1-mutation"`` to distinguish two materialized
        random values associated with the same variable, or a key such as
        ``"random_generator"`` for a particle-specific generator.

        The concrete implementation is responsible for knowing exactly which
        cached entries it requires and which keys identify them. It must not
        assume that the cache has a fixed size or fixed value types shared by all
        algorithms. Cache entries must not be removed from the dictionary with
        ``pop`` or equivalent destructive operations. Drawing from a cached
        generator may advance its internal state without removing the generator
        entry from the cache.

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
        method is used only by algorithms that require a second cost-function
        check for one or more particles during the same optimization iteration
        and have enabled ``double_particle_check``.

        This method is executed after the results of ``update_particle`` have
        been incorporated into the population and before the optional second
        particle-processing phase. It is responsible for modifying the algorithm
        state according to the results of the first particle update when that
        intermediate state is required by the second update.

        Before this method returns, ``double_check_ids`` must contain the
        identifiers of the particles in the current population that require
        ``second_update_particle`` during this iteration. The processor reads
        this list only after ``inter_iteration`` completes. If the list is empty,
        the second random-cache creation, particle update, and population update
        are skipped.

        Such operations may include consolidating intermediate particle states,
        updating population-level parameters, calculating selection information,
        or performing other preparations required by the algorithm before the
        second particle-processing phase.

        Random state required for particle processing must not be prepared
        directly by this method solely for the purpose of making it available
        to ``second_update_particle``. When ``double_check_ids`` is not empty,
        materialized values or independent particle-specific generator state
        are prepared separately by ``create_random_cache`` immediately after
        this method returns and only for those identifiers.

        The method may modify the algorithm state and the state of existing
        particles according to the lifecycle requirements of the concrete
        algorithm. It must not create or remove particles from the population.

        Algorithms that do not require a second particle-processing phase may
        leave this method unchanged.
        """
        return

    def second_update_particle(self, identifier: str) -> ParticleType:
        """
        Perform and return the second update of the identified particle.

        This method is used only by algorithms that require a second
        cost-function check for selected particles during the same optimization
        iteration and have enabled ``double_particle_check``. It is executed only
        for identifiers present in ``double_check_ids`` after
        ``inter_iteration`` has modified the intermediate algorithm state. If
        ``double_check_ids`` is empty, this method is not called during that
        iteration.

        The second update must calculate the candidate state of the specified
        particle according to the optimization algorithm and return the resulting
        particle. The returned particle may subsequently be used by the processor
        to replace the corresponding particle in the population.

        Any randomness required by the update must be obtained from the
        particle's ``random_cache``. Materialized random values are preferred and
        should be used whenever the required draws are known and reasonably sized.
        The cache is populated by ``create_random_cache`` immediately before the
        particle-processing phase, while the algorithm is still executing in the
        context that owns its persistent random-number generator.

        Random state in ``random_cache`` is identified by keys defined by the
        concrete algorithm. The update implementation must retrieve each entry
        using its corresponding key without removing it from the cache and must
        not rely on dictionary insertion order.

        If a cache entry contains an independent ``np.random.Generator`` because
        the required number of draws is dynamic or pre-materialization would be
        excessive, this method may retrieve that generator by key and draw from it
        locally. Such a generator must already have been derived by
        ``create_random_cache`` from fresh state of the persistent ``self._rng``.
        This method must not access ``self._rng`` directly, recreate a generator
        from the original algorithm seed, or use another untracked random source.

        The concrete implementation is responsible for knowing exactly which
        cached entries it requires and which keys identify them. It must not assume
        that the cache has a fixed size or fixed value types shared by all
        algorithms. Cache entries must not be removed from the dictionary with
        ``pop`` or equivalent destructive operations. Drawing from a cached
        generator may advance its internal state without removing the generator
        entry from the cache.

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

        Algorithms that do not require a second particle-processing phase may
        use the default implementation, which returns the particle unchanged.
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
        fitness is evaluated by ``initialize_particle`` when necessary and they
        are subsequently consolidated by ``consolidate_new_particles`` before
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
