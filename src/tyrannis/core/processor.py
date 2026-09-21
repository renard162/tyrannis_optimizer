import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Generic, Self, TypeVar

import numpy as np

from .algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    CostFunctionWrapperBase,
    ParticleBase,
)
from .backend_migration import MigrationDriverBase, MigrationProcessorBase
from .results import HistoryConfig, ProcessorResult
from .signals import EventProtocol

SignalType = TypeVar("SignalType", bound=EventProtocol)


class IntSequence:
    def __init__(self) -> None:
        self._value: int = 0

    def spawn(self) -> int:
        value = self._value
        self._value += 1
        return value


class ProcessorBase(ABC, Generic[SignalType]):
    """
    Base class for processor agents.

    A processor configured by the optimization orchestrator acts as a template
    for the processors that execute optimization islands. ``initialize_context``
    configures this template with the algorithm and optimization-wide execution
    parameters, while ``create_processors_pool`` creates independent replicas for
    the requested islands.

    Each replica receives its own algorithm configuration and migration processor
    before execution. Runtime resources that must exist only in the execution
    context are initialized after replication and removed before the processor
    leaves that context. Resources whose lifetime is naturally limited to
    ``run`` may instead be created and released inside that method.

    The original processor retains the replicas in ``processors_pool`` and may be
    used by a backend to collect the processors returned from their execution
    environments.
    """

    _migration_signal: SignalType | None
    _cost_function_wrapper: type[CostFunctionWrapperBase]
    _processors_pool: dict[str, Self]
    _migration_driver: MigrationDriverBase | None
    _migration_processor: MigrationProcessorBase | None
    _population: dict[str, np.float64]
    _result: ProcessorResult

    _excluded_attributes: tuple[str, ...] = (
        "_migration_signal",
        "_seed_sequence",
        "_processors_pool",
        "_pool_count_sequence",
        "_migration_driver",
    )

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the processor-specific state.

        This method defines the public initialization interface of the processor
        and must initialize only variables that are specific to the processor
        implementation. Any parameter required to configure behavior that is
        unique to the processor must be received and initialized here.

        Processor-specific parameters may include, for example, arguments required
        to configure the multiprocessing execution strategy, synchronization
        behavior, or any other configuration that is pertinent exclusively to the
        processor implementation.

        The implementation must not initialize runtime resources that cannot be
        safely serialized or replicated with the processor template. Such resources
        must be created only after replication, either by
        `initialize_execution_context` when they must span the complete processor
        execution or inside `run` when their lifetime can be restricted to that
        local execution scope.

        The `_cost_function_wrapper` attribute must be initialized with the
        processor-specific cost-function wrapper implementation. The wrapper must
        preserve the public callable interface of the cost function while providing
        any behavior required by the processor to execute or serialize it correctly.

        The `_excluded_attributes` tuple may be extended by the implementation
        during initialization to include any additional instance attributes that
        cannot be serialized or deep-copied and therefore must not be propagated
        when the processor is replicated. The attributes already excluded by the
        base implementation must remain excluded.

        This method constitutes the user-facing configuration interface of the
        processor. Therefore, its arguments must represent only configuration that
        belongs exclusively to the processor and must not expose parameters that
        belong to the optimization process as a whole or to another architectural
        component.
        """

    def initialize_execution_context(self) -> None:
        """
        Initialize resources that must span the processor execution context.

        This hook is executed after the processor has been replicated and, when
        applicable, deserialized in the environment in which it will run. It must
        initialize runtime resources that need to remain available across the
        complete processor execution and that must not be propagated through
        processor replication.

        Resources whose lifetime is naturally restricted to ``run`` may be created
        and released inside ``run`` itself. Pools, context managers, or similar
        resources therefore do not need to be moved into this hook solely because
        they use processes, threads, or another execution runtime.

        Every resource created by this method that survives beyond a narrower local
        context must have a corresponding cleanup operation in
        ``finalize_execution_context``.
        """

    def finalize_execution_context(self) -> None:
        """
        Finalize resources that span the processor execution context.

        This hook must release resources initialized by
        ``initialize_execution_context`` and restore any state that must not remain
        attached to the processor after execution. Resources created and fully
        managed inside ``run`` are outside this hook's ownership and must be cleaned
        up by the context that created them.

        After this method returns, the processor must not retain live resources
        owned by its execution context that would prevent it from being safely
        returned to the backend or reused according to the concrete processor's
        lifecycle.
        """

    @abstractmethod
    def initialize_loop_context(self) -> None:
        """
        Initialize runtime state shared with migration during the iteration loop.

        The concrete processor must create the migration signal required by its
        execution model, assign it to ``_migration_signal``, and provide that signal
        to the associated migration processor through its loop-context
        initialization. Any additional migration runtime state whose lifetime is
        restricted to the processor loop must also be initialized here.

        This method is called by ``run`` before the first call to
        ``migration_control``.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize_loop_context(self) -> None:
        """
        Finalize runtime state shared with migration during the iteration loop.

        The concrete processor must finalize the migration processor's loop context
        and release or clear the migration signal and any other runtime state
        initialized by ``initialize_loop_context``.

        This method must be safe to execute as part of the cleanup path of ``run``
        so that loop-specific migration resources do not remain attached to the
        processor after execution.
        """
        raise NotImplementedError

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new_processor = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_processor

        for name, value in self.__dict__.items():
            if name in self._excluded_attributes:
                continue
            setattr(new_processor, name, deepcopy(value, memo))

        new_processor._migration_signal = None
        new_processor._seed_sequence = None
        new_processor._pool_count_sequence = None
        new_processor._processors_pool = {}
        new_processor._migration_driver = None
        new_processor._migration_processor = None

        return new_processor

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        migration_driver: MigrationDriverBase,
        history_config: HistoryConfig,
        fitness_failure_strategy: str,
        seed: int | None,
    ) -> None:
        """
        Configure the processor template for an optimization execution.

        The optimization orchestrator supplies the complete execution context to
        this method before any processor replicas are created. The configured
        processor acts as a template: ``create_processors_pool`` deep-copies it and
        configures each replica with an independent algorithm random state and its
        own migration processor.

        This method initializes only serializable or replicable execution state.
        Resources that belong to the environment in which a replica actually runs
        must be created later by ``initialize_execution_context``,
        ``initialize_loop_context``, ``run``, or another concrete runtime scope as
        appropriate.

        All arguments supplied by the optimization orchestrator are required. This
        prevents an omitted orchestration parameter from being silently replaced by
        an internal default.
        """
        self._algorithm = algorithm
        self._n_iter = n_iter
        self._n_particles = n_particles

        if fitness_failure_strategy not in ("invalidate", "raise"):
            raise ValueError(
                f"Invalid fitness failure strategy {fitness_failure_strategy!r}. "
                "The strategy must be either 'invalidate' or 'raise'."
            )

        self._fitness_failure_strategy = fitness_failure_strategy
        self._migration_driver = migration_driver
        self._history_config = history_config

        self._migration_signal = None
        self._migration_processor = None
        self._identifier = "MainProcessor"
        self._population = {}

        self._seed_sequence = np.random.SeedSequence(seed)
        self._pool_count_sequence = IntSequence()
        self._result = ProcessorResult()
        self._processors_pool = {}

    def create_processors_pool(self, n_islands: int) -> None:
        """
        Create and register one independent processor replica for each island.

        Replicas are created from the processor template configured by
        ``initialize_context``. Each replica receives a unique identifier, an
        independently configured algorithm random state, and a migration processor
        associated with its island. The template itself is not inserted into the
        pool and is not used as an island executor.
        """
        for _ in range(n_islands):
            processor = self._replicate_processor()
            self._processors_pool[processor.identifier] = processor

    def _replicate_processor(self) -> Self:
        idx = self._pool_count_sequence.spawn()
        new_processor = deepcopy(self)

        processor_identifier = f"island:{idx}"

        new_processor.set_identifier(processor_identifier)
        new_processor._algorithm.configure(
            identifier=f"{processor_identifier}|algorithm",
            cost_function_wrapper=self._cost_function_wrapper,
            seed=self._seed_sequence.spawn(1)[0],
        )

        migration_driver = self._migration_driver

        if migration_driver is None:
            raise RuntimeError("Migration driver has not been initialized.")

        new_processor._migration_processor = migration_driver.create_processor_module(
            identifier=processor_identifier
        )

        return new_processor

    def update_processors_pool(self, new_processors: Iterable[Self]) -> None:
        self._processors_pool.update(
            {processor.identifier: processor for processor in new_processors}
        )

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def processors_pool(self) -> dict[str, Self]:
        return self._processors_pool

    @property
    def population(self) -> dict[str, np.float64]:
        return self._population

    @property
    def result(self) -> ProcessorResult:
        return self._result

    def set_identifier(self, identifier: str) -> None:
        self._identifier = identifier

    def init_particles(self) -> None:
        if len(self._algorithm.population) > 0:
            return

        for p_idx in range(self._n_particles):
            self._algorithm.create_particle(
                identifier=f"{self._identifier}|particle:{p_idx}"
            )

    def start_migration(self) -> None:
        if self._migration_processor is None:
            raise RuntimeError("Migration processor has not been initialized.")

        self._migration_processor.start()

    def stop_migration(self) -> None:
        if self._migration_processor is None:
            return

        self._migration_processor.stop()

    def migration_control(self, actual_iter: int) -> None:
        if self._migration_processor is None:
            raise RuntimeError("Migration processor has not been initialized.")

        iter_best = (
            None
            if self._algorithm.iter_best is None
            else self._algorithm.iter_best.dump()
        )

        self._migration_processor.migration_control(
            actual_iter=actual_iter,
            population=self._population,
            iter_best=iter_best,
            insert_arrival_particle=self._insert_arrival_particle,
            departure_particle=self._departure_particle,
        )

        self._migration_processor.synchronization_control(
            actual_iter=actual_iter,
            insert_arrival_particle=self._insert_arrival_particle,
            departure_particle=self._departure_particle,
        )

        self._algorithm.update_n_particles()

    def _insert_arrival_particle(self, particle_data: dict[str, Any]) -> None:
        particle_id = particle_data.get("identifier")

        if particle_id is None:
            raise ValueError("Arrival particle data must contain an identifier.")

        if particle_id not in self._algorithm.population:
            self._algorithm.create_particle(**particle_data)

    def _departure_particle(self, particle_id: str) -> dict[str, Any] | None:
        particle = self._algorithm.population.get(particle_id)

        if particle is None:
            return None

        particle_data = particle()

        self._algorithm.delete_particle(identifier=particle_id)
        self._population.pop(particle_id, None)

        return particle_data

    def update_status(self) -> None:
        self._update_partial_result()
        self._update_population()

    def _update_population(self) -> None:
        self._population = dict(
            sorted(
                (
                    (particle.identifier, particle.fitness)
                    for particle in self._algorithm.population.values()
                ),
                key=lambda item: item[1],
            )
        )

    def _update_partial_result(self) -> None:
        self._result.result = (
            None if self._algorithm.local_best is None else self._algorithm.local_best()
        )

    def pre_iteration_log(self, actual_iter: int) -> None:
        if not self._history_config.pre_iteration:
            return

        event = self._history_config.get_event("pre_iteration")

        for particle in self._algorithm.population.values():
            self._result.history.append(
                json.dumps(
                    {
                        "iteration": actual_iter,
                        "event": event,
                        "origin": self._identifier,
                        "particle": particle(),
                    }
                )
            )

    def new_particle_log(
        self, actual_iter: int, new_particles: Iterable[ParticleBase]
    ) -> None:
        if not self._history_config.new_particle:
            return

        event = self._history_config.get_event("new_particle")

        for particle in new_particles:
            self._result.history.append(
                json.dumps(
                    {
                        "iteration": actual_iter,
                        "event": event,
                        "origin": self._identifier,
                        "particle": particle(),
                    }
                )
            )

    def error_log(
        self, actual_iter: int, updated_particles: Iterable[ParticleBase]
    ) -> None:
        if not self._history_config.error:
            return

        event = self._history_config.get_event("error")

        for particle in updated_particles:
            if particle.error_fitness is None:
                continue

            particle_data = particle()
            particle_data["variables"] = particle.candidate_variables
            particle_data["fitness"] = particle.error_fitness

            self._result.history.append(
                json.dumps(
                    {
                        "iteration": actual_iter,
                        "event": event,
                        "origin": self._identifier,
                        "particle": particle_data,
                    }
                )
            )

    def iteration_log(self, actual_iter: int) -> None:
        if not self._history_config.iteration:
            return

        event = self._history_config.get_event("iteration")

        for particle in self._algorithm.population.values():
            self._result.history.append(
                json.dumps(
                    {
                        "iteration": actual_iter,
                        "event": event,
                        "origin": self._identifier,
                        "particle": particle(),
                    }
                )
            )

    def best_log(self, actual_iter: int) -> None:
        if not (self._history_config.best or self._history_config.status):
            return

        bests = [(self._algorithm.local_best, "local_best")]

        if self._history_config.status:
            bests.extend(
                [
                    (self._algorithm.iter_best, "iter_best"),
                    (self._algorithm.iter_worst, "iter_worst"),
                ]
            )

        for particle, event_name in bests:
            if particle is None:
                continue

            self._result.history.append(
                json.dumps(
                    {
                        "iteration": actual_iter,
                        "event": self._history_config.get_event(event_name),
                        "origin": self._identifier,
                        "particle": particle(),
                    }
                )
            )

    @abstractmethod
    def run(self) -> None:
        """
        Execute the algorithm iterations.

        The iteration loop must run `actual_iter` from 0 through `self._n_iter`,
        inclusive. Iteration 0 represents the initial execution required to
        establish the initial states of the algorithm and its particles and
        does not represent an actual optimization iteration.

        Before processing each iteration, the processor must call
        `migration_control` to allow the migration processor to enforce its
        synchronization policy and apply pending migration operations.

        The migration processor may block the processor loop through its migration
        signal when synchronization or migration processing requires the iteration
        loop to wait.

        The algorithm iteration must then be executed according to the
        algorithm's defined iteration lifecycle. Newly created particles must be
        initialized and consolidated before the regular update phase. From
        iteration 1 onward, the processor must create the random cache for the
        current population, execute the first particle update, and incorporate the
        returned particles into the population.

        When `double_particle_check` is enabled, `inter_iteration` must always be
        executed after the first population update and before determining which
        particles participate in the second update phase. After
        `inter_iteration` returns, the processor must copy `double_check_ids` and
        use that same list for the complete second phase.

        If the copied `double_check_ids` list is not empty, the processor must
        create the second random cache only for those identifiers, execute
        `second_update_particle` only for those identifiers, and update the
        population only with the particles returned by that second phase. If the
        list is empty, the second random-cache creation, particle processing, and
        population update must all be skipped.

        This ordering is required because `inter_iteration` may determine
        `double_check_ids` from the results of the first update phase. It also
        guarantees that the random values generated for the second phase correspond
        exactly to the particles submitted for `second_update_particle`.
        """


def evaluate_particle(
    particle_id: str,
    algorithm: AlgorithmBase,
    fitness_failure_strategy: str,
    initialize_particle: bool = False,
    second_update: bool = False,
) -> ParticleBase:
    try:
        if initialize_particle:
            particle = algorithm.initialize_particle(particle_id)
        elif second_update:
            particle = algorithm.second_update_particle(particle_id)
        else:
            particle = algorithm.update_particle(particle_id)

        candidate_fitness = particle.candidate_fitness

        if (candidate_fitness is not None) and np.isnan(candidate_fitness):
            algorithm.population[particle_id].error_fitness = candidate_fitness
            raise ValueError(
                f"Cost function returned NaN for particle '{particle_id}'. "
                "NaN is an invalid cost function result."
            )

        return particle

    except Exception:
        if fitness_failure_strategy == "invalidate":
            particle = algorithm.population[particle_id]
            particle.candidate_fitness = FITNESS_UNDEFINED
            if particle.error_fitness is None:
                particle.error_fitness = FITNESS_UNDEFINED
            return particle

        raise
