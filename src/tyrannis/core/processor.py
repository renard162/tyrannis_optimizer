import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from typing import Any, Generic, Self, TypeVar

import numpy as np

from .algorithm import AlgorithmBase, CostFunctionWrapperBase, ParticleBase
from .backend_migration import MigrationDriverBase, MigrationProcessorBase
from .results import HistoryConfig, ProcessorResult
from .signals import LocalEvent

SignalType = TypeVar("SignalType", bound=LocalEvent)


class IntSequence:
    def __init__(self) -> None:
        self._value: int = 0

    def spawn(self) -> int:
        value = self._value
        self._value += 1
        return value


class ProcessorBase(ABC, Generic[SignalType]):
    """Base class for processor agent."""

    _migration_signal: SignalType
    _cost_function_wrapper: type[CostFunctionWrapperBase]
    _processors_pool: dict[str, Self]
    _migration_driver: MigrationDriverBase | None
    _migration_processor: MigrationProcessorBase | None
    _population: dict[str, float]
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

        The implementation must not initialize execution-context resources such as
        synchronization primitives or other non-serializable objects. These
        resources must be created by `initialize_execution_context` immediately
        before the processor is executed and removed by
        `finalize_execution_context` when execution is finished.

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
        Initialize resources required exclusively during processor execution.

        This method must create any execution-specific resources that are required
        for the processor to operate but must not be present while the processor is
        being serialized or replicated. This includes any non-serializable
        variables, synchronization primitives, process or thread resources, or
        other runtime-specific objects required by the processor's execution
        strategy.

        The resources created by this method must be local to the execution context
        in which the processor will run. They must not be created during
        initialization or replication of the processor, since the processor may
        subsequently be serialized and distributed to another execution context.

        Every resource created by this method must have a corresponding cleanup
        operation implemented by `finalize_execution_context`.
        """

    def finalize_execution_context(self) -> None:
        """
        Remove resources associated with the processor's execution context.

        This method must release and remove any execution-specific resources
        created by `initialize_execution_context`, including non-serializable
        variables, synchronization primitives, process or thread resources, and
        other runtime-specific objects that must not be retained after execution.

        This method must leave the processor in a state that can safely be
        serialized, deep-copied, or returned from a worker to the driver without
        carrying resources that are specific to the execution context in which
        it was run.

        Every resource initialized by `initialize_execution_context` must be
        released or replaced here before the processor leaves its execution
        context.
        """

    @abstractmethod
    def initialize_loop_context(self) -> None:
        """
        Implement here all start logic related with migration event signal set

        Set here the _migration_signal
        """
        raise NotImplementedError

    @abstractmethod
    def finalize_loop_context(self) -> None:
        """
        Implement here all stop logic related with migration event signal set.

        Unset hete the _migration_signal
        """
        raise NotImplementedError

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new_processor = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_processor

        for name, value in self.__dict__.items():
            if name in self._excluded_attributes:
                continue
            setattr(new_processor, name, deepcopy(value, memo))

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
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
    ) -> None:
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

        self._migration_processor = None
        self._identifier = "MainProcessor"
        self._population = {}

        self._seed_sequence = np.random.SeedSequence(seed)
        self._pool_count_sequence = IntSequence()
        self._result = ProcessorResult()
        self._processors_pool = {}

    def create_processors_pool(self, n_islands: int) -> None:
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

        new_processor._migration_processor = (
            self._migration_driver.create_processor_module(  # type: ignore
                identifier=processor_identifier,
            )
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
    def population(self) -> dict[str, float]:
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
                identifier=f"{self._identifier}|particle:{p_idx}",
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

    def _insert_arrival_particle(self, particle_data: dict[str, Any]) -> None:
        particle_id = particle_data.get("identifier")

        if particle_id is None:
            raise ValueError("Arrival particle data must contain an identifier.")

        if particle_id not in self._algorithm.population:
            self._algorithm.create_particle(**particle_data)

    def _departure_particle(self, particle_id: str) -> None:
        if particle_id not in self._algorithm.population:
            return

        del self._algorithm.population[particle_id]

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
        self,
        actual_iter: int,
        new_particles: Iterable[ParticleBase],
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
        self,
        actual_iter: int,
        updated_particles: Iterable[ParticleBase],
    ) -> None:
        if not self._history_config.error:
            return

        event = self._history_config.get_event("error")

        for particle in updated_particles:
            candidate_fitness_is_inf = (
                particle.candidate_fitness is not None
                and np.isinf(particle.candidate_fitness)
            )

            if not candidate_fitness_is_inf:
                continue

            particle_data = particle()
            particle_data["variables"] = particle.candidate_variables
            particle_data["fitness"] = particle.candidate_fitness

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
        if not self._history_config.best:
            return

        bests = [
            (self._algorithm.local_best, "local_best"),
            (self._algorithm.iter_best, "iter_best"),
            (self._algorithm.iter_worst, "iter_worst"),
        ]

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
        algorithm's defined iteration lifecycle.
        """


def evaluate_particle(
    particle_id: str,
    algorithm: AlgorithmBase,
    fitness_failure_strategy: str,
    initialize_particle: bool = False,
) -> ParticleBase:
    try:
        if initialize_particle:
            return algorithm.initialize_particle(particle_id)

        return algorithm.update_particle(particle_id)

    except Exception:
        if fitness_failure_strategy == "invalidate":
            particle = algorithm.population[particle_id]
            particle.candidate_fitness = np.inf
            return particle

        raise
