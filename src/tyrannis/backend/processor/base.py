import json
from abc import ABC, abstractmethod
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from time import sleep
from typing import Any, Generic, Self, TypeVar

import numpy as np

from ...algorithm.base import AlgorithmBase, CostFunctionWrapperBase, ParticleBase


class LocalEvent:
    def __init__(self) -> None:
        self._state: bool = False

    def set(self) -> None:
        self._state = True

    def clear(self) -> None:
        self._state = False

    def is_set(self) -> bool:
        return self._state


SignalType = TypeVar("SignalType", bound=LocalEvent)


class IntSequence:
    def __init__(self) -> None:
        self._value: int = 0

    def spawn(self) -> int:
        value = self._value
        self._value += 1
        return value


@dataclass
class StatusVariables:
    population: dict[str, float | None]
    actual_iter: int = -1
    n_process: int = 1
    iter_waiting: bool = False

    best_particle_data: str = ""
    best_particle_fitness: float | None = None
    worst_particle_data: str = ""
    worst_particle_fitness: float | None = None


@dataclass
class ControlVariables:
    arrival_particle_id: str | None = None
    arrival_particle_data: str | None = None

    departure_particle_id: str | None = None
    departure_particle_data: str | None = None

    iter_until: int | None = None


class ProcessorBase(ABC, Generic[SignalType]):
    """Base class for processor agent."""

    _stop_signal: SignalType
    _wait_signal: SignalType
    _cost_function_wrapper: type[CostFunctionWrapperBase]
    _processors_pool: dict[str, Self]
    _excluded_attributes: tuple[str, ...] = (
        "_stop_signal",
        "_wait_signal",
        "_seed_sequence",
        "_processors_pool",
        "_pool_count_sequence",
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

        In particular, `_stop_signal` and `_wait_signal` must be initialized here
        using the synchronization primitives appropriate for the processor's
        execution strategy. For example, a processor based on multiprocessing may
        initialize these attributes with `multiprocessing.Event` objects, which are
        not serializable and therefore must only exist inside the execution
        context of the worker.

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

        In particular, `_stop_signal` and `_wait_signal` must be cleared by
        replacing them with `LocalEvent` instances. `LocalEvent` provides the
        local, serializable representation of the signals required by the
        processor outside its execution context, while execution-specific
        implementations such as `multiprocessing.Event` must not remain attached
        to the processor.

        This method must leave the processor in a state that can safely be
        serialized, deep-copied, or returned from a worker to the driver without
        carrying resources that are specific to the execution context in which it
        was run.

        Every resource initialized by `initialize_execution_context` must be
        released or replaced here before the processor leaves its execution
        context.
        """

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new_processor = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_processor

        for name, value in self.__dict__.items():
            if name in self._excluded_attributes:
                continue
            setattr(new_processor, name, deepcopy(value, memo))

        new_processor._pool_count_sequence = None
        new_processor._seed_sequence = None
        new_processor._processors_pool = {}
        return new_processor

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
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

        self._identifier = "MainProcessor"
        self._control = ControlVariables()
        self._status = StatusVariables(population={})
        self._seed_sequence = np.random.SeedSequence(seed)
        self._pool_count_sequence = IntSequence()
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
    def local_best(self) -> str:
        return self._status.best_particle_data

    def set_identifier(self, identifier: str) -> None:
        self._identifier = identifier

    def init_particles(self) -> None:
        if len(self._algorithm.population) > 0:
            return

        for p_idx in range(self._n_particles):
            self._algorithm.create_particle(
                identifier=f"{self._identifier}|particle:{p_idx}",
            )

    def wait_sync(self, actual_iter: int) -> None:
        self._status.iter_waiting = False
        if self._control.iter_until is None:
            return

        while (not self._stop_signal.is_set()) and (
            (actual_iter >= self._control.iter_until) or self._wait_signal.is_set()
        ):
            self._status.iter_waiting = True
            sleep(0.01)

        self._status.iter_waiting = False

    def migration_control(self) -> None:
        self._insert_arrival_particle()
        self._departure_particle()

    def _insert_arrival_particle(self) -> None:
        if self._control.arrival_particle_id is None:
            return

        if self._control.arrival_particle_id not in self._algorithm.population:
            if self._control.arrival_particle_data is None:
                raise ValueError("Arrival particle data cannot be None.")
            data = json.loads(self._control.arrival_particle_data)
            self._algorithm.create_particle(**data)

    def _departure_particle(self) -> None:
        """
        Remove a departing particle from the algorithm population.

        This method will be implemented after the interaction methods
        between optimization islands have been defined.
        """
        return

    def update_iter_counter(self, actual_iter: int) -> None:
        self._status.actual_iter = actual_iter

    def update_status(self) -> None:
        self._update_population_status()
        self._update_partial_result()

    def _update_partial_result(self) -> None:
        if (self._algorithm.iter_best is None) or (self._algorithm.iter_worst is None):
            return
        self._status.best_particle_fitness = self._algorithm.iter_best.fitness
        self._status.best_particle_data = self._algorithm.iter_best.dump()
        self._status.worst_particle_fitness = self._algorithm.iter_worst.fitness
        self._status.worst_particle_data = self._algorithm.iter_worst.dump()

    def _update_population_status(self) -> None:
        self._status.population = {
            p_id: p.fitness for p_id, p in self._algorithm.population.items()
        }

    @abstractmethod
    def run(self) -> None:
        """
        Execute the algorithm iterations.

        The iteration loop must run `actual_iter` from 0 through `self._n_iter`,
        inclusive. Iteration 0 represents the initial execution required to
        establish the initial states of the algorithm and its particles and
        does not represent an actual optimization iteration.

        The iteration flow must use `iter_until` as a synchronization
        control, preventing the processor from advancing beyond the
        iteration specified by this variable. The `stop_signal` control
        variable must be checked to allow the execution to be interrupted
        before subsequent iterations are performed.

        Before processing each iteration, the processor must call
        `insert_arrival_particle` to handle particles received from other
        processors and `remove_departure_particle` to handle particles that
        must leave the current population.

        The algorithm iteration must then be executed according to the
        algorithm's defined iteration lifecycle.
        """


def evaluate_particle(
    particle_id: str,
    algorithm: AlgorithmBase,
    stop_signal: LocalEvent,
    fitness_failure_strategy: str,
    initialize_particle: bool = False,
) -> ParticleBase:
    if stop_signal.is_set():
        return algorithm.get_unmodified_particle(particle_id)
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
