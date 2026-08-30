import json
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass
from time import sleep
from typing import Any, Generic, Protocol, Self, TypeVar

import numpy as np

from ...algorithm.base import AlgorithmBase, ParticleBase


class SignalProtocol(Protocol):
    """Just to use in _update_particle signature"""

    def is_set(self) -> bool: ...
    def set(self) -> None: ...
    def clear(self) -> None: ...


SignalType = TypeVar("SignalType", bound=SignalProtocol)


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
    _processors_pool: list[Self]

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the processor.

        The implementation must initialize the processor-specific state required
        by the execution strategy. Runtime-specific variables, including the
        `stop_signal` and `wait_signal`, must not be initialized by this method.
        These variables are initialized when the processor execution context is
        created or when the processor is replicated.

        Implementations must define their class-specific initialization without
        relying on a call to the `ProcessorBase` initializer.
        """

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        """
        Create a deep copy of the processor excluding execution control state.

        The default implementation recursively deep-copies the processor's
        instance attributes, except for `_stop_signal` and `_wait_signal`. These
        attributes represent execution-specific control state and must not be
        propagated to replicated processors.

        Both control signals must provide an `is_set` method that returns `True`
        when the signal is active and `False` otherwise, as well as `set` and
        `clear` methods to activate and deactivate the signal. Their concrete
        implementations must be appropriate for the processor's execution
        strategy and must be recreated by the subclass rather than copied from
        the original processor.

        The `stop_signal` controls termination of the processor's execution,
        while the `wait_signal` controls synchronization with other execution
        contexts. Their state must therefore be independent for each replicated
        processor.

        Processor implementations that require specific handling of
        synchronization primitives or other non-copyable execution resources must
        override this method. Subclasses may call this implementation through
        `super().__deepcopy__()` and initialize the excluded attributes according
        to their execution strategy.
        """
        new_processor = self.__class__.__new__(self.__class__)
        memo[id(self)] = new_processor

        excluded_attributes = {
            "_stop_signal",
            "_wait_signal",
            "_seed_sequence",
            "_processors_pool",
        }

        for name, value in self.__dict__.items():
            if name not in excluded_attributes:
                setattr(new_processor, name, deepcopy(value, memo))

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

        self._identifier = "ProcessorBase"
        self._control = ControlVariables()
        self._status = StatusVariables(population={})
        self._seed_sequence = np.random.SeedSequence(seed)
        self._processors_pool = []

    def create_processors_pool(self, n_islands: int) -> None:
        self._processors_pool = [
            self._replicate_processor(f"{idx + 1}") for idx in range(n_islands)
        ]

    def _replicate_processor(self, identifier: str) -> Self:
        new_processor = deepcopy(self)
        new_processor._identifier = identifier
        new_processor._algorithm.configure(
            identifier=f"island:{identifier}|algorithm",
            seed=self._seed_sequence.spawn(1)[0],
        )
        return new_processor

    @property
    def processors_pool(self) -> list[Self]:
        return self._processors_pool

    @property
    def local_best(self) -> str:
        return self._status.best_particle_data

    def init_particles(self) -> None:
        if len(self._algorithm.population) > 0:
            return

        for p_idx in range(self._n_particles):
            self._algorithm.create_particle(
                identifier=f"island:{self._identifier}|particle:{p_idx}",
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
    stop_signal: SignalProtocol,
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
