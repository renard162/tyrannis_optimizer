import json
from abc import ABC, abstractmethod
from collections.abc import Iterable

import numpy as np

from ..core.backend_migration import MigrationDriverBase
from ..core.processor import ProcessorBase
from .algorithm import AlgorithmBase, ParticleBase
from .backend import BackendBase
from .results import HistoryConfig


class ParallelBackendBase(BackendBase, ABC):
    """
    Base class for standalone parallel optimization backends.

    A parallel backend executes the optimization as a single island while
    delegating the parallel processing of its particles to the execution
    environment provided by the concrete backend.

    Unlike a distributed backend, this execution model has no concept of
    multiple islands, inter-island communication, migration, or
    `ProcessorBase`. The complete population belongs to a single optimization
    instance.

    The parallel backend does not contain an intermediate processor layer.
    The concrete backend submits particle processing to the parallel execution
    environment itself, while the cluster orchestrator, such as Spark or MPI,
    is responsible for distributing the computational workload among its
    workers.

    This differs from the distributed execution model, where the cluster
    orchestrator distributes the execution of processors and each
    `ProcessorBase` controls the optimization lifecycle of its own island. In
    a parallel backend, there is no processor controlling an island: the
    backend itself coordinates the execution and the external parallel runtime
    manages the distribution and scheduling of the computational work.

    The value of `n_particles` represents the complete initial population
    because this backend operates as a single island. The number of workers
    provided by the parallel execution environment does not change the
    population size.

    The backend integrates the optimization algorithm with the parallel
    execution environment. The algorithm remains responsible for defining
    the optimization lifecycle and the state transitions of its particles.
    The backend is responsible only for invoking the corresponding algorithm
    methods and executing particle-level operations through the parallel
    runtime.

    The general execution flow is:

        initialize_context
        -> init_particles
        -> loop:
            pre_iteration
            -> create_random_cache (new particles)
            -> initialize new particles in parallel
            -> consolidate new particles
            -> log new particles and errors
            -> update population
            -> create_random_cache (entire population)
            -> update particles in parallel
            -> log errors
            -> update population
            -> post_iteration
            -> log iteration
            -> log best
        -> update_result

    Particle initialization and particle updates may be executed concurrently
    by the concrete parallel execution environment. The backend must preserve
    the ordering and synchronization required between the algorithm lifecycle
    stages, while the cluster orchestrator is responsible for distributing
    and scheduling the actual computational work.

    The concrete backend defines how this parallel execution is implemented.
    For example, a Spark implementation may distribute particle operations
    through Spark tasks, while an MPI implementation may distribute them
    among MPI processes.

    The backend owns a single `ProcessorResult` instance. Its `result` field
    is updated dynamically as the algorithm progresses, while its `history`
    field is populated dynamically by the logging methods during execution.
    """

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None:
        """
        Initialize the parallel optimization context.

        The history configuration is stored by the backend because the
        parallel execution model does not use `ProcessorBase` and therefore
        owns its logging lifecycle directly.

        A single `ProcessorResult` is created for the execution. The same
        object is retained throughout the complete optimization lifecycle,
        with its `result` and `history` fields populated dynamically.
        """
        super().initialize_context(
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            processor=processor,
            migration=migration,
            fitness_failure_strategy=fitness_failure_strategy,
            seed=seed,
            history_config=history_config,
        )

    def init_particles(self) -> None:
        """
        Create the initial population for the single optimization island.

        If the algorithm already contains particles, no particles are created.
        Otherwise, exactly `_n_particles` particles are created and registered
        in the algorithm.

        This method only creates the particle objects. Particle initialization,
        including the evaluation or establishment of their initial state, is
        performed later through the normal algorithm lifecycle.
        """
        if self._algorithm.population:
            return

        for p_idx in range(self._n_particles):
            self._algorithm.create_particle(
                identifier=f"{self._identifier}|particle:{p_idx}",
            )

    def pre_iteration_log(self, actual_iter: int) -> None:
        """
        Record the population before an iteration.
        """
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
        """
        Record newly initialized and consolidated particles.
        """
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
        """
        Record particle fitness evaluation errors.

        A particle is recorded when its candidate fitness is infinite,
        indicating that its evaluation failed and was invalidated.
        """
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
        """
        Record the population after an optimization iteration.
        """
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
        """
        Record the best and worst particles for the current execution state.

        The recorded states are:

        - `local_best`: historical best particle of the algorithm;
        - `iter_best`: best particle of the current iteration;
        - `iter_worst`: worst particle of the current iteration.
        """
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

    def update_result(self) -> None:
        """
        Update the optimization result while preserving the accumulated
        history.

        The algorithm's `local_best` represents the best solution found by
        this single optimization island. Its serialized representation is
        stored in `self._result.result`.

        The existing `self._result.history` is intentionally left unchanged
        because history is constructed dynamically during execution.
        """
        self._result.result = (
            None if self._algorithm.local_best is None else self._algorithm.local_best()
        )

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the complete parallel optimization lifecycle.
        """
        raise NotImplementedError
