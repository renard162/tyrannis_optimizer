from abc import abstractmethod

import numpy as np

from .algorithm import AlgorithmBase
from .backend import BackendBase
from .backend_migration import MigrationDriverBase
from .processor import ProcessorBase
from .results import HistoryConfig, ProcessorResult


class DistributedBackendBase(BackendBase):
    """
    Base class for distributed optimization backends.

    A distributed backend executes an optimization as multiple independent
    islands. Each island owns its own `ProcessorBase`, algorithm instance,
    population, and migration processor. The backend running on the driver
    coordinates these islands through a `MigrationDriverBase` and the
    communication infrastructure associated with it.

    Unlike a standalone backend, a distributed backend does not execute the
    optimization population directly on the driver. Its responsibility is to
    construct and configure the distributed execution context, create the
    required processor instances, initialize the communication and migration
    infrastructure, delegate the execution of each processor to the concrete
    distributed execution mechanism, and consolidate the results returned by
    the islands.

    The distributed execution is configured through the following components:

    - `AlgorithmBase`: defines the optimization algorithm executed independently
      by each island.
    - `ProcessorBase`: defines the execution lifecycle of an individual
      island.
    - `MigrationDriverBase`: defines the driver-side migration strategy and
      creates the corresponding `MigrationProcessorBase` for each island.
    - `CommunicationDriverBase`: provides the driver-side communication
      infrastructure.
    - `CommunicationProcessorBase`: provides the communication infrastructure
      used by each island.

    The general configuration and execution flow is:

        __init__
        -> initialize_context
        -> create communication context
        -> create processor pool
        -> start migration
        -> execute processors through the distributed backend
        -> collect local results
        -> update result
        -> stop migration

    `initialize_context` first configures the common backend context and then
    connects the migration driver to the communication implementation required
    by the concrete distributed backend. The migration driver subsequently
    uses this communication context to create one migration processor for
    each island.

    The number of particles supplied to the backend represents the initial
    population size of each island. The backend's `n_executors` determines how
    many islands are created, so the distributed initial population contains
    `n_particles` particles per island.

    `init_processors` creates the processor pool. Each processor is an
    independent replica of the configured `ProcessorBase` and receives its own
    algorithm configuration, particle population, communication processor, and
    migration processor.

    `execute` is responsible for starting the migration runtime, delegating
    processor execution to the concrete distributed execution mechanism, and
    collecting the result object from each island. The exact mechanism used to
    distribute and execute the processors is backend-specific.

    The final result is consolidated by `update_result`. The method transfers
    the history accumulated by every processor into the backend result and
    selects the best solution among both the results returned by the current
    processors and the result already stored by the backend.

    Keeping the previously consolidated result as a candidate is necessary
    because `execute` may be called multiple times. This guarantees that a
    subsequent execution can only improve the final solution and cannot reset
    it with a worse result.

    Distributed backends may implement different execution mechanisms, such
    as Spark, multiprocessing, or other distributed runtimes, but they must
    preserve the lifecycle and component relationships defined by this class.
    """

    _local_bests: dict[str, ProcessorResult]
    _n_executors: int
    _migration: MigrationDriverBase

    @abstractmethod
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
        Initialize the distributed optimization context.

        This method must first initialize the common backend context through
        `BackendBase.initialize_context` and then configure the communication
        infrastructure required by the migration strategy.

        The concrete distributed backend is responsible for creating its
        driver-side communication implementation and providing it, together
        with the corresponding processor-side communication class and its
        configuration, to the migration driver.

        Parameters
        ----------
        algorithm:
            Optimization algorithm that will be independently executed by
            each island.

        n_iter:
            Number of optimization iterations.

        n_particles:
            Number of initial particles for each island.

            For a distributed backend, this value is applied independently to
            every island. The total initial number of particles is therefore
            `n_particles * n_executors`.

        migration:
            Driver-side migration strategy responsible for coordinating the
            migration protocol and creating the migration processor associated
            with each island.

        processor:
            Processor implementation used by each island. The processor is
            replicated by the distributed backend according to the number of
            islands.

        fitness_failure_strategy:
            Strategy used when evaluation of a particle's fitness fails.

        seed:
            Optional seed used to initialize the random state of the
            optimization. The processor is responsible for deriving
            independent random states for the replicated islands.

        Notes
        -----
        The communication objects required by the migration strategy are
        backend-specific and must be created by the concrete distributed
        backend.

        The migration driver itself is configured here but is not responsible
        for knowing which communication implementation will be used before
        this context is established.

        A concrete implementation must call
        `super().initialize_context(...)` before configuring the migration
        communication context. The base implementation establishes the
        algorithm, processor, migration, iteration, population, failure
        strategy, and seed configuration required by the distributed backend.
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

        """
        Configure the migration driver's communication context.

        The concrete implementation must create the communication driver and
        identify the communication processor class and configuration required
        by each distributed island.

        If the concrete backend supports an execution mode without migration,
        that behavior must be handled by the concrete backend rather than
        assumed by this base class.
        """

    def init_processors(self) -> None:
        """
        Create the processor pool representing the distributed islands.

        The number of processors created is determined by `_n_executors`.
        Each processor is an independent replica of the configured
        `ProcessorBase` and represents one optimization island.

        Raises
        ------
        RuntimeError
            If no processor has been configured for the backend.

        Notes
        -----
        Processor replication also creates the processor-side migration and
        communication modules associated with each island. These modules must
        remain independent from those belonging to other islands.
        """
        if self._processor is None:
            raise RuntimeError("Processor is not initialized.")

        self._processor.create_processors_pool(
            self._n_executors,
        )

    def update_result(self) -> None:
        """
        Consolidate the results returned by all distributed islands.

        `_local_bests` contains the complete `ProcessorResult` object returned
        by each processor. The history of every processor is transferred to
        the backend's `_result.history` and then cleared from the corresponding
        object in `_local_bests` to release the memory previously occupied by
        those history entries.

        The best solution is selected by comparing the `result` dictionaries
        contained in the current processor results with the result already
        stored in `_result.result`.

        The previously consolidated result is intentionally considered a
        candidate. This is required because the optimization execution may be
        performed multiple times. A new execution must therefore preserve the
        best solution found by previous executions if none of the new processor
        results improves it.

        Since Tyrannis minimizes the objective function, the candidate with the
        lowest valid fitness is selected.

        Notes
        -----
        The `ProcessorResult` objects in `_local_bests` are retained after
        consolidation, but their `history` lists are emptied. Their `result`
        objects remain available as the partial results returned by their
        respective processors.

        The backend history accumulates across calls to this method.
        """
        if self._result is None:
            self._result = ProcessorResult()

        best_result = self._result.result
        best_fitness = np.inf

        if best_result is not None:
            fitness = best_result.get("fitness")

            if isinstance(fitness, float) and fitness < best_fitness:
                best_fitness = fitness

        for processor_result in self._local_bests.values():
            self._result.history.extend(processor_result.history)
            processor_result.history.clear()

            candidate = processor_result.result

            if candidate is None:
                continue

            fitness = candidate.get("fitness")

            if not isinstance(fitness, float):
                continue

            if fitness < best_fitness:
                best_fitness = fitness
                best_result = candidate

        self._result.result = best_result

        migration_history = self._migration.consume_migration_history()
        self._result.history.extend(migration_history)
