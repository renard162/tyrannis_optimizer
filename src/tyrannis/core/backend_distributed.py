from abc import abstractmethod

import numpy as np

from .algorithm import AlgorithmBase
from .backend import BackendBase
from .backend_migration import MigrationDriverBase
from .processor import ProcessorBase


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
        -> collect local bests
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
    independent execution unit and receives its own algorithm configuration,
    particle population, communication processor, and migration processor.

    `execute` is responsible for starting the migration runtime, delegating
    processor execution to the concrete distributed execution mechanism, and
    collecting the local best result from each island. The exact mechanism
    used to distribute and execute the processors is backend-specific.

    The final result is not necessarily produced by a single island. After
    processor execution completes, `update_result` compares the valid local
    best results and selects the best solution found across all islands.

    Distributed backends may implement different execution mechanisms, such
    as Spark, multiprocessing, or other distributed runtimes, but they must
    preserve the lifecycle and component relationships defined by this class.
    """

    _local_bests: dict[str, dict[str, float | dict[str, float]] | None]
    _n_executors: int
    _migration: MigrationDriverBase

    @abstractmethod
    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase | None = None,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
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
        Consolidate the best results returned by all distributed islands.

        Each island independently maintains its local best solution during
        execution. After all processors have completed, `_local_bests`
        contains the result returned by each island.

        This method compares the valid local best results and stores the
        globally best candidate in `_result`.

        A local result with an invalid or unavailable fitness is ignored.
        Since Tyrannis minimizes the objective function, the candidate with
        the lowest valid fitness is selected.

        Notes
        -----
        This method performs result consolidation only. It does not modify
        the state of any processor, algorithm, or migration module.

        Concrete distributed backends are responsible for populating
        `_local_bests` after their execution mechanism has collected the
        processor results.
        """
        best_candidate = None
        best_fitness = np.inf

        for candidate in self._local_bests.values():
            if candidate is None:
                continue

            fitness = candidate["fitness"]

            if not isinstance(fitness, float):
                continue

            if fitness < best_fitness:
                best_fitness = fitness
                best_candidate = candidate

        if best_candidate is None:
            return

        self._result = {
            key: value
            for key, value in best_candidate.items()
            if key in self._result_keys
        }
