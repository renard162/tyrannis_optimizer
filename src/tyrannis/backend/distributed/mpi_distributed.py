from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import cloudpickle

from ...core.algorithm import AlgorithmBase, CostFunctionWrapperBase
from ...core.backend_distributed import DistributedBackendBase
from ...core.backend_migration import MigrationDriverBase
from ...core.processor import ProcessorBase
from ...core.results import HistoryConfig, ProcessorResult
from .communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)

if TYPE_CHECKING:
    from mpi4py.futures import MPIPoolExecutor
    from mpi4py.MPI import Intracomm


class MPIDistributedCostFunctionWrapper(CostFunctionWrapperBase):
    """MPI distributed cost-function wrapper."""


class MPIDistributed(DistributedBackendBase):
    """MPI backend for distributed processor execution."""

    _DEPENDENCIES = ("mpi4py",)

    def __init__(self) -> None:
        """
        MPI backend for distributed island-based optimization.

        `MPIDistributed` executes one complete optimization processor for each
        MPI worker rank. Rank 0 remains on the driver and coordinates the
        distributed execution, while each worker rank receives an independent
        processor containing its own algorithm, population, migration processor,
        and local optimization loop.

        The number of optimization islands is determined by the MPI execution
        environment. Rank 0 is reserved for the driver and every remaining rank
        represents one island. Consequently, the backend creates
        `MPI.COMM_WORLD.Get_size() - 1` islands.

        The backend does not determine rank placement across physical machines.
        Infrastructure-level placement remains the responsibility of the MPI
        launcher or resource manager. When one worker rank is assigned to each
        worker machine, every machine receives one processor and the processor
        controls any additional parallel processing performed locally on that
        machine.

        Notes
        -----
        `MPIDistributed` must be created on MPI rank 0 inside an MPI execution
        containing at least one worker rank.

        Processor objects are serialized with `cloudpickle` before submission to
        MPI workers. Runtime resources are initialized only after a processor has
        reached its worker rank through `initialize_execution_context` and are
        released there through `finalize_execution_context`.

        The processor migration lifecycle remains active on every island. Until
        the MPI communication implementation is connected to this backend, the
        inactive communication implementation is used only to satisfy the
        communication contract required when migration processors are created.
        The distributed backend itself does not impose or inspect a particular
        migration strategy.
        """
        self._check_dependencies("mpi")

        from mpi4py import MPI as mpi
        from mpi4py.futures import MPIPoolExecutor as mpi_pool_executor

        communicator = mpi.COMM_WORLD

        if communicator.Get_rank() != 0:
            raise RuntimeError("MPIDistributed must be created on MPI rank 0.")

        n_executors = communicator.Get_size() - 1

        if n_executors <= 0:
            raise RuntimeError(
                "MPIDistributed requires at least one MPI worker rank in addition "
                "to the driver rank."
            )

        self._communicator: Intracomm = communicator
        self._executor_class: type[MPIPoolExecutor] = mpi_pool_executor
        self._n_executors = n_executors

        self._identifier = "MPIDistributed"
        self._cost_function_wrapper = MPIDistributedCostFunctionWrapper

        self._local_bests = {}

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase[Any] | None,
        fitness_failure_strategy: str,
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None:
        super().initialize_context(
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration=migration,
            processor=processor,
            fitness_failure_strategy=fitness_failure_strategy,
            seed=seed,
            history_config=history_config,
        )

        island_ids = [f"island:{idx}" for idx in range(self._n_executors)]

        communication_driver = NoCommunicationDriver(island_ids=island_ids)

        self._migration.initialize_context(
            communication_driver=communication_driver,
            communication_processor_class=NoCommunicationProcessor,
            communication_processor_kargs={},
            history_config=self._history_config,
            seed=self._seed,
        )

    def execute(self) -> None:
        if self._processor is None:
            raise RuntimeError("Processor cannot be None.")

        with self._executor_class() as executor:
            executor.bootup(wait=True)

            if executor.num_workers != self._n_executors:
                raise RuntimeError(
                    "The MPI worker pool does not match the number of worker ranks "
                    "available in MPI.COMM_WORLD."
                )

            if not self._processor.processors_pool:
                self.init_processors()

            processors = list(self._processor.processors_pool.values())

            if len(processors) != executor.num_workers:
                raise RuntimeError(
                    "The number of distributed processors must match the number "
                    "of MPI worker ranks."
                )

            serialized_processors = [
                cloudpickle.dumps(processor) for processor in processors
            ]

            self._migration.start()

            try:
                futures = [
                    executor.submit(_run_processor, serialized_processor)
                    for serialized_processor in serialized_processors
                ]

                results = [future.result() for future in futures]

                self._local_bests = dict(results)
                self.update_result()

            finally:
                self._migration.stop()


def _run_processor(serialized_processor: bytes) -> tuple[str, ProcessorResult]:
    """
    Execute one complete optimization processor inside an MPI worker rank.

    Exactly one processor task is submitted for each MPI worker. The worker
    barrier ensures that every worker rank has received one processor before
    any island begins its optimization loop, preserving the persistent
    one-processor-per-worker execution model.

    The processor enters its execution context only after reaching the worker
    rank. Runtime resources are therefore created in the environment in which
    the processor actually executes and are finalized there after execution.
    """
    from mpi4py.futures import get_comm_workers

    worker_communicator = get_comm_workers()
    worker_communicator.Barrier()

    processor = cast(ProcessorBase[Any], cloudpickle.loads(serialized_processor))

    processor.initialize_execution_context()

    try:
        processor.run()

        return processor.identifier, processor.result

    finally:
        processor.finalize_execution_context()
