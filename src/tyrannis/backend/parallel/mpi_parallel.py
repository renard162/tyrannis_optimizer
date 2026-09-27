from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import cloudpickle
from joblib import Parallel, delayed
from joblib.parallel import BACKENDS

from ...core.algorithm import AlgorithmBase, CostFunctionWrapperBase, ParticleBase
from ...core.backend_parallel import ParallelBackendBase
from ...core.processor import evaluate_particle

if TYPE_CHECKING:
    from mpi4py.futures import MPIPoolExecutor


class MPIParallelCostFunctionWrapper(CostFunctionWrapperBase):
    """MPI parallel cost-function wrapper."""


class MPIParallel(ParallelBackendBase):
    _DEPENDENCIES = ("mpi4py",)

    def __init__(
        self,
        batch_size: int | None = None,
        n_aux_jobs: int = -1,
        aux_backend: str = "sequential",
        aux_batch_size: int | str = "auto",
        aux_pre_dispatch: int | str = "2 * n_jobs",
    ) -> None:
        """
        MPI-based backend for fully parallel particle optimization.

        `MPIParallel` executes the optimization loop as a single fully
        parallelized process. The complete optimization state and iteration
        lifecycle remain on the MPI master process, while particle
        initialization and updates are distributed among MPI worker processes.

        The backend is intended to be executed through the Tyrannis MPI
        launcher under an MPI runtime. The MPI runtime determines the available
        ranks and their placement across the execution infrastructure, while the
        backend uses the worker pool exposed by `mpi4py.futures` without defining
        the cluster topology itself.

        Parameters
        ----------
        batch_size:
            Number of particles grouped into each MPI task. If `None`, the
            current particle set is divided into approximately one batch per MPI
            worker. Smaller values provide finer workload balancing when particle
            evaluation times vary, while larger values reduce serialization and
            communication overhead. The default is `None`.

        n_aux_jobs:
            Number of jobs used by the auxiliary Joblib parallel processing
            performed locally by the driver. Positive values specify the exact
            number of jobs, while negative values specify the number of jobs
            relative to the available logical CPUs. The default is `-1`.

        aux_backend:
            Joblib backend used for auxiliary parallel processing. The default
            is `"sequential"`. Other available Joblib backends may be used when
            appropriate for the local driver workload.

        aux_batch_size:
            Number of tasks grouped into each batch for auxiliary Joblib
            processing. If `"auto"`, Joblib determines the batch size
            automatically. The default is `"auto"`.

        aux_pre_dispatch:
            Number of auxiliary Joblib batches that may be dispatched ahead of
            execution. The default is `"2 * n_jobs"`.

        Notes
        -----
        `MPIParallel` maintains a single optimization population on the driver.
        Before each distributed particle-processing phase, the current algorithm
        state is serialized with `cloudpickle`. Each MPI task receives one batch
        of particle identifiers and an independent serialized snapshot of that
        state. The worker deserializes the snapshot once, processes every
        particle in its batch, and returns the resulting particles serialized as
        bytes.

        Only updated particles are returned to the driver. Worker-side algorithm
        mutations other than the returned particle states are therefore local to
        the task and are intentionally discarded, matching the standalone
        parallel-backend contract.

        MPI worker processes are kept available by the `mpi4py.futures`
        execution environment for the duration of the optimization. A single
        `MPIPoolExecutor` is created for the complete `execute` call and reused
        across initialization, normal updates, and second-update phases.

        Communication occurs at every distributed particle-processing phase.
        This backend is therefore most appropriate when particle evaluations are
        sufficiently expensive to amortize serialization, network transfer, and
        synchronization overhead.

        Auxiliary Joblib processing is independent of MPI and runs only on the
        driver. It is used for lower-cost local operations such as consolidating
        newly initialized particles.
        """
        self._check_dependencies("mpi")

        from mpi4py.futures import MPIPoolExecutor as mpi_pool_executor

        if isinstance(batch_size, bool):
            raise TypeError("batch_size must be a positive integer or None.")

        if (batch_size is not None) and not isinstance(batch_size, int):
            raise TypeError("batch_size must be a positive integer or None.")

        if (batch_size is not None) and (batch_size <= 0):
            raise ValueError("batch_size must be greater than zero.")

        if not isinstance(n_aux_jobs, int) or isinstance(n_aux_jobs, bool):
            raise TypeError("n_aux_jobs must be an integer.")

        if n_aux_jobs == 0:
            raise ValueError("n_aux_jobs cannot be zero.")

        if not isinstance(aux_backend, str):
            raise TypeError("aux_backend must be a string.")

        if aux_backend not in BACKENDS:
            available_backends = ", ".join(sorted(BACKENDS))
            raise ValueError(
                f"Invalid Joblib backend {aux_backend!r}. "
                f"Available backends are: {available_backends}."
            )

        if isinstance(aux_batch_size, bool):
            raise TypeError("aux_batch_size must be a positive integer or 'auto'.")

        if isinstance(aux_batch_size, int):
            if aux_batch_size <= 0:
                raise ValueError("aux_batch_size must be greater than zero.")
        elif aux_batch_size != "auto":
            raise ValueError("aux_batch_size must be a positive integer or 'auto'.")

        if isinstance(aux_pre_dispatch, bool):
            raise TypeError("aux_pre_dispatch must be a positive integer or a string.")

        if isinstance(aux_pre_dispatch, int):
            if aux_pre_dispatch <= 0:
                raise ValueError("aux_pre_dispatch must be greater than zero.")
        elif not isinstance(aux_pre_dispatch, str):
            raise TypeError("aux_pre_dispatch must be a positive integer or a string.")

        self._executor_class: type[MPIPoolExecutor] = mpi_pool_executor
        self._mpi_batch_size = batch_size

        self._n_process = n_aux_jobs
        self._joblib_backend = aux_backend
        self._aux_batch_size = aux_batch_size
        self._aux_pre_dispatch = aux_pre_dispatch

        self._identifier = "MPIParallel"
        self._cost_function_wrapper = MPIParallelCostFunctionWrapper

    def execute(self) -> None:
        self.init_particles()

        with self._executor_class() as executor:
            executor.bootup(wait=True)

            if executor.num_workers <= 0:
                raise RuntimeError(
                    "MPIParallel requires at least one MPI worker process."
                )

            with Parallel(
                n_jobs=self._n_process,
                backend=self._joblib_backend,
                batch_size=cast(str, self._aux_batch_size),
                pre_dispatch=cast(str, self._aux_pre_dispatch),
                return_as="list",
            ) as parallel:
                for actual_iter in range(self._n_iter + 1):
                    self._algorithm.pre_iteration(actual_iter)
                    self.pre_iteration_log(actual_iter)

                    new_particles_ids = self._algorithm.new_particles_id

                    if new_particles_ids:
                        self._algorithm.create_random_cache(
                            particle_ids=new_particles_ids, initialize=True
                        )
                        new_particles = self._parallel_initialize_particles(
                            executor=executor, particle_ids=new_particles_ids
                        )
                        self.error_log(
                            actual_iter=actual_iter, updated_particles=new_particles
                        )
                        new_particles = parallel(
                            delayed(self._algorithm.consolidate_new_particles)(particle)
                            for particle in new_particles
                        )
                        self.new_particle_log(
                            actual_iter=actual_iter,
                            new_particles=cast(Iterable[ParticleBase], new_particles),
                        )
                        self._algorithm.update_population(
                            cast(Iterable[ParticleBase], new_particles)
                        )

                    if actual_iter > 0:
                        population = list(self._algorithm.population)

                        self._algorithm.create_random_cache(
                            particle_ids=population, initialize=False
                        )
                        processed_particles = self._parallel_update_particles(
                            executor=executor, particle_ids=population
                        )
                        self.error_log(
                            actual_iter=actual_iter,
                            updated_particles=processed_particles,
                        )
                        self._algorithm.update_population(processed_particles)

                        if self._algorithm.double_particle_check:
                            self._algorithm.inter_iteration(actual_iter)
                            double_check_ids = list(self._algorithm.double_check_ids)

                            if double_check_ids:
                                self._algorithm.create_random_cache(
                                    particle_ids=double_check_ids, initialize=False
                                )
                                processed_particles = self._parallel_update_particles(
                                    executor=executor,
                                    particle_ids=double_check_ids,
                                    second_update=True,
                                )
                                self.error_log(
                                    actual_iter=actual_iter,
                                    updated_particles=processed_particles,
                                )
                                self._algorithm.update_population(processed_particles)

                    self._algorithm.post_iteration(actual_iter)
                    self.iteration_log(actual_iter)

                    self.update_result()
                    self.best_log(actual_iter)

    def _parallel_initialize_particles(
        self, executor: MPIPoolExecutor, particle_ids: list[str]
    ) -> list[ParticleBase]:
        return self._parallel_process_particles(
            executor=executor, particle_ids=particle_ids, initialize_particle=True
        )

    def _parallel_update_particles(
        self,
        executor: MPIPoolExecutor,
        particle_ids: list[str] | dict[str, ParticleBase],
        second_update: bool = False,
    ) -> list[ParticleBase]:
        if isinstance(particle_ids, dict):
            particle_ids = list(particle_ids)

        return self._parallel_process_particles(
            executor=executor,
            particle_ids=particle_ids,
            initialize_particle=False,
            second_update=second_update,
        )

    def _parallel_process_particles(
        self,
        executor: MPIPoolExecutor,
        particle_ids: list[str],
        initialize_particle: bool,
        second_update: bool = False,
    ) -> list[ParticleBase]:
        if not particle_ids:
            return []

        batch_size = self._mpi_batch_size

        if batch_size is None:
            batch_size = max(
                1,
                (len(particle_ids) + executor.num_workers - 1) // executor.num_workers,
            )

        particle_batches = [
            particle_ids[index : index + batch_size]
            for index in range(0, len(particle_ids), batch_size)
        ]

        serialized_algorithm = cloudpickle.dumps(self._algorithm)

        tasks = [
            (
                serialized_algorithm,
                particle_batch,
                initialize_particle,
                second_update,
                self._fitness_failure_strategy,
            )
            for particle_batch in particle_batches
        ]

        serialized_results = executor.starmap(
            _process_particle_batch, tasks, chunksize=1
        )

        particles: list[ParticleBase] = []

        for serialized_particles in serialized_results:
            batch_particles = cast(
                list[ParticleBase], cloudpickle.loads(serialized_particles)
            )
            particles.extend(batch_particles)

        return particles


def _process_particle_batch(
    serialized_algorithm: bytes,
    particle_ids: list[str],
    initialize_particle: bool,
    second_update: bool,
    fitness_failure_strategy: str,
) -> bytes:
    algorithm = cast(AlgorithmBase, cloudpickle.loads(serialized_algorithm))
    particles: list[ParticleBase] = []

    for particle_id in particle_ids:
        particle = evaluate_particle(
            particle_id=particle_id,
            algorithm=algorithm,
            fitness_failure_strategy=fitness_failure_strategy,
            initialize_particle=initialize_particle,
            second_update=second_update,
        )
        particles.append(particle)

    return cloudpickle.dumps(particles)
