from functools import partial
from multiprocessing import Event
from multiprocessing.pool import ThreadPool

from ..core.algorithm import CostFunctionWrapperBase
from ..core.processor import ProcessorBase, evaluate_particle
from ..core.signals import EventProtocol


class ThreadsPoolCostFunctionWrapper(CostFunctionWrapperBase):
    """Threads pool processor cost-function wrapper."""


class ThreadsPool(ProcessorBase[EventProtocol]):
    def __init__(self, n_jobs: int | None = None, chunksize: int | None = None) -> None:
        """
        Thread-based processor for parallel particle evaluation.

        `ThreadsPool` evaluates particles concurrently using a pool of worker
        threads. The optimization algorithm and its migration system remain
        controlled by the main execution thread, while particle evaluation and
        new-particle consolidation are distributed across the thread pool.

        Parameters
        ----------
        n_jobs:
            Number of worker threads used to evaluate particles concurrently. If
            `None`, all logical CPUs available to the system are used.

        chunksize:
            Number of particles grouped into each task batch during parallel
            evaluation. If `None`, the chunk size is calculated automatically by
            the thread pool based on the number of particles and worker threads.
            Larger values reduce task-scheduling overhead, while smaller values
            provide finer workload distribution and can improve load balancing
            when evaluation times vary.

        Notes
        -----
        Unlike process-based processors, worker threads share the same memory
        space as the main process. Consequently, particles, the optimization
        algorithm, the search space, and the cost function do not need to be
        serialized and transferred between processes for each evaluation.

        The cost function is executed concurrently by multiple worker threads and
        must therefore be thread-safe and reentrant. It must safely support
        simultaneous calls from different threads and must not rely on shared
        mutable state that can be accessed or modified concurrently without
        proper synchronization. Resources shared between evaluations, such as
        files, database connections, random-number generators, or other mutable
        objects, must likewise support concurrent access or be independently
        managed by each thread. Evaluations must not depend on the execution order
        of other particles or modify shared state in a way that allows one
        evaluation to affect another.

        The performance benefit of increasing `n_jobs` depends on the cost
        function. CPU-bound Python code that does not release the Global
        Interpreter Lock (GIL) generally does not scale with additional threads,
        whereas I/O-bound operations and computations performed by native
        libraries that release the GIL can benefit from concurrent execution.

        The `chunksize` parameter controls the trade-off between task-scheduling
        overhead and workload distribution. Smaller values provide finer
        load balancing when particle evaluation times vary significantly, while
        larger values can be more efficient when evaluations have similar
        execution times.
        """
        if (chunksize is not None) and (chunksize <= 0):
            raise ValueError("chunksize must be greater than zero.")

        self._n_process = n_jobs
        self._chunksize = chunksize

        self._cost_function_wrapper = ThreadsPoolCostFunctionWrapper

    def initialize_execution_context(self) -> None:
        self.start_migration()

    def finalize_execution_context(self) -> None:
        self.stop_migration()

    def initialize_loop_context(self) -> None:
        migration_processor = self._migration_processor

        if migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        migration_signal: EventProtocol = Event()
        self._migration_signal = migration_signal

        migration_processor.initialize_loop_context(migration_signal=migration_signal)

    def finalize_loop_context(self) -> None:
        migration_processor = self._migration_processor

        if migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        migration_processor.finalize_loop_context()
        self._migration_signal = None

    def run(self) -> None:
        self.init_particles()
        self.initialize_loop_context()

        try:
            with ThreadPool(processes=self._n_process) as pool:
                for actual_iter in range(self._n_iter + 1):
                    self.migration_control(actual_iter)

                    self._algorithm.pre_iteration(actual_iter)
                    self.pre_iteration_log(actual_iter)

                    new_particles_ids = self._algorithm.new_particles_id

                    if new_particles_ids:
                        self._algorithm.create_random_cache(
                            particle_ids=new_particles_ids, initialize=True
                        )

                        worker = partial(
                            evaluate_particle,
                            algorithm=self._algorithm,
                            fitness_failure_strategy=self._fitness_failure_strategy,
                            initialize_particle=True,
                            second_update=False,
                        )

                        new_particles = pool.map(
                            worker, new_particles_ids, chunksize=self._chunksize
                        )

                        self.error_log(
                            actual_iter=actual_iter, updated_particles=new_particles
                        )

                        new_particles = pool.map(
                            self._algorithm.consolidate_new_particles,
                            new_particles,
                            chunksize=self._chunksize,
                        )

                        self.new_particle_log(
                            actual_iter=actual_iter, new_particles=new_particles
                        )

                        self._algorithm.update_population(new_particles)

                    if actual_iter > 0:
                        self._algorithm.create_random_cache(
                            particle_ids=list(self._algorithm.population),
                            initialize=False,
                        )

                        worker = partial(
                            evaluate_particle,
                            algorithm=self._algorithm,
                            fitness_failure_strategy=self._fitness_failure_strategy,
                            initialize_particle=False,
                            second_update=False,
                        )

                        processed_particles = pool.map(
                            worker,
                            self._algorithm.population,
                            chunksize=self._chunksize,
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

                                worker = partial(
                                    evaluate_particle,
                                    algorithm=self._algorithm,
                                    fitness_failure_strategy=(
                                        self._fitness_failure_strategy
                                    ),
                                    initialize_particle=False,
                                    second_update=True,
                                )

                                processed_particles = pool.map(
                                    worker, double_check_ids, chunksize=self._chunksize
                                )

                                self.error_log(
                                    actual_iter=actual_iter,
                                    updated_particles=processed_particles,
                                )

                                self._algorithm.update_population(processed_particles)

                    self._algorithm.post_iteration(actual_iter)
                    self.iteration_log(actual_iter)

                    self.update_status()
                    self.best_log(actual_iter)
        finally:
            self.finalize_loop_context()
