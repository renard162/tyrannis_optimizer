from collections.abc import Iterable
from functools import partial
from multiprocessing import Event
from typing import cast

from joblib import Parallel, delayed
from joblib.parallel import BACKENDS

from ..core.algorithm import CostFunctionWrapperBase, ParticleBase
from ..core.processor import ProcessorBase, evaluate_particle


class JoblibCostFunctionWrapper(CostFunctionWrapperBase):
    """Joblib processor cost-function wrapper."""


class Joblib(ProcessorBase):
    def __init__(
        self,
        n_jobs: int = -1,
        joblib_backend: str = "loky",
        batch_size: int | str = "auto",
        pre_dispatch: int | str = "2 * n_jobs",
    ) -> None:
        """
        Joblib-based processor for parallel particle evaluation.

        `Joblib` evaluates particles concurrently through the Joblib parallel
        execution framework. It is the preferred processor for parallel
        execution because it provides the broadest compatibility with different
        execution strategies and supports multiple parallel backends through a
        unified interface.

        Unlike `ThreadsPool` and `ProcessPool`, which are tied to a specific
        execution model, `Joblib` can select the execution strategy through
        `joblib_backend`. This allows the same processor to use processes,
        threads, or other execution mechanisms supported by the installed
        Joblib version, while also providing automatic task batching and
        control over how many tasks are dispatched ahead of execution.

        Parameters
        ----------
        n_jobs:
            Number of jobs executed concurrently. The default is `-1`, which
            uses all available CPUs. Positive values specify the exact number
            of jobs, with `1` disabling parallel execution. A value of `-1`
            uses all CPUs, while values below `-1` reserve CPUs from the total
            available: `-2` uses all but one CPU, `-3` uses all but two, and so
            on. The value `0` is invalid.

        joblib_backend:
            Joblib backend used to execute the parallel tasks. Supported backends
            include:

            - ``"loky"``: Process-based execution, avoiding the GIL for CPU-bound
            workloads. (default)
            - ``"threading"``: Thread-based execution, suitable for workloads that
            release the GIL.
            - ``"multiprocessing"``: Process-based execution using Python's
            multiprocessing, with broad compatibility for CPU-bound workloads.
            - ``"serialized"``: Sequential execution without parallelism.

        batch_size:
            Number of particle evaluations submitted as a single batch to each
            worker. If `"auto"`, Joblib dynamically adjusts the batch size to
            target batches with an execution time of approximately half a
            second. Smaller values provide finer workload distribution and can
            improve load balancing when evaluation times vary, while larger
            values reduce scheduling overhead when evaluations are relatively
            uniform. The default is `"auto"`.

        pre_dispatch:
            Number of batches that are pre-dispatched before workers begin
            completing tasks. It may be specified as a positive integer or as a
            string expression such as `"2 * n_jobs"`, which is the default. A
            larger value can keep workers supplied with tasks more consistently,
            at the cost of greater memory consumption and earlier task creation.
            A smaller value limits the amount of work queued ahead of execution.

        Notes
        -----
        The behavior and requirements of the cost function depend on the selected
        Joblib backend. With process-based backends, the cost function and all
        objects required for its evaluation must be serializable by the mechanism
        used by Joblib. With thread-based backends, worker threads share the same
        memory space as the main process, so serialization is not required for
        communication between workers; however, the cost function must be
        thread-safe and reentrant when evaluations can execute concurrently.

        The cost function must be independent between particle evaluations. It
        should not rely on execution order or modify shared mutable state in a
        manner that allows one evaluation to affect another. Any shared resources,
        such as files, database connections, random-number generators, caches, or
        other mutable objects, must either support concurrent access safely or be
        independently managed by each worker.

        The `joblib_backend` therefore provides an important trade-off between
        process-based and thread-based execution. Process-based backends avoid
        the GIL for CPU-bound Python code but require objects involved in the
        evaluation to be transferable to worker processes. Thread-based backends
        avoid this serialization and can be more efficient when the workload is
        I/O-bound or relies primarily on native code that releases the GIL.

        `batch_size` and `pre_dispatch` control different aspects of Joblib's
        scheduling behavior. `batch_size` determines how many individual tasks
        are grouped into a batch, whereas `pre_dispatch` determines how many
        batches are submitted ahead of execution. Their optimal values depend on
        the relative cost of particle evaluation, variability between evaluation
        times, and the overhead of task scheduling and serialization.
        """
        if not isinstance(n_jobs, int) or isinstance(n_jobs, bool):
            raise TypeError("n_jobs must be an integer.")

        if n_jobs == 0:
            raise ValueError("n_jobs cannot be zero.")

        if not isinstance(joblib_backend, str):
            raise TypeError("joblib_backend must be a string.")

        if joblib_backend not in BACKENDS:
            available_backends = ", ".join(sorted(BACKENDS))
            raise ValueError(
                f"Invalid Joblib backend {joblib_backend!r}. "
                f"Available backends are: {available_backends}."
            )

        if isinstance(batch_size, bool):
            raise TypeError("batch_size must be a positive integer or 'auto'.")

        if isinstance(batch_size, int):
            if batch_size <= 0:
                raise ValueError("batch_size must be greater than zero.")
        elif batch_size != "auto":
            raise ValueError("batch_size must be a positive integer or 'auto'.")

        if isinstance(pre_dispatch, bool):
            raise TypeError("pre_dispatch must be a positive integer or a string.")

        if isinstance(pre_dispatch, int):
            if pre_dispatch <= 0:
                raise ValueError("pre_dispatch must be greater than zero.")
        elif not isinstance(pre_dispatch, str):
            raise TypeError("pre_dispatch must be a positive integer or a string.")

        self._n_process = n_jobs
        self._joblib_backend = joblib_backend
        self._batch_size = batch_size
        self._pre_dispatch = pre_dispatch

        self._cost_function_wrapper = JoblibCostFunctionWrapper

    def initialize_execution_context(self) -> None:
        self.start_migration()

    def finalize_execution_context(self) -> None:
        self.stop_migration()

    def initialize_loop_context(self) -> None:
        self._migration_signal = Event()

        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.initialize_loop_context(
            migration_signal=self._migration_signal
        )

    def finalize_loop_context(self) -> None:
        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.finalize_loop_context()
        self._migration_signal = None

    def run(self) -> None:
        self.init_particles()
        self.initialize_loop_context()

        initialize_worker = partial(
            evaluate_particle,
            algorithm=self._algorithm,
            fitness_failure_strategy=self._fitness_failure_strategy,
            initialize_particle=True,
            second_update=False,
        )

        update_worker = partial(
            evaluate_particle,
            algorithm=self._algorithm,
            fitness_failure_strategy=self._fitness_failure_strategy,
            initialize_particle=False,
            second_update=False,
        )

        second_update_worker = partial(
            evaluate_particle,
            algorithm=self._algorithm,
            fitness_failure_strategy=self._fitness_failure_strategy,
            initialize_particle=False,
            second_update=True,
        )

        try:
            with Parallel(
                n_jobs=self._n_process,
                backend=self._joblib_backend,
                batch_size=cast(str, self._batch_size),
                pre_dispatch=cast(str, self._pre_dispatch),
                return_as="list",
            ) as parallel:
                for actual_iter in range(self._n_iter + 1):
                    self.migration_control(actual_iter)

                    self._algorithm.pre_iteration(actual_iter)
                    self.pre_iteration_log(actual_iter)

                    new_particles_ids = self._algorithm.new_particles_id

                    if new_particles_ids:
                        self._algorithm.create_random_cache(
                            particle_ids=new_particles_ids, initialize=True
                        )

                        new_particles = parallel(
                            delayed(initialize_worker)(particle_id)
                            for particle_id in new_particles_ids
                        )
                        self.error_log(
                            actual_iter=actual_iter,
                            updated_particles=cast(
                                Iterable[ParticleBase], new_particles
                            ),
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
                        processed_particles = parallel(
                            delayed(update_worker)(particle_id)
                            for particle_id in population
                        )
                        self.error_log(
                            actual_iter=actual_iter,
                            updated_particles=cast(
                                Iterable[ParticleBase], processed_particles
                            ),
                        )
                        self._algorithm.update_population(
                            cast(Iterable[ParticleBase], processed_particles)
                        )

                        if self._algorithm.double_particle_check:
                            self._algorithm.inter_iteration(actual_iter)
                            double_check_ids = list(self._algorithm.double_check_ids)

                            if double_check_ids:
                                self._algorithm.create_random_cache(
                                    particle_ids=double_check_ids, initialize=False
                                )
                                processed_particles = parallel(
                                    delayed(second_update_worker)(particle_id)
                                    for particle_id in double_check_ids
                                )
                                self.error_log(
                                    actual_iter=actual_iter,
                                    updated_particles=cast(
                                        Iterable[ParticleBase], processed_particles
                                    ),
                                )
                                self._algorithm.update_population(
                                    cast(Iterable[ParticleBase], processed_particles)
                                )

                    self._algorithm.post_iteration(actual_iter)
                    self.iteration_log(actual_iter)

                    self.update_status()
                    self.best_log(actual_iter)

        finally:
            self.finalize_loop_context()
