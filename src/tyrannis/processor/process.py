from __future__ import annotations

from collections.abc import Callable
from functools import partial
from multiprocessing import Event, get_all_start_methods, get_context
from multiprocessing.synchronize import Event as EventProtocol
from typing import Any

import cloudpickle
import numpy as np

from ..core.algorithm import CostFunctionWrapperBase
from ..core.processor import (
    ProcessorBase,
    evaluate_particle,
)
from ..core.signals import LocalEvent


class ProcessPoolCostFunctionWrapper(CostFunctionWrapperBase):
    """Process pool processor cost-function wrapper with cloudpickle-based serialization."""

    def __init__(self, function: Callable[..., float]) -> None:
        self._serialized_function = cloudpickle.dumps(function)
        self._function: Callable[..., float] | None = None

    def __call__(self, *args: Any, **kwargs: Any) -> np.float64:
        if self._function is None:
            self._function = cloudpickle.loads(self._serialized_function)

        return self._function(*args, **kwargs)  # type: ignore

    def __reduce__(
        self,
    ) -> tuple[Callable[[bytes], ProcessPoolCostFunctionWrapper], tuple[bytes]]:
        return (
            ProcessPoolCostFunctionWrapper._restore,
            (self._serialized_function,),
        )

    @staticmethod
    def _restore(
        serialized_function: bytes,
    ) -> ProcessPoolCostFunctionWrapper:
        instance = object.__new__(ProcessPoolCostFunctionWrapper)
        instance._serialized_function = serialized_function
        instance._function = None

        return instance


class PoolSignal(LocalEvent):
    def __init__(self) -> None:
        self._signal: EventProtocol = Event()
        self._manager_signal: EventProtocol | None = None

    def set(self) -> None:
        self._signal.set()
        if self._manager_signal is not None:
            self._manager_signal.set()

    def clear(self) -> None:
        self._signal.clear()
        if self._manager_signal is not None:
            self._manager_signal.clear()

    def is_set(self) -> bool:
        return self._signal.is_set()

    def set_manager_signal(self, manager_signal: EventProtocol) -> None:
        self._manager_signal = manager_signal
        if self._manager_signal is None:
            raise RuntimeError("Manager signal cannot be None.")
        if self._signal.is_set():
            self._manager_signal.set()
        else:
            self._manager_signal.clear()

    def clear_manager_signal(self) -> None:
        self._manager_signal = None

    @property
    def manager_signal(self) -> EventProtocol:
        if self._manager_signal is None:
            raise RuntimeError("Manager signal has not been initialized.")
        return self._manager_signal


class ProcessPool(ProcessorBase):
    def __init__(
        self,
        n_jobs: int | None = None,
        multiprocessing_context: str | None = None,
        maxtasksperchild: int | None = None,
        chunksize: int | None = None,
    ) -> None:
        """
        Process-based processor for parallel particle evaluation.

        `ProcessPool` evaluates particles concurrently using a pool of worker
        processes. Each worker process has its own memory space, while the optimization
        algorithm and its migration system remain controlled by the main process.
        Particle evaluation and new-particle consolidation are distributed across the
        process pool, with the required data serialized when exchanged between the
        main process and worker processes.

        Parameters
        ----------
        n_jobs:
            Number of worker processes used to evaluate particles concurrently. If
            `None`, the number of worker processes is determined automatically from
            the number of logical CPUs available to the system, corresponding to
            `os.cpu_count()`.

        multiprocessing_context:
            Multiprocessing context used to create worker processes. Supported
            contexts include:

            - ``"spawn"``: Workers start in a fresh Python process. This is the the
            only option available on Windows. (default)
            - ``"forkserver"``: Workers are created through a dedicated server
            process, providing isolation similar to ``"spawn"`` with lower
            process-creation overhead.
            - ``"fork"``: Workers inherit the parent process state. Fast, but may
            cause issues with resources that are not fork-safe.

        maxtasksperchild:
            Maximum number of tasks that a worker process can complete before it is
            replaced with a new worker process. If `None`, worker processes are not
            replaced based on the number of completed tasks. Setting this parameter
            can be useful for limiting memory growth or releasing resources that are
            not reclaimed during the lifetime of a worker process.

        chunksize:
            Number of particles grouped into each task batch during parallel
            evaluation. If `None`, the chunk size is calculated automatically by the
            multiprocessing pool based on the number of tasks and worker processes.
            Smaller values provide finer workload distribution and can improve load
            balancing when evaluation times vary, while larger values reduce task
            scheduling and communication overhead when evaluations have similar
            execution times.

        Notes
        -----
        `ProcessPool` is most suitable for CPU-bound cost functions, particularly when
        each evaluation is sufficiently expensive to amortize the serialization and
        inter-process communication overhead. Unlike threads, worker processes have
        independent memory spaces and are not subject to the Global Interpreter Lock
        (GIL) of the main process.

        The cost function and all objects required to evaluate it must be serializable
        by `cloudpickle` when using `"spawn"` or `"forkserver"`. Although `cloudpickle`
        supports functions defined locally or as lambdas, objects captured by the
        function must also be serializable. The function should not depend on mutable
        global state: worker processes may not see the same state as the parent process.
        Constants defined at module level are safe, but data required by the evaluation
        should preferably be passed explicitly or initialized independently in each
        worker.

        The module that starts the optimizer must be safely importable by worker
        processes. In particular, the call that creates or runs the optimizer and
        therefore initializes the `ProcessPool` must be protected by
        `if __name__ == "__main__":`. The cost function should be defined outside this
        block so that worker processes can import it without executing the optimizer
        again. Consequently, `ProcessPool` should normally be used from a Python
        script rather than an interactive interpreter, where the `__main__` module
        cannot be imported reliably.

        The default `"spawn"` context provides clean process initialization and is
        supported on both Windows and Linux. On Linux, `"forkserver"` can be used when
        the cost function or its dependencies cannot be serialized for `"spawn"` and
        is generally preferable to `"fork"`. Regardless of the selected context,
        large amounts of data should not be unnecessarily transferred between
        processes, as serialization and inter-process communication can substantially
        reduce the benefit of parallel evaluation.
        """
        available_contexts = get_all_start_methods()

        if (multiprocessing_context is not None) and (
            multiprocessing_context not in available_contexts
        ):
            raise ValueError(
                f"Invalid multiprocessing context "
                f"'{multiprocessing_context}'. "
                f"Available contexts are: {', '.join(available_contexts)}."
            )

        if (maxtasksperchild is not None) and (maxtasksperchild <= 0):
            raise ValueError("maxtasksperchild must be greater than zero.")

        if (chunksize is not None) and (chunksize <= 0):
            raise ValueError("chunksize must be greater than zero.")

        self._n_process = n_jobs
        self._multiprocessing_context = "spawn"
        if multiprocessing_context is not None:
            self._multiprocessing_context = multiprocessing_context
        self._maxtasksperchild = maxtasksperchild
        self._chunksize = chunksize

        self._cost_function_wrapper = ProcessPoolCostFunctionWrapper

    def initialize_execution_context(self) -> None:
        self.start_migration()

    def finalize_execution_context(self) -> None:
        self.stop_migration()

    def initialize_loop_context(self) -> None:
        self._migration_signal = PoolSignal()
        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.initialize_loop_context(
            migration_signal=self._migration_signal,
        )

    def finalize_loop_context(self) -> None:
        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.finalize_loop_context()
        self._migration_signal = None

    def run(self) -> None:
        self.init_particles()
        self.initialize_loop_context()

        context = get_context(self._multiprocessing_context)

        with context.Manager() as manager:
            self._migration_signal.set_manager_signal(manager.Event())

            with context.Pool(
                processes=self._n_process,
                maxtasksperchild=self._maxtasksperchild,
            ) as pool:
                for actual_iter in range(self._n_iter + 1):
                    self.migration_control(actual_iter)

                    self._algorithm.pre_iteration(actual_iter)
                    self.pre_iteration_log(actual_iter)

                    new_particles_ids = self._algorithm.new_particles_id
                    if new_particles_ids:
                        self._algorithm.create_random_cache(
                            particle_ids=new_particles_ids,
                            initialize=True,
                        )
                        worker = partial(
                            evaluate_particle,
                            algorithm=self._algorithm,
                            fitness_failure_strategy=self._fitness_failure_strategy,
                            initialize_particle=True,
                        )
                        new_particles = pool.map(
                            worker,
                            new_particles_ids,
                            chunksize=self._chunksize,
                        )
                        self.error_log(
                            actual_iter=actual_iter,
                            updated_particles=new_particles,
                        )
                        new_particles = pool.map(
                            self._algorithm.consolidate_new_particles,
                            new_particles,
                            chunksize=self._chunksize,
                        )
                        self.new_particle_log(
                            actual_iter=actual_iter,
                            new_particles=new_particles,
                        )
                        self._algorithm.update_population(new_particles)

                    if actual_iter > 0:
                        self._algorithm.create_random_cache(
                            particle_ids=[idx for idx in self._algorithm.population],
                            initialize=False,
                        )
                        worker = partial(
                            evaluate_particle,
                            algorithm=self._algorithm,
                            fitness_failure_strategy=self._fitness_failure_strategy,
                            initialize_particle=False,
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

                    self._algorithm.post_iteration(actual_iter)
                    self.iteration_log(actual_iter)

                    self.update_status()
                    self.best_log(actual_iter)
            self.finalize_loop_context()
