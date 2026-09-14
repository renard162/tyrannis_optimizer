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
            Multiprocessing start method used to create worker processes. If `None`,
            `"spawn"` is used on both Windows and Linux. On Windows, `"spawn"` is
            the only available context. On Linux, if `"spawn"` cannot be used,
            `"forkserver"` is preferred over `"fork"`.

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
        Unlike thread-based processors, worker processes do not share the same memory
        space as the main process. Data required by particle evaluation must therefore
        be serialized when it is transferred to worker processes, and results must be
        serialized when they are returned to the main process. This introduces
        serialization and inter-process communication overhead that should be
        considered when choosing `ProcessPool`.

        The cost function is executed independently in multiple worker processes and
        must be safe to execute in separate processes. It must be serializable by
        `cloudpickle`, including any objects captured through closures, and all data
        required by the function must likewise be serializable. The function should
        not depend on mutable state shared with the main process or with other worker
        processes, since ordinary Python objects are not shared between processes.
        Changes made to process-local state are therefore not visible to other
        workers or to the main process unless an explicit inter-process communication
        mechanism is used.

        The cost function should be deterministic with respect to its explicit inputs
        and its process-local state, and evaluations must not depend on the execution
        order of other particles. If the function uses external resources such as
        files, databases, network connections, or other system resources, those
        resources must be safely usable from independent processes, with each process
        managing its own process-local resources when necessary. Resources or objects
        that cannot be safely serialized or independently initialized in worker
        processes should not be captured by the cost function.

        `ProcessPool` is generally most useful for CPU-bound cost functions, especially
        when the function performs substantial Python-level computation. Unlike
        threads, separate processes are not subject to the Global Interpreter Lock
        (GIL) of the main process, allowing CPU-bound Python code to execute in
        parallel. However, the serialization and inter-process communication overhead
        can make a process pool inefficient for very inexpensive cost functions.

        The `multiprocessing_context` defaults to `"spawn"` regardless of the
        platform. On Windows, `"spawn"` is the only supported context. On Linux,
        `"spawn"` is generally preferred because worker processes start with a clean
        interpreter state. When the cost function or required objects cannot be
        serialized for `"spawn"`, `"forkserver"` provides an alternative on systems
        that support it and is generally preferable to `"fork"`, as it avoids
        inheriting the full state of the main process while still allowing objects
        that cannot be serialized for spawning to be used in the worker processes.

        The `chunksize` parameter controls the trade-off between communication
        overhead and workload distribution. Smaller values provide finer load
        balancing when particle evaluation times vary significantly, while larger
        values can be more efficient when evaluations have similar execution times.
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
