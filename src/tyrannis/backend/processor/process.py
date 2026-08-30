from functools import partial
from multiprocessing import Event, get_all_start_methods, get_context
from multiprocessing.synchronize import Event as EventProtocol
from typing import Self

from .base import (
    ProcessorBase,
    SignalProtocol,
    evaluate_particle,
)


class StopSignal(SignalProtocol):
    def __init__(self) -> None:
        self._signal: EventProtocol = Event()
        self._manager_signal: SignalProtocol | None = None

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

    def set_manager_signal(self, manager_signal: SignalProtocol) -> None:
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
    def manager_signal(self) -> SignalProtocol:
        if self._manager_signal is None:
            raise RuntimeError("Manager signal has not been initialized.")
        return self._manager_signal


class ProcessPool(ProcessorBase):
    def __init__(
        self,
        n_process: int | None = None,
        multiprocessing_context: str | None = None,
        maxtasksperchild: int | None = None,
        chunksize: int | None = None,
    ) -> None:
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

        self._n_process = n_process
        self._multiprocessing_context = multiprocessing_context
        self._maxtasksperchild = maxtasksperchild
        self._chunksize = chunksize

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        new_processor = super().__deepcopy__(memo)
        new_processor._stop_signal = StopSignal()
        new_processor._wait_signal = Event()
        return new_processor

    def run(self) -> None:
        self.init_particles()
        self._stop_signal.clear()

        context = get_context(self._multiprocessing_context)

        with context.Manager() as manager:
            self._stop_signal.set_manager_signal(manager.Event())

            with context.Pool(
                processes=self._n_process,
                maxtasksperchild=self._maxtasksperchild,
            ) as pool:
                self._status.n_process = pool._processes  # type: ignore

                for actual_iter in range(self._n_iter + 1):
                    self.update_iter_counter(actual_iter)

                    self.wait_sync(actual_iter)
                    if self._stop_signal.is_set():
                        break

                    self.migration_control()

                    self._algorithm.pre_iteration(actual_iter)

                    new_particles_ids = self._algorithm.new_particles_id
                    if new_particles_ids:
                        worker = partial(
                            evaluate_particle,
                            stop_signal=self._stop_signal.manager_signal,
                            algorithm=self._algorithm,
                            fitness_failure_strategy=self._fitness_failure_strategy,
                            initialize_particle=True,
                        )
                        new_particles = pool.map(
                            worker,
                            new_particles_ids,
                            chunksize=self._chunksize,
                        )
                        self._algorithm.update_population(new_particles)

                    if actual_iter > 0:
                        worker = partial(
                            evaluate_particle,
                            stop_signal=self._stop_signal.manager_signal,
                            algorithm=self._algorithm,
                            fitness_failure_strategy=self._fitness_failure_strategy,
                            initialize_particle=False,
                        )
                        processed_particles = pool.map(
                            worker,
                            self._algorithm.population,
                            chunksize=self._chunksize,
                        )
                        self._algorithm.update_population(processed_particles)

                    self._algorithm.post_iteration(actual_iter)

                    self.update_status()

            self._stop_signal.clear_manager_signal()
