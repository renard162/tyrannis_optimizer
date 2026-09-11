from __future__ import annotations

import json
from collections.abc import Callable
from queue import Empty
from threading import Event, Thread
from time import sleep
from typing import Any

import numpy as np

from ..core.backend_communication import CommunicationProcessorBase
from ..core.backend_migration import MigrationDriverBase, MigrationProcessorBase
from ..core.results import HistoryConfig
from ..core.signals import LocalEvent


class GlobalBestProcessor(MigrationProcessorBase):
    """Processor-side implementation of global-best migration."""

    MESSAGE_TYPE = "global_best_update"
    SYNCHRONIZATION_PAUSE = "synchronization_pause"
    SYNCHRONIZATION_RELEASE = "synchronization_release"

    def __init__(
        self,
        initial_iter: int,
        check_interval: int,
        synchronous: bool,
        communication_processor: CommunicationProcessorBase,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if initial_iter < 0:
            raise ValueError("initial_iter must be greater than or equal to 0.")

        if check_interval <= 0:
            raise ValueError("check_interval must be greater than 0.")

        self._initial_iter = initial_iter
        self._check_interval = check_interval
        self._synchronization_iter = initial_iter
        self._synchronous = synchronous
        self._communication_processor = communication_processor

        self._last_published_fitness: float | None = None
        self._migration_signal: LocalEvent | None = None
        self._population: dict[str, np.float64] | None = None

    def initialize_loop_context(self, migration_signal: LocalEvent) -> None:
        """Initialize resources required by the processor iteration loop."""

        self._migration_signal = migration_signal
        self._synchronization_iter = self._initial_iter
        self._last_published_fitness = None
        self._population = None
        self._communication_processor.set_message_signal(migration_signal)

    def finalize_loop_context(self) -> None:
        """Finalize resources associated with the processor iteration loop."""

        self._communication_processor.set_message_signal(None)
        self._migration_signal = None
        self._population = None

    def start(self) -> None:
        """Start the processor communication backend."""

        self._communication_processor.start()

    def stop(self) -> None:
        """Stop the migration processor communication backend."""

        self._communication_processor.stop()

    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, np.float64],
        iter_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Execute global-best migration control for the current iteration."""

        if self._migration_signal is None:
            raise RuntimeError("Migration loop context has not been initialized.")

        self._population = population

        self._consume_messages(
            population=population,
            insert_arrival_particle=insert_arrival_particle,
            departure_particle=departure_particle,
        )
        self._migration_signal.clear()

        if iter_best is None:
            return

        if self._synchronous:
            if actual_iter != self._synchronization_iter:
                return

            self._publish_if_new_best(iter_best=iter_best, actual_iter=actual_iter)
            return

        if actual_iter < self._synchronization_iter:
            return

        self._publish_if_new_best(iter_best=iter_best, actual_iter=actual_iter)

        while actual_iter >= self._synchronization_iter:
            self._synchronization_iter += self._check_interval

    def synchronization_control(
        self,
        actual_iter: int,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Block at synchronous checkpoints until the driver releases them."""

        if not self._synchronous:
            return

        if actual_iter < self._synchronization_iter:
            return

        if self._population is None:
            raise RuntimeError("Migration population has not been initialized.")

        checkpoint = self._synchronization_iter

        self._communication_processor.outgoing_queue.put(
            json.dumps({"type": self.SYNCHRONIZATION_PAUSE, "actual_iter": checkpoint})
        )

        while actual_iter >= self._synchronization_iter:
            message = self._communication_processor.messages.get()

            self._consume_message(
                message=message,
                population=self._population,
                insert_arrival_particle=insert_arrival_particle,
                departure_particle=departure_particle,
            )

    def _publish_if_new_best(self, iter_best: str, actual_iter: int) -> None:
        """Publish a local best candidate when it improves the last published best."""

        try:
            particle_data = json.loads(iter_best)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(particle_data, dict):
            return

        fitness = particle_data.get("fitness")

        if not isinstance(fitness, (int, float)):
            return

        fitness = float(fitness)

        if (
            self._last_published_fitness is not None
            and fitness >= self._last_published_fitness
        ):
            return

        self._communication_processor.outgoing_queue.put(
            json.dumps(
                {
                    "type": self.MESSAGE_TYPE,
                    "particle": particle_data,
                    "actual_iter": actual_iter,
                }
            )
        )
        self._last_published_fitness = fitness

    def _consume_messages(
        self,
        population: dict[str, np.float64],
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Consume all currently pending migration messages."""

        while True:
            try:
                message = self._communication_processor.messages.get_nowait()
            except Empty:
                return

            self._consume_message(
                message=message,
                population=population,
                insert_arrival_particle=insert_arrival_particle,
                departure_particle=departure_particle,
            )

    def _consume_message(
        self,
        message: str,
        population: dict[str, np.float64],
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        message_type = payload.get("type")

        if message_type == self.SYNCHRONIZATION_RELEASE:
            actual_iter = payload.get("actual_iter")
            if actual_iter is None:
                raise RuntimeError("actual_iter cannot be None!")

            if actual_iter == self._synchronization_iter:
                self._synchronization_iter = actual_iter + self._check_interval

            return

        if message_type != self.MESSAGE_TYPE:
            return

        particle_data = payload.get("particle")

        if not isinstance(particle_data, dict):
            return

        if particle_data.get("identifier") is None:
            return

        self._replace_worst_particle(
            population=population,
            particle_data=particle_data,
            insert_arrival_particle=insert_arrival_particle,
            departure_particle=departure_particle,
        )

    def _replace_worst_particle(
        self,
        population: dict[str, np.float64],
        particle_data: dict[str, Any],
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Replace the worst local particle with the arriving particle."""

        if not population:
            return

        valid_population = {
            particle_id: fitness
            for particle_id, fitness in population.items()
            if isinstance(fitness, (int, float))
        }

        if not valid_population:
            return

        worst_particle_id = max(
            valid_population,
            key=lambda particle_id: valid_population[particle_id],
        )

        arriving_particle = particle_data.copy()
        arriving_particle["identifier"] = worst_particle_id

        departure_particle(worst_particle_id)
        insert_arrival_particle(arriving_particle)

        population[worst_particle_id] = arriving_particle["fitness"]


class GlobalBest(MigrationDriverBase):
    """Global-best migration strategy."""

    _processor_class = GlobalBestProcessor

    def __init__(
        self, initial_iter: int = 1, check_interval: int = 1, synchronous: bool = False
    ) -> None:
        if initial_iter < 1:
            raise ValueError("initial_iter must be greater than or equal to 1.")

        if check_interval <= 0:
            raise ValueError("check_interval must be greater than 0.")

        self._migration_processor_init_kargs = {
            "initial_iter": initial_iter,
            "check_interval": check_interval,
            "synchronous": synchronous,
        }

        self._synchronous = synchronous
        self._global_best_fitness: float | None = None
        self._global_best_particle: dict[str, Any] | None = None

        self._synchronization_paused: dict[int, set[str]] = {}
        self._synchronization_particles: dict[int, dict[str, dict[str, Any]]] = {}
        self._synchronization_pending: set[int] = set()
        self._next_synchronization_iter = initial_iter

        self._running = Event()
        self._thread: Thread | None = None

    def initialize_context(
        self,
        communication_driver,
        communication_processor_class,
        communication_processor_kargs: dict[str, Any],
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None:
        super().initialize_context(
            communication_driver=communication_driver,
            communication_processor_class=communication_processor_class,
            communication_processor_kargs=communication_processor_kargs,
            history_config=history_config,
            seed=seed,
        )

    def start(self) -> None:
        """Start the communication driver and application-level router."""

        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Migration is already running.")

        self._global_best_fitness = None
        self._global_best_particle = None
        self._synchronization_paused.clear()
        self._synchronization_particles.clear()
        self._synchronization_pending.clear()
        self._next_synchronization_iter = self._migration_processor_init_kargs[
            "initial_iter"
        ]

        self._running.set()
        self._communication_driver.start()

        self._thread = Thread(
            target=self._routing_loop, name="global-best-migration", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the application-level router and communication driver."""

        self._running.clear()

        if self._thread is not None:
            self._thread.join()
            self._thread = None

        self._communication_driver.stop()

    def _routing_loop(self) -> None:
        while self._running.is_set():
            self._process_incoming()
            sleep(0.01)

        self._process_incoming()

    def _process_incoming(self) -> None:
        """Process pending messages from all islands."""

        for (
            island_id,
            incoming_queue,
        ) in self._communication_driver.incoming_queues.items():
            while True:
                try:
                    message = incoming_queue.get_nowait()
                except Empty:
                    break

                self._route_message(source_id=island_id, message=message)

    def _route_message(self, source_id: str, message: str) -> None:
        """Process a message received from an island."""

        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        message_type = payload.get("type")

        if message_type == GlobalBestProcessor.SYNCHRONIZATION_PAUSE:
            self._route_synchronization_pause(source_id=source_id, payload=payload)
            return

        if message_type != GlobalBestProcessor.MESSAGE_TYPE:
            return

        self._route_global_best(source_id=source_id, payload=payload)

    def _route_global_best(self, source_id: str, payload: dict[str, Any]) -> None:
        """Process a global-best candidate."""

        particle_data = payload.get("particle")

        if not isinstance(particle_data, dict):
            return

        actual_iter = payload.get("actual_iter")

        if not isinstance(actual_iter, int):
            return

        fitness = particle_data.get("fitness")

        if not isinstance(fitness, (int, float)):
            return

        fitness = float(fitness)

        if self._synchronous:
            if actual_iter != self._next_synchronization_iter:
                return

            self._synchronization_particles.setdefault(actual_iter, {})[source_id] = (
                particle_data.copy()
            )
            return

        if (
            self._global_best_fitness is not None
            and fitness >= self._global_best_fitness
        ):
            return

        self._global_best_fitness = fitness
        self._global_best_particle = particle_data.copy()

        self.migration_log(
            particle=particle_data,
            origin=source_id,
            destination="all",
            iteration=actual_iter,
        )

        self._broadcast_global_best(particle_data=particle_data, source_id=source_id)

    def _route_synchronization_pause(
        self, source_id: str, payload: dict[str, Any]
    ) -> None:
        """Register an island at a synchronous checkpoint."""

        actual_iter = payload.get("actual_iter")

        if not isinstance(actual_iter, int):
            return

        if actual_iter != self._next_synchronization_iter:
            return

        paused = self._synchronization_paused.setdefault(actual_iter, set())

        if source_id in paused:
            return

        paused.add(source_id)
        self._maybe_complete_synchronization(actual_iter=actual_iter)

    def _maybe_complete_synchronization(self, actual_iter: int) -> None:
        paused = self._synchronization_paused.get(actual_iter)

        if paused is None:
            return

        island_ids = set(self._communication_driver.outgoing_queues)

        if paused != island_ids:
            return

        if actual_iter in self._synchronization_pending:
            return

        self._synchronization_pending.add(actual_iter)
        self._complete_synchronization(actual_iter=actual_iter)

    def _complete_synchronization(self, actual_iter: int) -> None:
        """Complete a synchronous global-best checkpoint."""

        particles = self._synchronization_particles.pop(actual_iter, {})

        best_particle = self._global_best_particle
        best_fitness = self._global_best_fitness
        best_source: str | None = None

        for source_id, particle_data in particles.items():
            fitness = particle_data.get("fitness")

            if not isinstance(fitness, (int, float)):
                continue

            fitness = float(fitness)

            if best_fitness is None or fitness < best_fitness:
                best_particle = particle_data.copy()
                best_fitness = fitness
                best_source = source_id

        new_global_best = best_fitness is not None and (
            self._global_best_fitness is None
            or best_fitness < self._global_best_fitness
        )

        if best_particle is not None and best_fitness is not None:
            self._global_best_particle = best_particle.copy()
            self._global_best_fitness = best_fitness

            self._broadcast_global_best(
                particle_data=self._global_best_particle, source_id=best_source
            )

            if new_global_best and best_source is not None:
                self.migration_log(
                    particle=self._global_best_particle,
                    origin=best_source,
                    destination="all",
                    iteration=actual_iter,
                )

        release_message = json.dumps(
            {
                "type": GlobalBestProcessor.SYNCHRONIZATION_RELEASE,
                "actual_iter": actual_iter,
            }
        )

        for outgoing_queue in self._communication_driver.outgoing_queues.values():
            outgoing_queue.put(release_message)

        self._synchronization_paused.pop(actual_iter, None)
        self._synchronization_pending.discard(actual_iter)
        self._next_synchronization_iter = actual_iter + self._get_check_interval()

    def _broadcast_global_best(
        self, particle_data: dict[str, Any], source_id: str | None
    ) -> None:
        """Broadcast a global-best particle to all other islands."""

        message = json.dumps(
            {"type": GlobalBestProcessor.MESSAGE_TYPE, "particle": particle_data}
        )

        for (
            island_id,
            outgoing_queue,
        ) in self._communication_driver.outgoing_queues.items():
            if source_id is not None and island_id == source_id:
                continue

            outgoing_queue.put(message)

    def _get_check_interval(self) -> int:
        """Return the configured migration interval."""

        return self._migration_processor_init_kargs["check_interval"]
