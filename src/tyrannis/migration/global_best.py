from __future__ import annotations

import json
from collections.abc import Callable
from queue import Empty
from threading import Event, Thread
from time import sleep
from typing import Any

from ..core.backend_communication import CommunicationProcessorBase
from ..core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
from ..core.results import HistoryConfig
from ..core.signals import LocalEvent


class GlobalBestProcessor(MigrationProcessorBase):
    """Migration processor for global-best migration.

    Islands execute independently when ``synchronous`` is false. At each
    ``check_interval`` point, starting at ``initial_iter``, the island publishes
    its iteration best when that best is better than the best it has previously
    published.

    When ``synchronous`` is true, all islands stop at each synchronization point
    and wait until the driver has received the synchronization message from all
    islands. The driver then broadcasts the global best and releases all
    islands to continue until the next synchronization point.

    When a global-best particle arrives, it replaces the worst particle in the
    receiving island. The identifier of the local particle being replaced is
    preserved, while the state of the arriving particle is copied into that
    identifier.

    The communication protocol is transport-independent. The communication
    backend exposes only strings through its queues.
    """

    MESSAGE_TYPE = "global_best_update"
    SYNC_READY_MESSAGE_TYPE = "sync_ready"
    SYNC_RELEASE_MESSAGE_TYPE = "sync_release"

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
        self._synchronization_released = False

    def initialize_loop_context(
        self,
        migration_signal: LocalEvent,
    ) -> None:
        """Initialize resources required by the processor iteration loop."""

        self._migration_signal = migration_signal

        self._communication_processor.set_message_signal(
            migration_signal,
        )

    def finalize_loop_context(self) -> None:
        """Finalize resources associated with the processor iteration loop."""

        self._communication_processor.set_message_signal(None)
        self._migration_signal = None

    def start(self) -> None:
        """Start the processor communication backend."""

        self._communication_processor.start()

    def stop(self) -> None:
        """Stop the migration processor communication backend."""

        self._communication_processor.stop()

    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, float],
        iter_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Execute global-best migration control."""

        if self._migration_signal is None:
            raise RuntimeError("Migration loop context has not been initialized.")

        self._consume_messages(
            population=population,
            insert_arrival_particle=insert_arrival_particle,
            departure_particle=departure_particle,
        )

        self._migration_signal.clear()

        if self._synchronization_iter is None:
            return

        if actual_iter < self._synchronization_iter:
            return

        if self._synchronous:
            self._synchronize(
                actual_iter=actual_iter,
                iter_best=iter_best,
                population=population,
                insert_arrival_particle=insert_arrival_particle,
                departure_particle=departure_particle,
            )
            return

        if iter_best is not None:
            self._publish_if_new_best(
                iter_best=iter_best,
                actual_iter=actual_iter,
            )

        while actual_iter >= self._synchronization_iter:
            self._synchronization_iter += self._check_interval

    def _synchronize(
        self,
        actual_iter: int,
        iter_best: str | None,
        population: dict[str, float],
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Wait for the driver to release the current synchronization point."""

        if iter_best is not None:
            try:
                particle_data = json.loads(iter_best)
            except (TypeError, json.JSONDecodeError):
                particle_data = None

            if isinstance(particle_data, dict):
                fitness = particle_data.get("fitness")

                if isinstance(fitness, (int, float)):
                    self._communication_processor.outgoing_queue.put(
                        json.dumps(
                            {
                                "type": self.SYNC_READY_MESSAGE_TYPE,
                                "actual_iter": actual_iter,
                                "particle": particle_data,
                            }
                        )
                    )
                else:
                    self._send_sync_ready(actual_iter)
            else:
                self._send_sync_ready(actual_iter)
        else:
            self._send_sync_ready(actual_iter)

        self._synchronization_released = False

        while not self._synchronization_released:
            self._consume_messages(
                population=population,
                insert_arrival_particle=insert_arrival_particle,
                departure_particle=departure_particle,
            )

            if self._synchronization_released:
                break

            sleep(0.001)

        self._synchronization_released = False

    def _send_sync_ready(
        self,
        actual_iter: int,
    ) -> None:
        """Notify the driver that the synchronization point was reached."""

        self._communication_processor.outgoing_queue.put(
            json.dumps(
                {
                    "type": self.SYNC_READY_MESSAGE_TYPE,
                    "actual_iter": actual_iter,
                    "particle": None,
                }
            )
        )

    def _publish_if_new_best(
        self,
        iter_best: str,
        actual_iter: int,
    ) -> None:
        """Publish the iteration best when it improves the last published best."""

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

        message = json.dumps(
            {
                "type": self.MESSAGE_TYPE,
                "particle": particle_data,
                "actual_iter": actual_iter,
            }
        )

        self._communication_processor.outgoing_queue.put(message)
        self._last_published_fitness = fitness

    def _consume_messages(
        self,
        population: dict[str, float],
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Consume all currently pending migration messages."""

        while True:
            try:
                message = self._communication_processor.messages.get_nowait()
            except Empty:
                return

            try:
                payload = json.loads(message)
            except (TypeError, json.JSONDecodeError):
                continue

            if not isinstance(payload, dict):
                continue

            message_type = payload.get("type")

            if message_type == self.SYNC_RELEASE_MESSAGE_TYPE:
                next_iter = payload.get("next_iter")

                if isinstance(next_iter, int):
                    self._synchronization_iter = next_iter
                    self._synchronization_released = True

                continue

            if message_type != self.MESSAGE_TYPE:
                continue

            particle_data = payload.get("particle")

            if not isinstance(particle_data, dict):
                continue

            if particle_data.get("identifier") is None:
                continue

            self._replace_worst_particle(
                population=population,
                particle_data=particle_data,
                insert_arrival_particle=insert_arrival_particle,
                departure_particle=departure_particle,
            )

    def _replace_worst_particle(
        self,
        population: dict[str, float],
        particle_data: dict[str, Any],
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """Replace the worst local particle with the arriving particle."""

        if not population:
            return

        worst_particle_id = max(
            population,
            key=lambda particle_id: population[particle_id],
        )

        arriving_particle = particle_data.copy()
        arriving_particle["identifier"] = worst_particle_id

        departure_particle(worst_particle_id)
        insert_arrival_particle(arriving_particle)


class GlobalBest(MigrationDriverBase):
    """Global-best migration strategy.

    Parameters
    ----------
    initial_iter:
        First iteration at which migration or synchronization may occur.

    check_interval:
        Number of iterations between migration or synchronization points.

    synchronous:
        If ``False``, islands execute independently and global-best updates are
        propagated asynchronously.

        If ``True``, all islands stop at each synchronization point. The driver
        waits until every island reaches that point, determines the global best,
        sends it to the other islands, and then releases all islands to continue
        until the next synchronization point.
    """

    _processor_class = GlobalBestProcessor

    def __init__(
        self,
        initial_iter: int = 1,
        check_interval: int = 1,
        synchronous: bool = False,
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

        self._synchronization_ready: dict[int, set[str]] = {}
        self._synchronization_particles: dict[
            int,
            dict[str, dict[str, Any]],
        ] = {}

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
        self._synchronization_ready.clear()
        self._synchronization_particles.clear()

        self._running.set()

        self._communication_driver.start()

        self._thread = Thread(
            target=self._routing_loop,
            name="global-best-migration",
            daemon=True,
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

    def _route_message(
        self,
        source_id: str,
        message: str,
    ) -> None:
        """Process a message received from an island."""

        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        message_type = payload.get("type")

        if message_type == GlobalBestProcessor.SYNC_READY_MESSAGE_TYPE:
            self._route_sync_ready(
                source_id=source_id,
                payload=payload,
            )
            return

        if message_type != GlobalBestProcessor.MESSAGE_TYPE:
            return

        self._route_global_best(
            source_id=source_id,
            payload=payload,
        )

    def _route_global_best(
        self,
        source_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Process an asynchronous global-best update."""

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

        self._broadcast_global_best(
            particle_data=particle_data,
            source_id=source_id,
        )

    def _route_sync_ready(
        self,
        source_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Register an island at a synchronous migration point."""

        actual_iter = payload.get("actual_iter")

        if not isinstance(actual_iter, int):
            return

        particle_data = payload.get("particle")

        if not isinstance(particle_data, dict):
            particle_data = None

        ready_islands = self._synchronization_ready.setdefault(
            actual_iter,
            set(),
        )
        ready_islands.add(source_id)

        if particle_data is not None:
            fitness = particle_data.get("fitness")

            if isinstance(fitness, (int, float)):
                self._synchronization_particles.setdefault(
                    actual_iter,
                    {},
                )[source_id] = particle_data

        island_ids = set(self._communication_driver.outgoing_queues.keys())

        if not island_ids.issubset(ready_islands):
            return

        self._complete_synchronization(actual_iter)

    def _complete_synchronization(
        self,
        actual_iter: int,
    ) -> None:
        """Complete a synchronous migration barrier."""

        particles = self._synchronization_particles.pop(
            actual_iter,
            {},
        )

        best_particle = self._global_best_particle

        for particle_data in particles.values():
            fitness = particle_data.get("fitness")

            if not isinstance(fitness, (int, float)):
                continue

            fitness = float(fitness)

            if best_particle is None or fitness < float(best_particle["fitness"]):
                best_particle = particle_data

        new_global_best = best_particle is not self._global_best_particle

        if best_particle is not None:
            best_particle = best_particle.copy()

            self._global_best_particle = best_particle
            self._global_best_fitness = float(best_particle["fitness"])

        source_id = None

        if best_particle is not None:
            for island_id, particle_data in particles.items():
                if particle_data.get("identifier") == best_particle.get("identifier"):
                    source_id = island_id
                    break

            self._broadcast_global_best(
                particle_data=best_particle,
                source_id=source_id,
            )

            if new_global_best and source_id is not None:
                self.migration_log(
                    particle=best_particle,
                    origin=source_id,
                    destination="all",
                    iteration=actual_iter,
                )

        next_iter = actual_iter + self._get_check_interval()

        release_message = json.dumps(
            {
                "type": GlobalBestProcessor.SYNC_RELEASE_MESSAGE_TYPE,
                "next_iter": next_iter,
            }
        )

        for outgoing_queue in self._communication_driver.outgoing_queues.values():
            outgoing_queue.put(release_message)

        self._synchronization_ready.pop(actual_iter, None)

    def _broadcast_global_best(
        self,
        particle_data: dict[str, Any],
        source_id: str | None,
    ) -> None:
        """Broadcast a global-best particle to all other islands."""

        message = json.dumps(
            {
                "type": GlobalBestProcessor.MESSAGE_TYPE,
                "particle": particle_data,
            }
        )

        for (
            island_id,
            outgoing_queue,
        ) in self._communication_driver.outgoing_queues.items():
            if source_id is not None and island_id == source_id:
                continue

            outgoing_queue.put(message)

    def _get_check_interval(self) -> int:
        """Return the configured synchronization interval."""

        return self._migration_processor_init_kargs["check_interval"]

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

                self._route_message(
                    source_id=island_id,
                    message=message,
                )
