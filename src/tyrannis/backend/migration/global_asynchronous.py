from __future__ import annotations

import json
from collections.abc import Callable
from queue import Empty
from threading import Event, Thread
from time import sleep
from typing import Any

from ...core.backend_communication import CommunicationProcessorBase
from ...core.backend_migration import (
    MigrationDriverBase,
    MigrationProcessorBase,
)
from ...core.signals import LocalEvent


class GlobalAsynchronousProcessor(MigrationProcessorBase):
    """Migration processor for global asynchronous migration.

    Islands execute independently. At each ``check_interval`` point, starting
    at ``initial_iter``, the island publishes its local best when that best is
    newer than the best it has previously published. Incoming global-best
    updates are consumed without blocking the optimization loop.

    When a global-best particle arrives, it replaces the worst particle in the
    receiving island. The identifier of the local particle being replaced is
    preserved, while the state of the arriving particle is copied into that
    identifier.

    The communication protocol is intentionally transport-independent. The
    communication backend exposes only strings through its queues; this class
    uses JSON messages with the following application-level format::

        {
            "type": "global_best_update",
            "particle": { ... serialized particle ... }
        }

    The driver receives those messages, keeps the best global fitness seen so
    far, and broadcasts an accepted update to all other islands.
    """

    MESSAGE_TYPE = "global_best_update"

    def __init__(
        self,
        initial_iter: int,
        check_interval: int,
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
        self._communication_processor = communication_processor

        self._last_published_fitness: float | None = None
        self._migration_signal: LocalEvent | None = None

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

    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        """Start the processor communication backend."""

        self._communication_processor.start(
            stop_signal=stop_signal,
        )

    def stop(self) -> None:
        """Stop the migration processor communication backend."""

        self._communication_processor.stop()

    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, float | None],
        local_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
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

        if local_best is not None:
            self._publish_if_new_best(local_best)

        while actual_iter >= self._synchronization_iter:
            self._synchronization_iter += self._check_interval

    def _publish_if_new_best(
        self,
        local_best: str,
    ) -> None:
        """Publish the local best when it improves the last published best."""

        try:
            particle_data = json.loads(local_best)
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
            }
        )

        self._communication_processor.outgoing_queue.put(message)
        self._last_published_fitness = fitness

    def _consume_messages(
        self,
        population: dict[str, float | None],
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

            if payload.get("type") != self.MESSAGE_TYPE:
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
        population: dict[str, float | None],
        particle_data: dict[str, Any],
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        if not population:
            return

        worst_particle_id = max(
            population,
            key=lambda particle_id: self._get_particle_fitness(
                population,
                particle_id,
            ),
        )

        arriving_particle = particle_data.copy()
        arriving_particle["identifier"] = worst_particle_id

        departure_particle(worst_particle_id)
        insert_arrival_particle(arriving_particle)


class GlobalAsynchronous(MigrationDriverBase):
    """Global asynchronous migration strategy.

    ``initial_iter`` defines the first iteration at which migration can occur.
    ``check_interval`` defines the interval between migration/synchronization
    checks.

    Therefore, the synchronization points are::

        initial_iter,
        initial_iter + check_interval,
        initial_iter + 2 * check_interval,
        ...

    No global barrier is introduced: an island never waits for the other
    islands to reach a synchronization point.
    """

    _processor_class = GlobalAsynchronousProcessor

    def __init__(
        self,
        initial_iter: int = 1,
        check_interval: int = 1,
    ) -> None:
        if initial_iter < 1:
            raise ValueError("initial_iter must be greater than or equal to 1.")

        if check_interval <= 0:
            raise ValueError("check_interval must be greater than 0.")

        self._migration_processor_init_kargs = {
            "initial_iter": initial_iter,
            "check_interval": check_interval,
        }

        self._global_best_fitness: float | None = None
        self._running = Event()
        self._thread = None

    def initialize_context(
        self,
        communication_driver,
        communication_processor_class,
        communication_processor_kargs: dict[str, Any],
    ) -> None:
        super().initialize_context(
            communication_driver=communication_driver,
            communication_processor_class=communication_processor_class,
            communication_processor_kargs=communication_processor_kargs,
        )

    def start(self) -> None:
        """Start the communication driver and application-level router."""

        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("Migration is already running.")

        self._global_best_fitness = None
        self._running.set()

        self._communication_driver.start()

        self._thread = Thread(
            target=self._routing_loop,
            name="global-asynchronous-migration",
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

        if payload.get("type") != GlobalAsynchronousProcessor.MESSAGE_TYPE:
            return

        particle_data = payload.get("particle")

        if not isinstance(particle_data, dict):
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

        message = json.dumps(
            {
                "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
                "particle": particle_data,
            }
        )

        for (
            island_id,
            outgoing_queue,
        ) in self._communication_driver.outgoing_queues.items():
            if island_id == source_id:
                continue

            outgoing_queue.put(message)

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
