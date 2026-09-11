from __future__ import annotations

import json
from collections.abc import Callable
from queue import Empty
from threading import Event, Thread
from time import sleep
from typing import Any

import numpy as np
import scipy as sp

from ..core.backend_communication import CommunicationProcessorBase
from ..core.backend_migration import MigrationDriverBase, MigrationProcessorBase
from ..core.results import HistoryConfig
from ..core.signals import LocalEvent


class IslandMigrationProcessor(MigrationProcessorBase):
    """Processor-side implementation of island migration."""

    STATE = "state"
    MIGRATION_REQUEST = "migration_request"
    PARTICLE = "particle"
    RING_RELEASE = "ring_release"

    def __init__(
        self,
        initial_iter: int,
        selection: str,
        migration_size: int,
        min_interval: int,
        communication_processor: CommunicationProcessorBase,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self._initial_iter = initial_iter
        self._selection = selection
        self._migration_size = migration_size
        self._min_interval = min_interval
        self._communication_processor = communication_processor
        self._migration_signal: LocalEvent | None = None
        self._algorithm = None
        self._ring_iter = initial_iter
        self._ring_released = False
        self._rng = np.random.default_rng()

    def start(self) -> None:
        self._communication_processor.start()

    def stop(self) -> None:
        self._communication_processor.stop()

    def initialize_loop_context(self, migration_signal: LocalEvent) -> None:
        self._migration_signal = migration_signal
        self._communication_processor.set_message_signal(migration_signal)

    def finalize_loop_context(self) -> None:
        self._communication_processor.set_message_signal(None)
        self._migration_signal = None

    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, float],
        iter_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        if self._migration_signal is None:
            raise RuntimeError("Migration loop context has not been initialized.")

        self._consume_messages(
            insert_arrival_particle=insert_arrival_particle,
            departure_particle=departure_particle,
        )

        if actual_iter < self._initial_iter:
            self._migration_signal.clear()
            return

        self._send_state(
            actual_iter=actual_iter,
            population=population,
        )

        if actual_iter != self._ring_iter:
            self._migration_signal.clear()
            return

        self._ring_released = False

        while not self._ring_released:
            self._consume_messages(
                insert_arrival_particle=insert_arrival_particle,
                departure_particle=departure_particle,
            )

            if not self._ring_released:
                sleep(0.001)

        self._ring_iter += self._min_interval
        self._ring_released = False
        self._migration_signal.clear()

    def _send_state(self, actual_iter: int, population: dict[str, float]) -> None:
        self._communication_processor.outgoing_queue.put(
            json.dumps(
                {
                    "type": self.STATE,
                    "actual_iter": actual_iter,
                    "population": population,
                }
            )
        )

    def _consume_messages(
        self,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
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

            if message_type == self.MIGRATION_REQUEST:
                self._send_selected_particle(
                    request_id=payload.get("request_id"),
                    actual_iter=payload.get("actual_iter"),
                    departure_particle=departure_particle,
                )

            elif message_type == self.PARTICLE:
                particle = payload.get("particle")

                if isinstance(particle, dict):
                    insert_arrival_particle(particle)

            elif message_type == self.RING_RELEASE:
                self._ring_released = True

    def _send_selected_particle(
        self,
        request_id: Any,
        actual_iter: Any,
        departure_particle: Callable[[str], None],
    ) -> None:
        if self._algorithm is None:
            raise RuntimeError("Migration processor algorithm is not initialized.")

        if not isinstance(request_id, str):
            return

        if not isinstance(actual_iter, int):
            return

        particle_ids = list(self._algorithm.population)

        if not particle_ids:
            particle = None

        else:
            particle_id = self._select_particle(
                particle_ids=particle_ids,
            )

            particle = self._algorithm.population[particle_id]()
            departure_particle(particle_id)

        self._communication_processor.outgoing_queue.put(
            json.dumps(
                {
                    "type": self.PARTICLE,
                    "request_id": request_id,
                    "actual_iter": actual_iter,
                    "particle": particle,
                }
            )
        )

    def _select_particle(self, particle_ids: list[str]) -> str:
        if self._algorithm is None:
            raise RuntimeError("Migration processor algorithm is not initialized.")

        algorithm = self._algorithm

        if self._selection == "random":
            random_values = self._rng.random(len(particle_ids))
            return particle_ids[int(np.argmin(random_values))]

        fitness = np.asarray(
            [
                (
                    np.inf
                    if algorithm.population[particle_id].fitness is None
                    else algorithm.population[particle_id].fitness
                )
                for particle_id in particle_ids
            ],
            dtype=float,
        )

        if self._selection == "best":
            target = np.min(fitness)
        else:
            target = np.max(fitness)

        candidates = np.flatnonzero(fitness == target)
        return particle_ids[int(self._rng.choice(candidates))]


class IslandMigration(MigrationDriverBase):
    """Configurable island migration strategy."""

    _processor_class = IslandMigrationProcessor

    def __init__(
        self,
        initial_iter: int = 1,
        min_interval: int = 1,
        min_population: int = 0,
        migration_size: int = 1,
        trigger: str = "n_iter",
        movement_strategy: str = "random",
        selection: str = "random",
        balance_population: bool = True,
        migration_probability: float = 0.5,
    ) -> None:
        movement_strategies = {
            "random",
            "random_neighbor",
            "greedy",
            "greedy_neighbor",
            "greedy_random",
            "star",
            "ring",
            "ring_sequential",
        }

        selections = {
            "random",
            "best",
            "worst",
        }

        triggers = {
            "n_iter",
            "random",
        }

        if initial_iter < 1:
            raise ValueError("initial_iter must be greater than 0.")

        if movement_strategy not in movement_strategies:
            raise ValueError(f"Invalid movement_strategy: {movement_strategy!r}.")

        if selection not in selections:
            raise ValueError(f"Invalid selection: {selection!r}.")

        if trigger not in triggers:
            raise ValueError(f"Invalid trigger: {trigger!r}.")

        if movement_strategy == "ring" and trigger != "n_iter":
            raise ValueError("movement_strategy='ring' requires trigger='n_iter'.")

        if migration_size <= 0:
            raise ValueError("migration_size must be greater than 0.")

        if min_population < 0:
            raise ValueError("min_population must be greater than or equal to 0.")

        if min_interval <= 0:
            raise ValueError("min_interval must be greater than 0.")

        if not 0 <= migration_probability <= 1:
            raise ValueError("migration_probability must be between 0 and 1.")

        self._initial_iter = initial_iter
        self._movement_strategy = movement_strategy
        self._trigger = trigger
        self._migration_size = migration_size
        self._min_population = min_population
        self._min_interval = min_interval
        self._balance_population = balance_population
        self._migration_probability = migration_probability

        self._migration_processor_init_kargs = {
            "initial_iter": initial_iter,
            "selection": selection,
            "migration_size": migration_size,
            "min_interval": min_interval,
        }

        self._island_ids: list[str] = []
        self._states: dict[str, dict[str, Any]] = {}
        self._next_migration_iter = initial_iter
        self._last_trigger_check_iter: int | None = None
        self._pending_requests: dict[str, tuple[str, str, int]] = {}
        self._ring_pending: dict[int, int] = {}
        self._ring_received: dict[int, int] = {}
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

        self._island_ids = sorted(
            communication_driver.incoming_queues,
        )

    def start(self) -> None:
        self._states.clear()
        self._pending_requests.clear()
        self._ring_pending.clear()
        self._ring_received.clear()

        self._next_migration_iter = self._initial_iter
        self._last_trigger_check_iter = None

        self._running.set()

        self._communication_driver.start()

        self._thread = Thread(
            target=self._routing_loop,
            name="island-migration",
            daemon=True,
        )

        self._thread.start()

    def stop(self) -> None:
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
                    island_id=island_id,
                    message=message,
                )

    def _route_message(self, island_id: str, message: str) -> None:
        try:
            payload = json.loads(message)
        except (TypeError, json.JSONDecodeError):
            return

        if not isinstance(payload, dict):
            return

        message_type = payload.get("type")

        if message_type == IslandMigrationProcessor.STATE:
            actual_iter = payload.get("actual_iter")
            population = payload.get("population")

            if not isinstance(actual_iter, int):
                return

            if not isinstance(population, dict):
                return

            self._states[island_id] = {
                "actual_iter": actual_iter,
                "population": population,
            }

            if self._movement_strategy == "ring":
                self._maybe_activate_ring(
                    actual_iter=actual_iter,
                )

            else:
                self._maybe_activate(
                    actual_iter=actual_iter,
                )

            return

        if message_type == IslandMigrationProcessor.PARTICLE:
            self._receive_particle(
                payload=payload,
            )

    def _maybe_activate(self, actual_iter: int) -> None:
        if actual_iter < self._next_migration_iter:
            return

        if self._trigger == "random":
            if self._last_trigger_check_iter == actual_iter:
                return

            self._last_trigger_check_iter = actual_iter

            if self._rng.random() >= self._migration_probability:
                return

        self._activate_migration(
            actual_iter=actual_iter,
        )

        self._next_migration_iter = actual_iter + self._min_interval

    def _activate_migration(self, actual_iter: int) -> None:
        for _ in range(self._migration_size):
            pair = self._choose_migration_pair()

            if pair is None:
                continue

            donor, receiver = pair

            request_id = f"migration:{actual_iter}:{len(self._pending_requests)}"

            self._pending_requests[request_id] = (
                donor,
                receiver,
                actual_iter,
            )

            self._request_particle(
                donor=donor,
                request_id=request_id,
                actual_iter=actual_iter,
            )

    def _request_particle(self, donor: str, request_id: str, actual_iter: int) -> None:
        self._communication_driver.outgoing_queues[donor].put(
            json.dumps(
                {
                    "type": (IslandMigrationProcessor.MIGRATION_REQUEST),
                    "request_id": request_id,
                    "actual_iter": actual_iter,
                }
            )
        )

    def _choose_migration_pair(self) -> tuple[str, str] | None:
        donors = self._donor_candidates()

        if not donors:
            return None

        donor = self._choose_donor(
            candidates=donors,
        )

        receivers = [island_id for island_id in self._island_ids if island_id != donor]

        if not receivers:
            return None

        receiver = self._choose_receiver(
            donor=donor,
            candidates=receivers,
        )

        return donor, receiver

    def _donor_candidates(self) -> list[str]:
        return [
            island_id
            for island_id in self._island_ids
            if len(
                self._states.get(
                    island_id,
                    {},
                ).get(
                    "population",
                    {},
                )
            )
            > self._min_population
        ]

    def _choose_donor(self, candidates: list[str]) -> str:
        random_values = self._rng.random(
            len(candidates),
        )

        if not self._balance_population:
            scores = random_values

        else:
            populations = np.asarray(
                [
                    len(self._states[island_id]["population"])
                    for island_id in candidates
                ],
                dtype=float,
            )

            total_population = populations.sum()

            x = populations / (total_population + 1)
            y = x / (x - 1)
            penalty = 2 * sp.special.expit(y) - 1

            scores = random_values + penalty

        minimum = np.min(scores)

        choices = np.flatnonzero(
            scores == minimum,
        )

        return candidates[int(self._rng.choice(choices))]

    def _choose_receiver(self, donor: str, candidates: list[str]) -> str:
        strategy = self._movement_strategy

        if strategy == "random":
            return self._random_island(candidates)

        if strategy == "random_neighbor":
            neighbors = [
                island_id
                for island_id in self._neighbors(donor)
                if island_id in candidates
            ]

            return self._random_island(neighbors)

        if strategy == "star":
            hub = self._island_ids[0]

            if donor != hub:
                return hub

            return self._random_island(candidates)

        if strategy == "greedy":
            return self._best_fitness_island(
                candidates,
            )

        if strategy == "greedy_neighbor":
            neighbors = [
                island_id
                for island_id in self._neighbors(donor)
                if island_id in candidates
            ]

            return self._best_fitness_island(
                neighbors,
            )

        if strategy == "greedy_random":
            fitness = np.asarray(
                [self._island_fitness(island_id) for island_id in candidates],
                dtype=float,
            )

            if np.all(np.isinf(fitness)):
                return self._random_island(
                    candidates,
                )

            random_values = self._rng.random(
                len(candidates),
            )

            greedy = 2 * sp.special.expit(fitness) - 1
            scores = random_values * greedy
            minimum = np.min(scores)

            choices = np.flatnonzero(
                scores == minimum,
            )

            return candidates[int(self._rng.choice(choices))]

        if strategy == "ring_sequential":
            return self._next_ring_island(
                donor,
            )

        raise RuntimeError(f"Unsupported movement strategy: {strategy!r}")

    def _random_island(self, candidates: list[str]) -> str:
        random_values = self._rng.random(
            len(candidates),
        )

        minimum = np.min(random_values)

        choices = np.flatnonzero(
            random_values == minimum,
        )

        return candidates[int(self._rng.choice(choices))]

    def _best_fitness_island(self, candidates: list[str]) -> str:
        if not candidates:
            raise RuntimeError("No receiver candidates are available.")

        fitness = np.asarray(
            [self._island_fitness(island_id) for island_id in candidates],
            dtype=float,
        )

        if np.all(np.isinf(fitness)):
            return self._random_island(
                candidates,
            )

        minimum = np.min(fitness)

        choices = np.flatnonzero(
            fitness == minimum,
        )

        return candidates[int(self._rng.choice(choices))]

    def _island_fitness(self, island_id: str) -> float:
        population = self._states[island_id]["population"]

        if not population:
            return np.inf

        fitness = np.asarray(
            [(np.inf if value is None else value) for value in population.values()],
            dtype=float,
        )

        return float(np.min(fitness))

    def _neighbors(self, island_id: str) -> list[str]:
        index = self._island_ids.index(
            island_id,
        )

        return [
            self._island_ids[(index - 1) % len(self._island_ids)],
            self._island_ids[(index + 1) % len(self._island_ids)],
        ]

    def _next_ring_island(self, donor: str) -> str:
        return self._island_ids[
            (self._island_ids.index(donor) + 1) % len(self._island_ids)
        ]

    def _receive_particle(self, payload: dict[str, Any]) -> None:
        request_id = payload.get(
            "request_id",
        )

        particle = payload.get(
            "particle",
        )

        if not isinstance(request_id, str):
            return

        request = self._pending_requests.pop(
            request_id,
            None,
        )

        if request is None:
            return

        donor, receiver, actual_iter = request

        if not isinstance(particle, dict):
            self._finish_ring_request(
                actual_iter=actual_iter,
            )
            return

        self._states[donor]["population"].pop(
            particle["identifier"],
            None,
        )

        self._states[receiver]["population"][particle["identifier"]] = particle[
            "fitness"
        ]

        self._communication_driver.outgoing_queues[receiver].put(
            json.dumps(
                {
                    "type": (IslandMigrationProcessor.PARTICLE),
                    "particle": particle,
                    "actual_iter": actual_iter,
                }
            )
        )

        self.migration_log(
            particle=particle,
            origin=donor,
            destination=receiver,
            iteration=actual_iter,
        )

        self._finish_ring_request(
            actual_iter=actual_iter,
        )

    def _finish_ring_request(self, actual_iter: int) -> None:
        if actual_iter not in self._ring_pending:
            return

        self._ring_received[actual_iter] += 1

        self._finish_ring_if_ready(
            actual_iter=actual_iter,
        )

    def _maybe_activate_ring(self, actual_iter: int) -> None:
        if actual_iter != self._next_migration_iter:
            return

        if not all(
            self._states.get(
                island_id,
                {},
            ).get("actual_iter")
            == actual_iter
            for island_id in self._island_ids
        ):
            return

        if actual_iter in self._ring_pending:
            return

        self._activate_ring(
            actual_iter=actual_iter,
        )

        self._next_migration_iter = actual_iter + self._min_interval

    def _activate_ring(self, actual_iter: int) -> None:
        donors = self._donor_candidates()

        self._ring_pending[actual_iter] = len(donors)

        self._ring_received[actual_iter] = 0

        for donor in donors:
            receiver = self._next_ring_island(
                donor,
            )

            request_id = f"ring:{actual_iter}:{donor}"

            self._pending_requests[request_id] = (
                donor,
                receiver,
                actual_iter,
            )

            self._request_particle(
                donor=donor,
                request_id=request_id,
                actual_iter=actual_iter,
            )

        self._finish_ring_if_ready(
            actual_iter=actual_iter,
        )

    def _finish_ring_if_ready(self, actual_iter: int) -> None:
        expected = self._ring_pending.get(
            actual_iter,
        )

        if expected is None:
            return

        if self._ring_received[actual_iter] < expected:
            return

        release = json.dumps({"type": (IslandMigrationProcessor.RING_RELEASE)})

        for queue in self._communication_driver.outgoing_queues.values():
            queue.put(release)

        del self._ring_pending[actual_iter]

        del self._ring_received[actual_iter]
