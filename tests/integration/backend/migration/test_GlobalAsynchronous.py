from __future__ import annotations

import json
import socket
import time
from collections.abc import Callable
from threading import Event, Thread

import pytest

from tyrannis.backend.distributed.communication.spark_communication import (
    SparkCommunicationDriver,
    SparkCommunicationProcessor,
)
from tyrannis.backend.migration.global_asynchronous import (
    GlobalAsynchronous,
    GlobalAsynchronousProcessor,
)
from tyrannis.backend.processor.serial import Serial
from tyrannis.core.algorithm import AlgorithmBase, ParticleBase
from tyrannis.core.signals import LocalEvent

# ============================================================================
# Constants
# ============================================================================

HOST = "127.0.0.1"

FIRST_ISLAND = "island:0"
SECOND_ISLAND = "island:1"

ISLAND_IDS = [
    FIRST_ISLAND,
    SECOND_ISLAND,
]

INITIAL_ITER = 2
CHECK_INTERVAL = 2

TIMEOUT = 5.0


# ============================================================================
# Helpers
# ============================================================================


def find_free_port() -> int:
    """Return an available local TCP port."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((HOST, 0))
        return int(sock.getsockname()[1])


def wait_for(
    condition: Callable[[], bool],
    timeout: float = TIMEOUT,
) -> None:
    """Wait until a condition becomes true."""

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        if condition():
            return

        time.sleep(0.01)

    raise AssertionError(
        "Condition was not satisfied before timeout.",
    )


def serialize_particle(
    identifier: str,
    variables: dict[str, float],
    fitness: float,
) -> str:
    """Serialize a particle using the Tyrannis particle protocol."""

    return json.dumps(
        {
            "identifier": identifier,
            "variables": variables,
            "fitness": fitness,
        }
    )


def serialize_update(
    identifier: str,
    variables: dict[str, float],
    fitness: float,
) -> str:
    """Serialize a GlobalAsynchronous update message."""

    return json.dumps(
        {
            "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
            "particle": {
                "identifier": identifier,
                "variables": variables,
                "fitness": fitness,
            },
        }
    )


def create_migration_processor(
    communication_processor: SparkCommunicationProcessor,
    initial_iter: int = INITIAL_ITER,
    check_interval: int = CHECK_INTERVAL,
) -> GlobalAsynchronousProcessor:
    """Create a migration processor with its loop context initialized."""

    processor = GlobalAsynchronousProcessor(
        initial_iter=initial_iter,
        check_interval=check_interval,
        communication_processor=communication_processor,
    )

    processor.initialize_loop_context(
        migration_signal=LocalEvent(),
    )

    return processor


# ============================================================================
# Deterministic algorithm
# ============================================================================


class MigrationTestParticle(ParticleBase):
    """Particle used by the integration tests."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: float | None = None,
    ) -> None:
        super().__init__(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
        )


class MigrationTestAlgorithm(AlgorithmBase):
    """
    Deterministic algorithm used to exercise the migration infrastructure.

    island:0 starts with:

        island:0|particle:0 -> x=0, fitness=0
        island:0|particle:1 -> x=1, fitness=1

    island:1 starts with:

        island:1|particle:0 -> x=100, fitness=10000
        island:1|particle:1 -> x=101, fitness=10201

    No optimization step changes particle variables. Therefore any change in
    island:1 is caused exclusively by migration.
    """

    def __init__(self) -> None:
        self._iteration_started = Event()
        self._release_iteration = Event()

    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: float | None = None,
        *args: object,
        **kwargs: object,
    ) -> None:
        if variables is None:
            if identifier.startswith(f"{FIRST_ISLAND}|"):
                if identifier.endswith("particle:0"):
                    variables = {"x": 0.0}
                else:
                    variables = {"x": 1.0}
            else:
                if identifier.endswith("particle:0"):
                    variables = {"x": 100.0}
                else:
                    variables = {"x": 101.0}

        self._population[identifier] = MigrationTestParticle(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
        )

    def delete_particle(
        self,
        identifier: str | None,
    ) -> None:
        if identifier is None:
            raise ValueError(
                "Particle identifier cannot be None.",
            )

        del self._population[identifier]

    def pre_iteration(
        self,
        actual_iter: int,
    ) -> None:
        self._iteration_started.set()

        if actual_iter == 1:
            self._release_iteration.wait(
                timeout=TIMEOUT,
            )

    def initialize_particle(
        self,
        identifier: str,
    ) -> ParticleBase:
        particle = self._population[identifier]

        if particle.fitness is None:
            particle.update(
                variables=particle.variables,
                fitness_function=self._fitness_function,
            )

            particle.consolidate(
                consolidate_new=True,
            )

        return particle

    def update_particle(
        self,
        identifier: str,
    ) -> ParticleBase:
        particle = self._population[identifier]

        if particle.fitness is None:
            raise RuntimeError(
                f"Particle '{identifier}' has no fitness.",
            )

        return particle

    def post_iteration(
        self,
        actual_iter: int,
    ) -> None:
        if not self._population:
            self._local_best = None
            return

        self._local_best = min(
            self._population.values(),
            key=lambda particle: (
                particle.fitness if particle.fitness is not None else float("inf")
            ),
        )

    def create_random_cache(
        self,
        particle_ids: list[str],
    ) -> None:
        for particle_id in particle_ids:
            self._population[particle_id].random_cache = []


def sphere(
    variables: dict[str, float],
) -> float:
    return sum(value * value for value in variables.values())


# ============================================================================
# Communication fixture
# ============================================================================


@pytest.fixture
def communication_environment():
    """
    Create a complete real TCP communication environment.

    The following production components are used:

        GlobalAsynchronous
        SparkCommunicationDriver
        SparkCommunicationProcessor
        GlobalAsynchronousProcessor

    No communication mock is used.
    """

    port = find_free_port()

    driver_stop_signal = LocalEvent()

    driver = SparkCommunicationDriver(
        island_ids=ISLAND_IDS,
        port=port,
        stop_signal=driver_stop_signal,
    )

    migration = GlobalAsynchronous(
        initial_iter=INITIAL_ITER,
        check_interval=CHECK_INTERVAL,
    )

    migration.initialize_context(
        communication_driver=driver,
        communication_processor_class=SparkCommunicationProcessor,
        communication_processor_kargs={
            "driver_ip": HOST,
            "port": port,
        },
    )

    driver.start()

    processors = {
        island_id: SparkCommunicationProcessor(
            driver_ip=HOST,
            port=port,
            identification=island_id,
        )
        for island_id in ISLAND_IDS
    }

    stop_signals = {island_id: LocalEvent() for island_id in ISLAND_IDS}

    try:
        for island_id in ISLAND_IDS:
            processors[island_id].start(
                stop_signal=stop_signals[island_id],
            )

        wait_for(
            lambda: set(driver._connections) == set(ISLAND_IDS),
        )

        yield (
            migration,
            driver,
            processors,
            stop_signals,
            driver_stop_signal,
        )

    finally:
        for processor in processors.values():
            processor.stop()

        driver.stop()


# ============================================================================
# GlobalAsynchronous configuration
# ============================================================================


class TestGlobalAsynchronousConfiguration:
    """Integration tests for GlobalAsynchronous configuration."""

    def test_initial_iter_is_first_synchronization_point(self) -> None:
        migration = GlobalAsynchronous(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
        )

        assert migration._migration_processor_init_kargs == {
            "initial_iter": INITIAL_ITER,
            "check_interval": CHECK_INTERVAL,
        }

    def test_check_interval_advances_synchronization_point(
        self,
        communication_environment,
    ) -> None:
        (
            _migration,
            _,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[FIRST_ISLAND],
        )

        assert processor._initial_iter == INITIAL_ITER
        assert processor._check_interval == CHECK_INTERVAL
        assert processor._synchronization_iter == INITIAL_ITER

        processor.migration_control(
            actual_iter=INITIAL_ITER,
            population={},
            local_best=None,
            insert_arrival_particle=lambda _: None,
            departure_particle=lambda _: None,
        )

        assert processor._synchronization_iter == (INITIAL_ITER + CHECK_INTERVAL)

        processor.migration_control(
            actual_iter=INITIAL_ITER + CHECK_INTERVAL,
            population={},
            local_best=None,
            insert_arrival_particle=lambda _: None,
            departure_particle=lambda _: None,
        )

        assert processor._synchronization_iter == (INITIAL_ITER + 2 * CHECK_INTERVAL)

    def test_iteration_before_initial_iter_does_not_publish(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        communication = SparkCommunicationProcessor(
            driver_ip=HOST,
            port=driver._port,
            identification="test-island",
        )

        communication._messages = __import__("queue").Queue()  # type: ignore
        communication._outgoing_queue = __import__("queue").Queue()  # type: ignore

        processor = GlobalAsynchronousProcessor(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
            communication_processor=communication,
        )

        processor.initialize_loop_context(
            migration_signal=LocalEvent(),
        )

        local_best = serialize_particle(
            identifier="particle:0",
            variables={"x": 0.0},
            fitness=0.0,
        )

        processor.migration_control(
            actual_iter=INITIAL_ITER - 1,
            population={
                "particle:0": 0.0,
            },
            local_best=local_best,
            insert_arrival_particle=lambda _: None,
            departure_particle=lambda _: None,
        )

        assert communication.outgoing_queue.empty()
        assert processor._synchronization_iter == INITIAL_ITER

    def test_iteration_at_initial_iter_publishes(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[FIRST_ISLAND],
        )

        local_best = serialize_particle(
            identifier=f"{FIRST_ISLAND}|particle:0",
            variables={"x": 0.0},
            fitness=0.0,
        )

        processor.migration_control(
            actual_iter=INITIAL_ITER,
            population={
                f"{FIRST_ISLAND}|particle:0": 0.0,
            },
            local_best=local_best,
            insert_arrival_particle=lambda _: None,
            departure_particle=lambda _: None,
        )

        wait_for(
            lambda: not driver.incoming_queues[FIRST_ISLAND].empty(),
        )

        message = driver.incoming_queues[FIRST_ISLAND].get()

        payload = json.loads(message)

        assert payload["type"] == GlobalAsynchronousProcessor.MESSAGE_TYPE

    def test_initialize_loop_context_sets_migration_signal(
        self,
        communication_environment,
    ) -> None:
        (
            _migration,
            _driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = GlobalAsynchronousProcessor(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
            communication_processor=processors[FIRST_ISLAND],
        )

        migration_signal = LocalEvent()

        processor.initialize_loop_context(
            migration_signal=migration_signal,
        )

        assert processor._migration_signal is migration_signal

    def test_initialize_loop_context_sets_message_signal(
        self,
        communication_environment,
    ) -> None:
        (
            _migration,
            _driver,
            processors,
            _,
            _,
        ) = communication_environment

        communication_processor = processors[FIRST_ISLAND]

        processor = GlobalAsynchronousProcessor(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
            communication_processor=communication_processor,
        )

        migration_signal = LocalEvent()

        processor.initialize_loop_context(
            migration_signal=migration_signal,
        )

        assert communication_processor._message_signal is migration_signal

    def test_finalize_loop_context_clears_message_signal(
        self,
        communication_environment,
    ) -> None:
        (
            _migration,
            _driver,
            processors,
            _,
            _,
        ) = communication_environment

        communication_processor = processors[FIRST_ISLAND]

        processor = GlobalAsynchronousProcessor(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
            communication_processor=communication_processor,
        )

        processor.initialize_loop_context(
            migration_signal=LocalEvent(),
        )

        processor.finalize_loop_context()

        assert processor._migration_signal is None
        assert communication_processor._message_signal is None

    def test_migration_control_requires_loop_context(
        self,
        communication_environment,
    ) -> None:
        (
            _migration,
            _driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = GlobalAsynchronousProcessor(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
            communication_processor=processors[FIRST_ISLAND],
        )

        with pytest.raises(
            RuntimeError,
            match="Migration loop context has not been initialized.",
        ):
            processor.migration_control(
                actual_iter=INITIAL_ITER,
                population={},
                local_best=None,
                insert_arrival_particle=lambda _: None,
                departure_particle=lambda _: None,
            )

    def test_large_iteration_jump_advances_to_next_synchronization_point(
        self,
        communication_environment,
    ) -> None:
        (
            _migration,
            _driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = GlobalAsynchronousProcessor(
            initial_iter=2,
            check_interval=3,
            communication_processor=processors[FIRST_ISLAND],
        )

        processor.initialize_loop_context(
            migration_signal=LocalEvent(),
        )

        processor.migration_control(
            actual_iter=10,
            population={},
            local_best=None,
            insert_arrival_particle=lambda _: None,
            departure_particle=lambda _: None,
        )

        assert processor._synchronization_iter == 11


# ============================================================================
# Driver server lifecycle
# ============================================================================


class TestGlobalAsynchronousDriverServer:
    """Integration tests for the real TCP server."""

    def test_driver_creates_server_on_start(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        assert driver._server_socket is not None
        assert driver._selector is not None
        assert driver._thread is not None
        assert driver._thread.is_alive()
        assert driver._running.is_set()

    def test_processors_connect_to_driver(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        assert set(driver._connections) == set(ISLAND_IDS)

        for island_id in ISLAND_IDS:
            processor = processors[island_id]

            assert processor._socket is not None
            assert processor._thread is not None
            assert processor._thread.is_alive()

    def test_driver_registers_correct_processor_identity(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        registered_islands = set(driver._connection_identifications.values())

        assert registered_islands == set(ISLAND_IDS)

    def test_driver_stop_closes_server_and_connections(
        self,
    ) -> None:
        port = find_free_port()

        stop_signal = LocalEvent()

        driver = SparkCommunicationDriver(
            island_ids=ISLAND_IDS,
            port=port,
            stop_signal=stop_signal,
        )

        driver.start()

        assert driver._server_socket is not None
        assert driver._thread is not None

        driver.stop()

        assert driver._server_socket is None
        assert driver._selector is None
        assert driver._thread is None
        assert driver._connections == {}
        assert driver._connection_identifications == {}
        assert driver._receive_buffers == {}


# ============================================================================
# TCP protocol
# ============================================================================


class TestGlobalAsynchronousProtocol:
    """Integration tests for STX/ETX/STOP communication."""

    def test_processor_message_reaches_correct_driver_queue(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        message = "message-from-island-0"

        processors[FIRST_ISLAND].outgoing_queue.put(
            message,
        )

        wait_for(
            lambda: not driver.incoming_queues[FIRST_ISLAND].empty(),
        )

        assert driver.incoming_queues[FIRST_ISLAND].get() == message

        assert driver.incoming_queues[SECOND_ISLAND].empty()

    def test_driver_message_reaches_correct_processor(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        message = "message-from-driver"

        driver.outgoing_queues[SECOND_ISLAND].put(
            message,
        )

        wait_for(
            lambda: not processors[SECOND_ISLAND].messages.empty(),
        )

        assert processors[SECOND_ISLAND].messages.get() == message

        assert processors[FIRST_ISLAND].messages.empty()

    def test_stx_etx_framing_is_removed_before_application_queue(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = processors[FIRST_ISLAND]

        assert processor._socket is not None

        message = "framed-message"

        processor._socket.sendall(
            (processor.STX + message + processor.ETX).encode("utf-8"),
        )

        wait_for(
            lambda: not driver.incoming_queues[FIRST_ISLAND].empty(),
        )

        assert driver.incoming_queues[FIRST_ISLAND].get() == message

    def test_multiple_messages_preserve_order(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        messages = [f"message-{index}" for index in range(10)]

        for message in messages:
            processors[FIRST_ISLAND].outgoing_queue.put(
                message,
            )

        wait_for(
            lambda: driver.incoming_queues[FIRST_ISLAND].qsize() == len(messages),
        )

        received = [driver.incoming_queues[FIRST_ISLAND].get() for _ in messages]

        assert received == messages

    def test_stop_signal_is_propagated_to_processors(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            _,
            stop_signals,
            _,
        ) = communication_environment

        driver._send_stop_signal()

        wait_for(
            lambda: all(signal.is_set() for signal in stop_signals.values()),
        )

        assert stop_signals[FIRST_ISLAND].is_set()
        assert stop_signals[SECOND_ISLAND].is_set()

    def test_driver_stop_signal_is_observed_by_communication_loop(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            _,
            _,
            driver_stop_signal,
        ) = communication_environment

        assert driver._thread is not None
        assert driver._thread.is_alive()

        driver_stop_signal.set()

        wait_for(
            lambda: driver._thread is None or not driver._thread.is_alive(),
        )


# ============================================================================
# GlobalAsynchronous routing
# ============================================================================


class TestGlobalAsynchronousRouting:
    """Integration tests for driver-side global-best determination."""

    def test_first_global_best_is_accepted(
        self,
        communication_environment,
    ) -> None:
        (
            migration,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        message = serialize_update(
            identifier=f"{FIRST_ISLAND}|particle:0",
            variables={"x": 0.0},
            fitness=10.0,
        )

        driver.incoming_queues[FIRST_ISLAND].put(
            message,
        )

        migration._process_incoming()

        assert migration._global_best_fitness == 10.0

        assert driver.outgoing_queues[FIRST_ISLAND].empty()

        assert driver.outgoing_queues[SECOND_ISLAND].qsize() == 1

    def test_better_global_best_replaces_previous(
        self,
        communication_environment,
    ) -> None:
        (
            migration,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        first_message = serialize_update(
            identifier=f"{FIRST_ISLAND}|particle:0",
            variables={"x": 10.0},
            fitness=10.0,
        )

        better_message = serialize_update(
            identifier=f"{SECOND_ISLAND}|particle:0",
            variables={"x": 5.0},
            fitness=5.0,
        )

        driver.incoming_queues[FIRST_ISLAND].put(
            first_message,
        )

        migration._process_incoming()

        assert migration._global_best_fitness == 10.0

        driver.outgoing_queues[SECOND_ISLAND].get()

        driver.incoming_queues[SECOND_ISLAND].put(
            better_message,
        )

        migration._process_incoming()

        assert migration._global_best_fitness == 5.0

        assert driver.outgoing_queues[FIRST_ISLAND].qsize() == 1

        assert driver.outgoing_queues[SECOND_ISLAND].empty()

        payload = json.loads(driver.outgoing_queues[FIRST_ISLAND].get())

        assert payload["particle"]["fitness"] == 5.0

    def test_equal_global_best_is_ignored(
        self,
        communication_environment,
    ) -> None:
        (
            migration,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        first_message = serialize_update(
            identifier=f"{FIRST_ISLAND}|particle:0",
            variables={"x": 0.0},
            fitness=5.0,
        )

        equal_message = serialize_update(
            identifier=f"{SECOND_ISLAND}|particle:0",
            variables={"x": 1.0},
            fitness=5.0,
        )

        driver.incoming_queues[FIRST_ISLAND].put(
            first_message,
        )

        migration._process_incoming()

        driver.outgoing_queues[SECOND_ISLAND].get()

        driver.incoming_queues[SECOND_ISLAND].put(
            equal_message,
        )

        migration._process_incoming()

        assert migration._global_best_fitness == 5.0

        assert driver.outgoing_queues[FIRST_ISLAND].empty()
        assert driver.outgoing_queues[SECOND_ISLAND].empty()

    def test_worse_global_best_is_ignored(
        self,
        communication_environment,
    ) -> None:
        (
            migration,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        best_message = serialize_update(
            identifier=f"{FIRST_ISLAND}|particle:0",
            variables={"x": 0.0},
            fitness=5.0,
        )

        worse_message = serialize_update(
            identifier=f"{SECOND_ISLAND}|particle:0",
            variables={"x": 10.0},
            fitness=10.0,
        )

        driver.incoming_queues[FIRST_ISLAND].put(
            best_message,
        )

        migration._process_incoming()

        driver.outgoing_queues[SECOND_ISLAND].get()

        driver.incoming_queues[SECOND_ISLAND].put(
            worse_message,
        )

        migration._process_incoming()

        assert migration._global_best_fitness == 5.0

        assert driver.outgoing_queues[FIRST_ISLAND].empty()
        assert driver.outgoing_queues[SECOND_ISLAND].empty()

    def test_global_best_is_not_sent_back_to_source(
        self,
        communication_environment,
    ) -> None:
        (
            migration,
            driver,
            _,
            _,
            _,
        ) = communication_environment

        message = serialize_update(
            identifier=f"{FIRST_ISLAND}|particle:0",
            variables={"x": 0.0},
            fitness=0.0,
        )

        driver.incoming_queues[FIRST_ISLAND].put(
            message,
        )

        migration._process_incoming()

        assert driver.outgoing_queues[FIRST_ISLAND].empty()

        assert driver.outgoing_queues[SECOND_ISLAND].qsize() == 1


# ============================================================================
# GlobalAsynchronous processor reception
# ============================================================================


class TestGlobalAsynchronousReception:
    """Integration tests for migration reception and replacement."""

    def test_received_particle_replaces_worst_particle(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[SECOND_ISLAND],
        )

        incoming_particle = serialize_update(
            identifier=f"{FIRST_ISLAND}|particle:0",
            variables={"x": 0.0},
            fitness=0.0,
        )

        driver.outgoing_queues[SECOND_ISLAND].put(
            incoming_particle,
        )

        wait_for(
            lambda: not processors[SECOND_ISLAND].messages.empty(),
        )

        population = {
            f"{SECOND_ISLAND}|particle:0": 10000.0,
            f"{SECOND_ISLAND}|particle:1": 10201.0,
        }

        inserted: list[dict] = []
        departed: list[str] = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert departed == [
            f"{SECOND_ISLAND}|particle:1",
        ]

        assert inserted == [
            {
                "identifier": f"{SECOND_ISLAND}|particle:1",
                "variables": {"x": 0.0},
                "fitness": 0.0,
            }
        ]

    def test_arriving_particle_preserves_receiver_identity(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[SECOND_ISLAND],
        )

        driver.outgoing_queues[SECOND_ISLAND].put(
            serialize_update(
                identifier=f"{FIRST_ISLAND}|particle:0",
                variables={
                    "x": 123.0,
                    "y": 456.0,
                },
                fitness=3.5,
            )
        )

        wait_for(
            lambda: not processors[SECOND_ISLAND].messages.empty(),
        )

        population = {
            f"{SECOND_ISLAND}|particle:0": 10.0,
            f"{SECOND_ISLAND}|particle:1": 20.0,
            f"{SECOND_ISLAND}|particle:2": 50.0,
        }

        inserted: list[dict] = []
        departed: list[str] = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert departed == [
            f"{SECOND_ISLAND}|particle:2",
        ]

        assert inserted[0]["identifier"] == (f"{SECOND_ISLAND}|particle:2")

        assert inserted[0]["identifier"] != (f"{FIRST_ISLAND}|particle:0")

        assert inserted[0]["variables"] == {
            "x": 123.0,
            "y": 456.0,
        }

        assert inserted[0]["fitness"] == 3.5

    def test_best_particle_is_not_replaced_when_not_worst(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[SECOND_ISLAND],
        )

        driver.outgoing_queues[SECOND_ISLAND].put(
            serialize_update(
                identifier=f"{FIRST_ISLAND}|particle:0",
                variables={"x": 0.0},
                fitness=0.0,
            )
        )

        wait_for(
            lambda: not processors[SECOND_ISLAND].messages.empty(),
        )

        population = {
            f"{SECOND_ISLAND}|particle:0": 1.0,
            f"{SECOND_ISLAND}|particle:1": 100.0,
            f"{SECOND_ISLAND}|particle:2": 200.0,
        }

        inserted: list[dict] = []
        departed: list[str] = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert departed == [
            f"{SECOND_ISLAND}|particle:2",
        ]

        assert inserted[0]["identifier"] == (f"{SECOND_ISLAND}|particle:2")

    def test_empty_population_does_not_insert(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[SECOND_ISLAND],
        )

        driver.outgoing_queues[SECOND_ISLAND].put(
            serialize_update(
                identifier=f"{FIRST_ISLAND}|particle:0",
                variables={"x": 0.0},
                fitness=0.0,
            )
        )

        wait_for(
            lambda: not processors[SECOND_ISLAND].messages.empty(),
        )

        inserted: list[dict] = []
        departed: list[str] = []

        processor.migration_control(
            actual_iter=1,
            population={},
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert inserted == []
        assert departed == []

    def test_all_pending_messages_are_consumed(
        self,
        communication_environment,
    ) -> None:
        (
            _,
            driver,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[SECOND_ISLAND],
        )

        messages = [
            serialize_update(
                identifier=f"{FIRST_ISLAND}|particle:{index}",
                variables={"x": float(index)},
                fitness=float(index),
            )
            for index in range(3)
        ]

        for message in messages:
            driver.outgoing_queues[SECOND_ISLAND].put(
                message,
            )

        wait_for(
            lambda: processors[SECOND_ISLAND].messages.qsize() == len(messages),
        )

        population = {
            f"{SECOND_ISLAND}|particle:0": 10.0,
            f"{SECOND_ISLAND}|particle:1": 20.0,
            f"{SECOND_ISLAND}|particle:2": 30.0,
        }

        inserted: list[dict] = []
        departed: list[str] = []

        processor.migration_control(
            actual_iter=1,
            population=population,  # type: ignore
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert len(inserted) == 3
        assert len(departed) == 3
        assert processors[SECOND_ISLAND].messages.empty()

        assert [particle["fitness"] for particle in inserted] == [
            0.0,
            1.0,
            2.0,
        ]


# ============================================================================
# Asynchronous execution
# ============================================================================


class TestGlobalAsynchronousParallelExecution:
    """Integration tests for asynchronous execution."""

    def create_processor(
        self,
        migration: GlobalAsynchronous,
        identifier: str,
    ) -> Serial:
        algorithm = MigrationTestAlgorithm()

        algorithm.initialize_context(
            fitness_function=sphere,
            boundaries={
                "x": (-1000.0, 1000.0),
            },
        )

        processor = Serial()

        processor.initialize_context(
            algorithm=algorithm,
            n_iter=2,
            n_particles=2,
            migration_driver=migration,
            seed=42,
        )

        processor.set_identifier(identifier)

        processor._migration_processor = migration.create_processor_module(identifier)

        return processor

    def test_communication_starts_before_main_loop(
        self,
        communication_environment,
    ) -> None:
        (
            _migration,
            driver,
            processors,
            _stop_signals,
            _driver_stop_signal,
        ) = communication_environment

        communication_processor = processors[FIRST_ISLAND]

        # The communication fixture has already started the real TCP
        # communication processors and has waited until both islands are
        # registered by the driver.
        #
        # Creating another processor for FIRST_ISLAND here would attempt to
        # register a second TCP connection with the same island identifier.
        # SparkCommunicationDriver explicitly rejects that connection and sets
        # the stop signal. Therefore this test must inspect the communication
        # processor created by the fixture rather than create a duplicate one.

        assert communication_processor._thread is not None
        assert communication_processor._thread.is_alive()

        assert communication_processor._socket is not None

        assert FIRST_ISLAND in driver._connections

        assert communication_processor._running is not None
        assert communication_processor._running.is_set()

    def test_communication_runs_while_algorithm_loop_is_active(
        self,
    ) -> None:
        port = find_free_port()

        driver_stop_signal = LocalEvent()

        driver = SparkCommunicationDriver(
            island_ids=[FIRST_ISLAND],
            port=port,
            stop_signal=driver_stop_signal,
        )

        migration = GlobalAsynchronous(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
        )

        migration.initialize_context(
            communication_driver=driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": HOST,
                "port": port,
            },
        )

        processor = self.create_processor(
            migration=migration,
            identifier=FIRST_ISLAND,
        )

        try:
            # The migration processor is responsible for starting its real
            # communication processor. The driver must already be listening.
            driver.start()

            processor.initialize_execution_context()

            wait_for(
                lambda: set(driver._connections) == {FIRST_ISLAND},
            )

            algorithm = processor._algorithm

            assert isinstance(
                algorithm,
                MigrationTestAlgorithm,
            )

            migration_processor = processor._migration_processor

            assert migration_processor is not None

            communication_processor = migration_processor._communication_processor

            assert communication_processor._thread is not None  # type: ignore
            assert communication_processor._thread.is_alive()  # type: ignore

            run_thread = Thread(
                target=processor.run,
            )

            run_thread.start()

            wait_for(
                lambda: algorithm._iteration_started.is_set(),
            )

            assert run_thread.is_alive()

            # The optimization loop is intentionally blocked inside
            # pre_iteration(). Communication must continue independently.
            assert communication_processor._thread is not None  # type: ignore
            assert communication_processor._thread.is_alive()  # type: ignore

            assert FIRST_ISLAND in driver._connections

            # Release the optimization loop.
            algorithm._release_iteration.set()

            run_thread.join(
                timeout=TIMEOUT,
            )

            assert not run_thread.is_alive()

        finally:
            processor.finalize_execution_context()
            driver.stop()

    def test_stop_signal_is_shared_with_communication_processor(
        self,
        communication_environment,
    ) -> None:
        (
            migration,
            _,
            _,
            _,
            _,
        ) = communication_environment

        processor = self.create_processor(
            migration=migration,
            identifier=FIRST_ISLAND,
        )

        processor.initialize_execution_context()

        try:
            migration_processor = processor._migration_processor

            assert migration_processor is not None

            communication_processor = migration_processor._communication_processor

            assert communication_processor._stop_signal is (processor._stop_signal)  # type: ignore

        finally:
            processor.finalize_execution_context()

    def test_processor_stop_signal_stops_execution(
        self,
        communication_environment,
    ) -> None:
        (
            migration,
            _,
            _,
            _,
            _,
        ) = communication_environment

        processor = self.create_processor(
            migration=migration,
            identifier=FIRST_ISLAND,
        )

        processor.initialize_execution_context()

        try:
            processor._stop_signal.set()

            processor.run()

            assert processor._stop_signal.is_set()

        finally:
            processor.finalize_execution_context()


# ============================================================================
# End-to-end Serial migration
# ============================================================================


class TestGlobalAsynchronousSerialEndToEnd:
    """Complete migration execution using real Serial processors."""

    @staticmethod
    def create_processor(
        migration: GlobalAsynchronous,
        identifier: str,
        n_iter: int = 6,
        n_particles: int = 2,
    ) -> Serial:
        algorithm = MigrationTestAlgorithm()

        algorithm.initialize_context(
            fitness_function=sphere,
            boundaries={
                "x": (-1000.0, 1000.0),
            },
        )

        processor = Serial()

        processor.initialize_context(
            algorithm=algorithm,
            n_iter=n_iter,
            n_particles=n_particles,
            migration_driver=migration,
            seed=42,
        )

        processor.set_identifier(identifier)

        processor._migration_processor = migration.create_processor_module(identifier)

        return processor

    def test_initial_iter_is_used_by_real_serial_processors(
        self,
    ) -> None:
        port = find_free_port()

        driver_stop_signal = LocalEvent()

        driver = SparkCommunicationDriver(
            island_ids=ISLAND_IDS,
            port=port,
            stop_signal=driver_stop_signal,
        )

        migration = GlobalAsynchronous(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
        )

        migration.initialize_context(
            communication_driver=driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": HOST,
                "port": port,
            },
        )

        processors = [
            self.create_processor(
                migration,
                FIRST_ISLAND,
            ),
            self.create_processor(
                migration,
                SECOND_ISLAND,
            ),
        ]

        driver.start()

        try:
            for processor in processors:
                processor.initialize_execution_context()

            wait_for(
                lambda: set(driver._connections) == set(ISLAND_IDS),
            )

            for processor in processors:
                migration_processor = processor._migration_processor

                assert migration_processor is not None

                assert migration_processor._synchronization_iter == (INITIAL_ITER)

        finally:
            for processor in processors:
                processor.finalize_execution_context()

            driver.stop()

    def test_complete_migration_preserves_population_size(
        self,
    ) -> None:
        port = find_free_port()

        driver_stop_signal = LocalEvent()

        driver = SparkCommunicationDriver(
            island_ids=ISLAND_IDS,
            port=port,
            stop_signal=driver_stop_signal,
        )

        migration = GlobalAsynchronous(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
        )

        migration.initialize_context(
            communication_driver=driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": HOST,
                "port": port,
            },
        )

        processors = [
            self.create_processor(
                migration,
                FIRST_ISLAND,
                n_iter=20,
                n_particles=2,
            ),
            self.create_processor(
                migration,
                SECOND_ISLAND,
                n_iter=20,
                n_particles=2,
            ),
        ]

        threads = [
            Thread(
                target=processor.run,
            )
            for processor in processors
        ]

        try:
            # Start the complete migration infrastructure, including the
            # communication driver and the asynchronous routing thread.
            migration.start()

            for processor in processors:
                processor.initialize_execution_context()

            wait_for(
                lambda: set(driver._connections) == set(ISLAND_IDS),
            )

            # Start both optimization loops.
            for thread in threads:
                thread.start()

            # Wait until both processors have reached pre_iteration(1).
            #
            # MigrationTestAlgorithm intentionally blocks there until the test
            # explicitly releases the iteration. This synchronization removes
            # any dependency on thread scheduling or timeout expiration.
            for processor in processors:
                algorithm = processor._algorithm

                assert isinstance(
                    algorithm,
                    MigrationTestAlgorithm,
                )

                wait_for(
                    algorithm._iteration_started.is_set,
                )

            # Release both optimization loops explicitly.
            for processor in processors:
                algorithm = processor._algorithm

                assert isinstance(
                    algorithm,
                    MigrationTestAlgorithm,
                )

                algorithm._release_iteration.set()

            # The threads can now progress deterministically to completion.
            for thread in threads:
                thread.join(
                    timeout=TIMEOUT,
                )

            assert all(not thread.is_alive() for thread in threads)

            first_population = processors[0]._algorithm.population
            second_population = processors[1]._algorithm.population

            assert len(first_population) == 2
            assert len(second_population) == 2

        finally:
            # Always release the optimization loops in case an assertion above
            # fails while one of them is blocked in pre_iteration(1).
            for processor in processors:
                algorithm = processor._algorithm

                if isinstance(
                    algorithm,
                    MigrationTestAlgorithm,
                ):
                    algorithm._release_iteration.set()

            for thread in threads:
                if thread.is_alive():
                    thread.join(
                        timeout=TIMEOUT,
                    )

            for processor in processors:
                processor.finalize_execution_context()

    def test_complete_migration_replaces_worst_particle_and_preserves_identity(
        self,
    ) -> None:
        port = find_free_port()

        driver_stop_signal = LocalEvent()

        driver = SparkCommunicationDriver(
            island_ids=ISLAND_IDS,
            port=port,
            stop_signal=driver_stop_signal,
        )

        migration = GlobalAsynchronous(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
        )

        migration.initialize_context(
            communication_driver=driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": HOST,
                "port": port,
            },
        )

        first = self.create_processor(
            migration,
            FIRST_ISLAND,
            n_iter=20,
            n_particles=2,
        )

        second = self.create_processor(
            migration,
            SECOND_ISLAND,
            n_iter=20,
            n_particles=2,
        )

        first_algorithm = first._algorithm
        second_algorithm = second._algorithm

        assert isinstance(
            first_algorithm,
            MigrationTestAlgorithm,
        )

        assert isinstance(
            second_algorithm,
            MigrationTestAlgorithm,
        )

        first_thread = Thread(
            target=first.run,
            name="test-first-island",
        )

        second_thread = Thread(
            target=second.run,
            name="test-second-island",
        )

        try:
            migration.start()

            first.initialize_execution_context()
            second.initialize_execution_context()

            wait_for(
                lambda: set(driver._connections) == set(ISLAND_IDS),
            )

            assert first._migration_processor is not None
            first_communication = first._migration_processor._communication_processor

            assert second._migration_processor is not None
            second_communication = second._migration_processor._communication_processor

            assert first_communication._thread is not None  # type: ignore
            assert first_communication._thread.is_alive()  # type: ignore

            assert second_communication._thread is not None  # type: ignore
            assert second_communication._thread.is_alive()  # type: ignore

            # Start the receiver FIRST.
            #
            # Serial.run() executes migration_control(0) before update_status().
            # Therefore the receiver must complete iteration 0 and reach the
            # blocking point in iteration 1 before the migration message arrives.
            #
            # This guarantees that, when migration_control(2) eventually runs,
            # self._population already contains the initialized/ordered particles.
            second_thread.start()

            wait_for(
                lambda: second_algorithm._iteration_started.is_set(),
            )

            # The receiver is now blocked inside pre_iteration(1).
            #
            # Its population has already been initialized by iteration 0.
            wait_for(
                lambda: (
                    set(
                        second._population,
                    )
                    == {
                        f"{SECOND_ISLAND}|particle:0",
                        f"{SECOND_ISLAND}|particle:1",
                    }
                ),
            )

            # Start the donor only after the receiver is safely waiting at
            # iteration 1.
            first_thread.start()

            wait_for(
                lambda: first_algorithm._iteration_started.is_set(),
            )

            # Allow the donor to proceed from iteration 1 to iteration 2.
            first_algorithm._release_iteration.set()

            # The donor reaches migration_control(2), publishes its best particle
            # and the GlobalAsynchronous driver accepts it as the global best.
            wait_for(
                lambda: migration._global_best_fitness == 0.0,
            )

            # The driver has accepted the global-best message. Now wait until the
            # receiver's communication processor has actually received the TCP
            # message.
            wait_for(
                lambda: not second_communication.messages.empty(),
            )

            # The receiver is still blocked in pre_iteration(1), so its population
            # is intact and the migration message is waiting in its communication
            # queue. Release it so that it reaches migration_control(2), where the
            # message is consumed.
            second_algorithm._release_iteration.set()

            wait_for(
                lambda: (
                    (
                        particle := second_algorithm.population.get(
                            f"{SECOND_ISLAND}|particle:1",
                        )
                    )
                    is not None
                    and particle.fitness == 0.0
                    and particle.variables == {"x": 0.0}
                ),
            )

            first_population = first_algorithm.population
            second_population = second_algorithm.population

            # ------------------------------------------------------------------
            # Donor
            # ------------------------------------------------------------------

            assert set(first_population) == {
                f"{FIRST_ISLAND}|particle:0",
                f"{FIRST_ISLAND}|particle:1",
            }

            donor_best = first_population[f"{FIRST_ISLAND}|particle:0"]

            assert donor_best.identifier == (f"{FIRST_ISLAND}|particle:0")

            assert donor_best.variables == {
                "x": 0.0,
            }

            assert donor_best.fitness == 0.0

            donor_second = first_population[f"{FIRST_ISLAND}|particle:1"]

            assert donor_second.identifier == (f"{FIRST_ISLAND}|particle:1")

            assert donor_second.variables == {
                "x": 1.0,
            }

            assert donor_second.fitness == 1.0

            # ------------------------------------------------------------------
            # Receiver
            # ------------------------------------------------------------------

            assert set(second_population) == {
                f"{SECOND_ISLAND}|particle:0",
                f"{SECOND_ISLAND}|particle:1",
            }

            received_particle = second_population[f"{SECOND_ISLAND}|particle:1"]

            # The receiver particle keeps its own identity.
            assert received_particle.identifier == (f"{SECOND_ISLAND}|particle:1")

            # Its state was replaced by the donor's global best.
            assert received_particle.variables == {
                "x": 0.0,
            }

            assert received_particle.fitness == 0.0

            # The receiver's original best particle was not replaced.
            local_best = second_population[f"{SECOND_ISLAND}|particle:0"]

            assert local_best.identifier == (f"{SECOND_ISLAND}|particle:0")

            assert local_best.variables == {
                "x": 100.0,
            }

            assert local_best.fitness == 10000.0

            # Migration must not change the population size.
            assert len(first_population) == 2
            assert len(second_population) == 2

            first_thread.join(
                timeout=TIMEOUT,
            )

            second_thread.join(
                timeout=TIMEOUT,
            )

            assert not first_thread.is_alive()
            assert not second_thread.is_alive()

        finally:
            # Both algorithms can be blocked at iteration 1 if an assertion fails.
            first_algorithm._release_iteration.set()
            second_algorithm._release_iteration.set()

            if first_thread.is_alive():
                first_thread.join(
                    timeout=TIMEOUT,
                )

            if second_thread.is_alive():
                second_thread.join(
                    timeout=TIMEOUT,
                )

            first.finalize_execution_context()
            second.finalize_execution_context()

            migration.stop()

    def test_migration_checks_continue_at_configured_intervals(
        self,
    ) -> None:
        port = find_free_port()

        driver_stop_signal = LocalEvent()

        driver = SparkCommunicationDriver(
            island_ids=[FIRST_ISLAND],
            port=port,
            stop_signal=driver_stop_signal,
        )

        migration = GlobalAsynchronous(
            initial_iter=2,
            check_interval=3,
        )

        migration.initialize_context(
            communication_driver=driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": HOST,
                "port": port,
            },
        )

        driver.start()

        processor = Serial()

        algorithm = MigrationTestAlgorithm()

        algorithm.initialize_context(
            fitness_function=sphere,
            boundaries={
                "x": (-1000.0, 1000.0),
            },
        )

        processor.initialize_context(
            algorithm=algorithm,
            n_iter=8,
            n_particles=1,
            migration_driver=migration,
            seed=42,
        )

        processor.set_identifier(FIRST_ISLAND)

        processor._migration_processor = migration.create_processor_module(
            FIRST_ISLAND,
        )

        try:
            processor.initialize_execution_context()

            wait_for(
                lambda: FIRST_ISLAND in driver._connections,
            )

            migration_processor = processor._migration_processor

            assert migration_processor is not None

            processor.initialize_loop_context()

            assert migration_processor._synchronization_iter == 2

            processor.migration_control(0)

            assert migration_processor._synchronization_iter == 2

            processor.migration_control(1)

            assert migration_processor._synchronization_iter == 2

            processor.migration_control(2)

            assert migration_processor._synchronization_iter == 5

            processor.migration_control(3)

            assert migration_processor._synchronization_iter == 5

            processor.migration_control(4)

            assert migration_processor._synchronization_iter == 5

            processor.migration_control(5)

            assert migration_processor._synchronization_iter == 8

            processor.migration_control(6)

            assert migration_processor._synchronization_iter == 8

            processor.migration_control(7)

            assert migration_processor._synchronization_iter == 8

            processor.migration_control(8)

            assert migration_processor._synchronization_iter == 11

        finally:
            processor.finalize_execution_context()
            driver.stop()

    def test_stop_processes_pending_messages(self) -> None:
        port = find_free_port()

        driver_stop_signal = LocalEvent()

        driver = SparkCommunicationDriver(
            island_ids=ISLAND_IDS,
            port=port,
            stop_signal=driver_stop_signal,
        )

        migration = GlobalAsynchronous(
            initial_iter=INITIAL_ITER,
            check_interval=CHECK_INTERVAL,
        )

        migration.initialize_context(
            communication_driver=driver,
            communication_processor_class=SparkCommunicationProcessor,
            communication_processor_kargs={
                "driver_ip": HOST,
                "port": port,
            },
        )

        message = json.dumps(
            {
                "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
                "particle": {
                    "identifier": "source|particle:0",
                    "fitness": 1.0,
                    "variables": {
                        "x": 0.0,
                    },
                },
            }
        )

        driver.incoming_queues[FIRST_ISLAND].put(message)

        try:
            migration.start()
            migration.stop()

            assert migration._global_best_fitness == 1.0

        finally:
            migration.stop()


# ============================================================================
# Invalid protocol messages
# ============================================================================


class TestGlobalAsynchronousInvalidProtocol:
    """Integration tests for malformed protocol messages."""

    @pytest.mark.parametrize(
        "message",
        [
            "",
            "not-json",
            "{}",
            "[]",
            json.dumps(
                {
                    "type": "other",
                }
            ),
            json.dumps(
                {
                    "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
                }
            ),
            json.dumps(
                {
                    "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
                    "particle": None,
                }
            ),
            json.dumps(
                {
                    "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
                    "particle": [],
                }
            ),
            json.dumps(
                {
                    "type": GlobalAsynchronousProcessor.MESSAGE_TYPE,
                    "particle": {
                        "identifier": None,
                        "fitness": 0.0,
                    },
                }
            ),
        ],
    )
    def test_invalid_messages_are_ignored_by_processor(
        self,
        message: str,
        communication_environment,
    ) -> None:
        (
            _,
            _,
            processors,
            _,
            _,
        ) = communication_environment

        processor = create_migration_processor(
            communication_processor=processors[SECOND_ISLAND],
        )

        processors[SECOND_ISLAND].messages.put(
            message,
        )

        inserted: list[dict] = []
        departed: list[str] = []

        processor.migration_control(
            actual_iter=1,
            population={
                f"{SECOND_ISLAND}|particle:0": 10.0,
                f"{SECOND_ISLAND}|particle:1": 20.0,
            },
            local_best=None,
            insert_arrival_particle=inserted.append,
            departure_particle=departed.append,
        )

        assert inserted == []
        assert departed == []
