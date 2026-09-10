from threading import Event, Lock
from unittest.mock import Mock, patch

from tyrannis.backend.distributed.communication.spark_communication import (
    SparkCommunicationDriver,
    SparkCommunicationProcessor,
)


def test_processor_flushes_outgoing_message_when_it_is_enqueued() -> None:
    processor = SparkCommunicationProcessor(
        driver_ip="127.0.0.1",
        port=5000,
        identifier="island:0",
    )
    communication_socket = Mock()
    communication_thread = Mock()

    with (
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.socket.socket",
            return_value=communication_socket,
        ),
        patch(
            "tyrannis.backend.distributed.communication.spark_communication.Thread",
            return_value=communication_thread,
        ),
    ):
        processor.start()

    communication_socket.sendall.reset_mock()

    processor.outgoing_queue.put("migration")

    communication_socket.sendall.assert_called_once_with(b"\x02migration\x03")


def test_driver_flushes_outgoing_message_when_it_is_enqueued() -> None:
    driver = SparkCommunicationDriver(
        island_ids=["island:0"],
        port=5000,
    )
    communication_socket = Mock()

    driver._connections["island:0"] = communication_socket
    driver._running = Event()
    driver._running.set()
    driver._send_lock = Lock()

    driver.outgoing_queues["island:0"].put("migration")

    communication_socket.sendall.assert_called_once_with(b"\x02migration\x03")


def test_driver_flushes_queued_message_when_an_island_connects() -> None:
    driver = SparkCommunicationDriver(
        island_ids=["island:0"],
        port=5000,
    )
    communication_socket = Mock()

    driver._running = Event()
    driver._running.set()
    driver._send_lock = Lock()
    driver.outgoing_queues["island:0"].put("migration")

    assert driver.outgoing_queues["island:0"].qsize() == 1

    assert driver._register_connection(communication_socket, "island:0")

    communication_socket.sendall.assert_called_once_with(b"\x02migration\x03")
