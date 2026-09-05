import inspect

from doubles.communication import DummyCommunicationDriver

from tyrannis.core.backend_communication import CommunicationDriverBase
from tyrannis.core.processor import LocalEvent


def test_is_abstract() -> None:
    assert inspect.isabstract(CommunicationDriverBase)


def test_implementation_satisfies_contract() -> None:
    stop_signal = LocalEvent()
    driver = DummyCommunicationDriver(
        ["island:0", "island:1"],
        stop_signal,
    )

    assert driver._island_ids == ["island:0", "island:1"]
    assert driver._stop_signal is stop_signal

    assert set(driver.incoming_queues) == {
        "island:0",
        "island:1",
    }
    assert set(driver.outgoing_queues) == {
        "island:0",
        "island:1",
    }


def test_incoming_and_outgoing_queues_are_distinct() -> None:
    driver = DummyCommunicationDriver(
        ["island:0"],
        LocalEvent(),
    )

    assert driver.incoming_queues is not driver.outgoing_queues
    assert driver.incoming_queues["island:0"] is not driver.outgoing_queues["island:0"]


def test_implementation_accepts_backend_configuration() -> None:
    driver = DummyCommunicationDriver(
        ["island:0"],
        LocalEvent(),
        port=5000,
    )

    assert driver._island_ids == ["island:0"]
