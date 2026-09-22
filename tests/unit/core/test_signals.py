"""State transitions of the local event."""

from tyrannis.core.signals import LocalEvent


def test_local_event_set_and_clear() -> None:
    event = LocalEvent()

    assert not event.is_set()
    event.set()
    assert event.is_set()
    event.clear()
    assert not event.is_set()
