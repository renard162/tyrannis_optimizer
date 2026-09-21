from typing import Protocol


class EventProtocol(Protocol):
    """
    Define the synchronization event contract used by Tyrannis.

    Any synchronization primitive used by processors, migration modules, or
    communication modules must provide these operations. Implementations do not
    need to inherit from this protocol; compatibility is determined
    structurally.

    This allows Tyrannis to use different event implementations, such as
    `LocalEvent`, multiprocessing events, manager-backed events, or
    backend-specific synchronization primitives, without coupling the core
    interfaces to a concrete event implementation.
    """

    def set(self) -> None:
        """Set the event state."""
        ...

    def clear(self) -> None:
        """Clear the event state."""
        ...

    def is_set(self) -> bool:
        """Return whether the event is currently set."""
        ...


class LocalEvent:
    """
    Provide an in-process implementation of the Tyrannis event contract.

    This event stores its synchronization state locally and does not provide
    synchronization between threads or processes. It is intended for execution
    models in which the event is accessed within the same execution context.

    `LocalEvent` satisfies `EventProtocol` structurally without requiring
    explicit inheritance from the protocol.
    """

    def __init__(self) -> None:
        self._state: bool = False

    def set(self) -> None:
        self._state = True

    def clear(self) -> None:
        self._state = False

    def is_set(self) -> bool:
        return self._state
