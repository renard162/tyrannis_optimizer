from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Queue

from .signals import LocalEvent


class CommunicationProcessorBase(ABC):
    """Abstract interface for the processor communication layer."""

    @abstractmethod
    def __init__(
        self,
        identification: str,
        **kwargs: object,
    ) -> None:
        """Initialize the processor communication layer.

        The constructor must receive and store all information required to
        identify the processor and configure the communication backend.

        The constructor must only initialize the communication object. It
        must not establish a connection with the driver, start communication
        threads, start background workers, or otherwise activate the
        communication backend.

        Activation of the communication backend must be performed exclusively
        by :meth:`start`.

        Implementations may require additional backend-specific parameters.
        Such parameters must be accepted by the concrete implementation's
        constructor and used only to configure the communication object.

        Parameters
        ----------
        identification:
            Unique identifier assigned to this processor. The identifier is
            used by the communication backend to associate the processor with
            its corresponding driver-side endpoint.
        **kwargs:
            Backend-specific configuration parameters required to initialize
            the communication layer.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def messages(self) -> Queue[str]:
        """Queue containing messages received from the driver.

        Each item in the queue is a message represented by a ``str``. The
        communication backend is responsible for receiving data from the driver,
        decoding it according to its underlying protocol, and placing each
        complete message into this queue.

        Consumers read received messages using the standard
        :class:`queue.Queue` interface. For example::

            message = processor.messages.get()

        To process messages without blocking, ``get_nowait()`` may be used::

            message = processor.messages.get_nowait()

        The queue contains only complete application-level messages. Transport
        details such as packet boundaries, framing, serialization, connection
        management, database rows, or protocol control fields must not be exposed
        through this API.

        The order in which messages are inserted into the queue must be preserved
        when they are consumed from the queue.

        An implementation must not replace the queue object while the
        communication backend is running.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def outgoing_queue(self) -> Queue[str]:
        """Queue containing messages to be sent to the driver.

        Each item placed in the queue represents one complete application-level
        message and must be a ``str``. The caller writes messages to the queue
        using the standard :class:`queue.Queue` interface. For example::

            processor.outgoing_queue.put("message")

        Multiple messages may be queued before the communication backend sends
        them::

            processor.outgoing_queue.put("message 1")
            processor.outgoing_queue.put("message 2")
            processor.outgoing_queue.put("message 3")

        The communication backend is responsible for consuming messages from
        this queue and converting them to the representation required by its
        underlying communication mechanism.

        The caller must not perform serialization, framing, encoding, socket
        operations, or any other transport-specific operation.

        Messages must be transmitted in the same order in which they are placed
        in the queue.

        The queue is an asynchronous boundary between the processor and the
        communication backend. Calling ``put()`` only places the message in the
        outgoing queue; it does not imply that the message has already been
        transmitted to the driver.

        An implementation must not replace the queue object while the
        communication backend is running.
        """
        raise NotImplementedError

    @abstractmethod
    def start(
        self,
        stop_signal: LocalEvent,
    ) -> None:
        """Start the processor communication backend.

        The communication backend must be started asynchronously so that this
        method does not block the processor's main execution loop.

        After this method returns, the communication backend must be running in
        parallel with the processor's main execution loop. Communication tasks
        such as receiving messages, processing incoming data, and transmitting
        messages placed in :attr:`outgoing_queue` must be performed independently
        of the processor's main loop.

        The implementation may use threads, processes, asynchronous tasks, or
        any other mechanism appropriate for its underlying communication
        protocol. The mechanism used to achieve parallel execution is an
        implementation detail and must not be exposed through this interface.

        The method must configure the communication backend to use the supplied
        stop signal for communication with the processor's main loop. The
        communication backend must not control the processor's synchronization
        or iteration state. Synchronization and migration decisions are the
        responsibility of the migration processor.

        Parameters
        ----------
        stop_signal:
            Event used by the communication backend to signal that the
            communication must stop.

        Notes
        -----
        This method must return only after the communication backend has been
        successfully initialized and its asynchronous execution has been
        started.

        The method must not wait for the communication backend to terminate.
        Waiting for termination would prevent the processor's main execution loop
        from running concurrently with the communication backend.

        Calling this method must not require the caller to execute any additional
        communication loop manually.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop the processor communication backend and release its resources."""
        raise NotImplementedError


class CommunicationDriverBase(ABC):
    """Abstract interface for the driver communication layer."""

    @abstractmethod
    def __init__(
        self,
        island_ids: list[str],
        stop_signal: LocalEvent,
        **kwargs: object,
    ) -> None:
        """Initialize the driver communication layer.

        The constructor must receive and store all information required to
        identify the communication endpoints managed by the driver and to
        configure the communication backend.

        The constructor must only initialize the communication object. It
        must not start a server, bind or listen on a communication endpoint,
        establish connections, start communication threads, start background
        workers, or otherwise activate the communication backend.

        Activation of the communication backend must be performed exclusively
        by :meth:`start`.

        Implementations may require additional backend-specific parameters.
        Such parameters must be accepted by the concrete implementation's
        constructor and used only to configure the communication object.

        Parameters
        ----------
        island_ids:
            Identifiers of the processor endpoints that this driver is
            responsible for communicating with.

        stop_signal:
            Event used by the communication backend to signal and propagate a
            global stop condition.

        **kwargs:
            Backend-specific configuration parameters required to initialize
            the communication layer.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def incoming_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages received from processors.

        The dictionary maps each processor identifier to a
        :class:`queue.Queue` containing the messages received from that
        processor.

        The dictionary key is the processor identifier used by the driver to
        associate messages with their source. Each dictionary value is a
        ``Queue[str]`` in which every item represents one complete
        application-level message received from the corresponding processor.

        Messages are read from the queues using the standard
        :class:`queue.Queue` interface. For example::

            message = driver.incoming_queues["island:0"].get()

        To read a message without blocking, ``get_nowait()`` may be used::

            message = driver.incoming_queues["island:0"].get_nowait()

        Multiple messages may be waiting in the same queue::

            message_1 = driver.incoming_queues["island:0"].get()
            message_2 = driver.incoming_queues["island:0"].get()

        Messages must be inserted into the queue in the same order in which
        they are received from the corresponding processor.

        The communication backend is responsible for receiving the data,
        identifying its source processor, decoding and deserializing it as
        necessary, and placing each complete application-level message into
        the appropriate queue.

        The messages exposed by this API must not contain transport-specific
        information. Protocol framing, serialization, packet boundaries,
        sockets, database rows, message-broker metadata, or other details of
        the underlying communication mechanism must remain internal to the
        implementation.

        The caller only interacts with the resulting ``Queue[str]`` objects.
        It must not perform any transport-specific receive operation.

        The dictionary and its queues must remain valid while the communication
        backend is running. An implementation must not replace the dictionary
        or its queues during normal operation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def outgoing_queues(self) -> dict[str, Queue[str]]:
        """Queues containing messages to be sent to processors.

        The dictionary maps each processor identifier to a
        :class:`queue.Queue` containing the messages that the driver intends to
        send to that processor.

        The dictionary key is the processor identifier that determines the
        destination of the messages. Each dictionary value is a
        ``Queue[str]`` in which every item represents one complete
        application-level message that must be delivered to the corresponding
        processor.

        The caller writes messages to the queues using the standard
        :class:`queue.Queue` interface. For example::

            driver.outgoing_queues["island:0"].put("message")

        Multiple messages may be queued before the communication backend sends
        them::

            driver.outgoing_queues["island:0"].put("message 1")
            driver.outgoing_queues["island:0"].put("message 2")
            driver.outgoing_queues["island:0"].put("message 3")

        Messages placed in a queue must be sent to the processor identified by
        that queue's dictionary key. Messages must be transmitted in the same
        order in which they are placed in each processor's queue.

        The communication backend is responsible for consuming the messages
        from these queues and converting them into whatever representation is
        required by the underlying communication mechanism.

        The caller must not perform serialization, framing, encoding, socket
        operations, database insertion, message-broker operations, or any other
        transport-specific operation.

        Calling ``put()`` only places the message in the outgoing queue. It does
        not imply that the message has already been transmitted to the target
        processor or that the processor has received it.

        The queues therefore constitute an asynchronous boundary between the
        driver and the communication backend.

        The dictionary and its queues must remain valid while the communication
        backend is running. An implementation must not replace the dictionary
        or its queues during normal operation.
        """
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """Start the driver communication backend.

        The communication backend must be started asynchronously so that this
        method does not block the driver's main execution loop.

        After this method returns, the communication backend must be running in
        parallel with the driver's main execution loop. Communication tasks such
        as accepting connections, receiving messages, routing messages to the
        appropriate processor queues, and transmitting messages placed in
        :attr:`outgoing_queues` must be performed independently of the driver's
        main loop.

        The implementation may use threads, processes, asynchronous tasks, or
        any other mechanism appropriate for its underlying communication
        protocol. The mechanism used to achieve parallel execution is an
        implementation detail and must not be exposed through this interface.

        The method is responsible for activating the communication mechanism,
        such as starting a server, opening a database polling mechanism, creating
        communication workers, or establishing other backend-specific resources.

        Parameters
        ----------
        None

        Notes
        -----
        This method must return only after the communication backend has been
        successfully initialized and its asynchronous execution has been
        started.

        The method must not wait for the communication backend to terminate.
        Waiting for termination would prevent the driver's main execution loop
        from running concurrently with the communication backend.

        Calling this method must not require the caller to execute any additional
        communication loop manually.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """Stop the driver communication backend and release its resources."""
        raise NotImplementedError
