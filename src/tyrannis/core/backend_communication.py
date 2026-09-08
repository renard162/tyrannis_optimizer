from __future__ import annotations

from abc import ABC, abstractmethod
from queue import Queue

from .signals import LocalEvent


class CommunicationProcessorBase(ABC):
    """
    Base class for the communication layer running alongside an optimization
    processor.

    A communication processor provides the transport-independent interface
    between one processor and the driver. It is responsible only for
    transporting complete application-level messages and exposing them through
    queues. It must not contain migration logic, optimization logic, or
    decisions about how received messages affect the processor or algorithm.

    The communication processor has two asynchronous boundaries:

    - `messages` receives complete messages from the driver;
    - `outgoing_queue` receives complete messages produced by the processor-side
      components that must be sent to the driver.

    The communication implementation is responsible for all transport-specific
    operations, including connections, serialization, framing, encoding,
    routing, retries, and background execution. None of these details should
    be exposed through this interface.

    The communication processor is started by `start` and remains active
    independently of the processor's optimization loop until `stop` is called.
    Therefore, communication may receive messages while the optimization
    processor is executing other work.

    `set_message_signal` provides an optional synchronization mechanism for
    notifying the processor-side execution that a new message has been placed
    in `messages`. The signal only reports communication activity; it must not
    itself determine how or when the received message modifies processor or
    algorithm state. Synchronization of migration and algorithm execution is
    the responsibility of the migration layer.

    Implementations must keep the communication interface independent from the
    concrete transport mechanism. A TCP implementation, multiprocessing
    implementation, Spark implementation, or any other backend must expose the
    same application-level contract through this class.
    """

    @abstractmethod
    def __init__(
        self,
        identification: str,
        **kwargs: object,
    ) -> None:
        """
        Initialize the processor communication configuration.

        The constructor must configure the communication object but must not
        activate the communication backend.

        In particular, the constructor must not establish connections, start
        communication threads or processes, start background workers, or
        otherwise perform operations that require an active execution
        environment.

        Parameters
        ----------
        identification:
            Unique identifier assigned to this processor. The identifier is
            used by the communication backend to associate this processor with
            its corresponding driver-side endpoint.

        **kwargs:
            Backend-specific configuration parameters required to configure
            the communication layer.

        Notes
        -----
        Constructor arguments must describe communication configuration only.
        Migration-specific behavior and optimization state must not be handled
        by the communication layer.

        Runtime resources are initialized by `start` and released by `stop`.
        This separation is important because processor communication objects
        may be constructed before they are materialized in their final
        execution environment.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def messages(self) -> Queue[str]:
        """
        Return the queue containing complete messages received from the driver.

        Returns
        -------
        Queue[str]
            Queue containing complete application-level messages received from
            the driver.

        Notes
        -----
        Each queue item must be a complete message represented by `str`.
        Transport details such as packet boundaries, framing, encoding,
        serialization, connection metadata, or protocol control fields must
        not be exposed through this interface.

        The communication backend is responsible for receiving and decoding
        transport data and placing each complete application-level message in
        this queue.

        Message order must be preserved: messages must be inserted into the
        queue in the same order in which they are received from the driver.

        Consumers interact with the queue using the standard
        :class:`queue.Queue` interface, for example::

            message = processor.messages.get()

        or, for non-blocking access::

            message = processor.messages.get_nowait()

        The queue object must remain stable while the communication backend is
        running. An implementation must not replace the queue during normal
        operation.

        The queue represents received communication only. Reading or
        interpreting a message does not imply that the message's contents
        should immediately modify processor or algorithm state.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def outgoing_queue(self) -> Queue[str]:
        """
        Return the queue containing messages to be sent to the driver.

        Returns
        -------
        Queue[str]
            Queue into which complete application-level messages are placed
            for transmission to the driver.

        Notes
        -----
        Each item placed in the queue must be a complete message represented
        by `str`.

        The communication backend is responsible for consuming messages from
        this queue and converting them into whatever representation is
        required by the underlying transport.

        The caller must not perform serialization, framing, encoding, socket
        operations, or any other transport-specific operation.

        Messages must be transmitted in the same order in which they are
        placed in the queue.

        Calling `put()` only places a message in the queue. It does not imply
        that the message has already been transmitted or received by the
        driver.

        The queue is therefore an asynchronous boundary between the
        processor-side logic and the communication backend.

        The queue object must remain stable while the communication backend is
        running. An implementation must not replace the queue during normal
        operation.
        """
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """
        Start the processor communication backend.

        The communication backend must begin its communication activity
        asynchronously and return control to the processor without waiting
        for the communication runtime to terminate.

        Notes
        -----
        After this method returns, the communication backend must be operating
        independently of the processor's main optimization loop.

        The implementation may use threads, processes, asynchronous tasks, or
        any other mechanism appropriate to its transport. The mechanism is an
        implementation detail and must not be exposed through this interface.

        `start` must initialize all runtime resources required by the
        communication backend. It must not require the caller to execute an
        additional communication loop.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the processor communication backend and release its resources.

        This method must terminate all communication activity started by
        `start` and release every runtime resource owned by the communication
        processor.

        Notes
        -----
        After this method returns, no communication thread, process, worker,
        connection, or other runtime resource created by `start` may remain
        active.

        The method must also leave the communication processor in a state
        where its runtime resources are no longer considered available.
        """
        raise NotImplementedError

    @abstractmethod
    def set_message_signal(
        self,
        message_signal: LocalEvent | None,
    ) -> None:
        """
        Set or clear the signal used to notify message arrival.

        Parameters
        ----------
        message_signal:
            Shared local event used to notify the processor-side execution
            that at least one complete application-level message has been
            inserted into `messages`.

            Passing `None` disables message-arrival signaling. This is used
            when the loop context associated with the signal has been
            finalized.

        Notes
        -----
        Whenever a complete application-level message is inserted into
        `messages`, the communication backend must set the configured signal.

        The signal is a notification mechanism, not a migration or algorithm
        synchronization policy. The communication backend must not use the
        signal to directly modify processor or algorithm state.

        In particular, receiving a message and setting this signal must not
        cause the communication layer to insert, remove, or otherwise modify
        particles. The migration processor is responsible for deciding when
        and how received messages affect the optimization state.

        The supplied signal may be a shared synchronization object created for
        the processor's loop context. The communication implementation must
        retain the exact object supplied by this method while signaling is
        enabled.

        Passing `None` must disable further signaling and release the
        communication layer's reference to the previous signal when
        appropriate.
        """
        raise NotImplementedError


class CommunicationDriverBase(ABC):
    """
    Base class for the communication layer running on the optimization driver.

    A communication driver provides the transport-independent communication
    interface between the driver and all optimization processors. It is
    responsible only for transporting complete application-level messages,
    routing received messages to their source processor, and routing outgoing
    messages to their destination processor.

    The driver communication layer manages one incoming and one outgoing queue
    for each processor:

    - `incoming_queues[processor_id]` contains messages received from that
      processor;
    - `outgoing_queues[processor_id]` contains messages that must be sent to
      that processor.

    Queues form the asynchronous boundary between the driver-side application
    logic and the communication backend. The application places or consumes
    complete messages through these queues without knowing how the messages
    are transported.

    The communication driver must remain independent of migration and
    optimization logic. It must not interpret application-level migration
    messages or modify algorithm state based on their contents. Its
    responsibility ends at delivering messages to the appropriate queues.

    The communication backend runs independently of the driver's main
    execution loop after `start` and remains active until `stop`.

    Concrete implementations may use TCP, multiprocessing, Spark, message
    brokers, shared memory, or any other transport mechanism, provided that
    they preserve the communication contract defined by this class.
    """

    @abstractmethod
    def __init__(
        self,
        island_ids: list[str],
        **kwargs: object,
    ) -> None:
        """
        Initialize the driver communication configuration.

        The constructor must configure the communication object and establish
        the endpoints that it will manage, but must not activate the
        communication backend.

        In particular, the constructor must not start a server, bind or listen
        on a communication endpoint, establish active connections, start
        communication threads or processes, or otherwise activate runtime
        communication.

        Parameters
        ----------
        island_ids:
            Identifiers of all processor endpoints that this driver is
            responsible for communicating with.

        **kwargs:
            Backend-specific configuration parameters required to configure
            the communication layer.

        Notes
        -----
        The constructor should establish configuration and data structures
        required by the communication backend. Runtime resources that require
        an active execution environment must be created by `start`.

        The communication driver must not contain migration-specific or
        algorithm-specific configuration.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def incoming_queues(self) -> dict[str, Queue[str]]:
        """
        Return the queues containing messages received from processors.

        Returns
        -------
        dict[str, Queue[str]]
            Dictionary mapping each processor identifier to the queue
            containing complete application-level messages received from that
            processor.

        Notes
        -----
        The dictionary key identifies the source processor. Each queue must
        contain only complete messages received from that processor.

        The communication backend is responsible for identifying the source,
        decoding or deserializing transport data as necessary, and inserting
        the resulting application-level messages into the corresponding queue.

        Messages received from the same processor must be inserted in their
        reception order.

        Transport-specific information such as packet boundaries, framing,
        serialization, sockets, database rows, message-broker metadata, or
        connection state must not be exposed through this interface.

        Consumers interact only with the queues. For example::

            message = driver.incoming_queues["island:0"].get()

        or, for non-blocking access::

            message = driver.incoming_queues["island:0"].get_nowait()

        The dictionary and its queues must remain valid while the
        communication backend is running. An implementation must not replace
        them during normal operation.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def outgoing_queues(self) -> dict[str, Queue[str]]:
        """
        Return the queues containing messages to be sent to processors.

        Returns
        -------
        dict[str, Queue[str]]
            Dictionary mapping each processor identifier to the queue
            containing complete application-level messages destined for that
            processor.

        Notes
        -----
        The dictionary key identifies the destination processor. Each item
        placed in a queue must be delivered to the processor represented by
        that key.

        The communication backend is responsible for consuming messages from
        these queues and converting them into whatever representation is
        required by the underlying transport.

        The caller must not perform serialization, framing, encoding, socket
        operations, database operations, message-broker operations, or other
        transport-specific work.

        Messages placed in each queue must be transmitted in insertion order.

        Calling `put()` only places a message in the outgoing queue. It does
        not imply that the message has already been transmitted or received
        by the target processor.

        The dictionary and its queues must remain valid while the
        communication backend is running. An implementation must not replace
        them during normal operation.
        """
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """
        Start the driver communication backend.

        The communication backend must begin operating asynchronously and
        return control to the driver's main execution without waiting for the
        communication runtime to terminate.

        Notes
        -----
        After this method returns, the communication backend must operate
        independently of the driver's main execution loop.

        Depending on the transport, this may include starting a server,
        opening communication endpoints, establishing connections, starting
        communication threads or processes, or initializing other
        backend-specific runtime resources.

        The implementation may use any mechanism appropriate to the transport.
        The mechanism used to achieve asynchronous execution is an
        implementation detail and must not be exposed through this interface.

        The method must not require the caller to execute an additional
        communication loop manually.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the driver communication backend and release its resources.

        This method must terminate all communication activity started by
        `start` and release every runtime resource owned by the communication
        driver.

        Notes
        -----
        After this method returns, no communication thread, process, worker,
        server, connection, selector, or other runtime resource created by
        `start` may remain active.

        The communication driver must not leave background communication
        activity running after the driver-side execution has finished.
        """
        raise NotImplementedError
