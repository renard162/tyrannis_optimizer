import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import numpy as np

from ..core.results import HistoryConfig
from .algorithm import AlgorithmBase
from .backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from .signals import LocalEvent


class MigrationProcessorBase(ABC):
    """
    Base class for migration strategies executed by an optimization processor.

    A migration processor implements the migration behavior of a single
    processor (island). It is responsible for the migration protocol and for
    coordinating the interaction between the processor, the communication
    layer, and the migration strategy. It must not directly manipulate the
    processor's internal state or create dependencies between processors.

    The migration processor has two independent lifecycle boundaries.
    `start` and `stop` delimit the lifetime of the migration runtime and are
    responsible for resources required by the execution environment,
    including resources that must be created after distributed serialization
    and destroyed before the processor execution ends.

    `initialize_loop_context` and `finalize_loop_context`, on the other hand,
    delimit the lifetime of resources associated specifically with the
    optimization loop. These resources may also be non-serializable. A
    typical example is a multiprocessing context manager that creates shared
    objects used by all processes participating in the loop.

    Consequently, no non-serializable object may be created in the
    constructor. Such objects must be created in `start` or
    `initialize_loop_context`, according to their lifetime:

    - `start`: resources required by the migration runtime but not tied to
      the optimization loop.
    - `initialize_loop_context`: resources created for and shared during the
      optimization loop.

    Resources created in `start` must be destroyed in `stop`, while resources
    created in `initialize_loop_context` must be destroyed in
    `finalize_loop_context`. Destruction should normally be explicit, such as
    assigning `None` to the attribute holding the resource.

    The execution lifecycle is:

        __init__
        -> start
        -> initialize_loop_context
        -> migration_control (repeated for each iteration)
        -> finalize_loop_context
        -> stop

    During `initialize_loop_context`, the migration strategy receives a
    `migration_signal`. This is a shared synchronization object used by the
    processor, communication layer, and migration strategy to coordinate
    migration-related execution.

    The signal may be used in different ways depending on the migration
    strategy. For asynchronous strategies, it can coordinate when migration
    events received by the communication layer may be consumed and their
    effects applied to the processor or algorithm state. For synchronous
    strategies, it can also be used to block the optimization loop until the
    communication layer receives the command required to continue execution,
    such as a command from the driver to execute up to a specific iteration
    and then wait.

    Therefore, `migration_signal` is not merely a notification that migration
    data is available. It is part of the synchronization contract between the
    processor, communication layer, and migration strategy. Concrete
    implementations must preserve its synchronization semantics and must not
    replace it with independent synchronization mechanisms when coordinating
    the same execution state.

    `migration_control` is the main extension point of a concrete migration
    strategy. It is called during the optimization loop and receives the
    current population and the best particle found during the current
    iteration (`iter_best`), together with callbacks through which it can
    request particle insertion and removal.

    The migration strategy must remain independent of the implementation of
    the processor and the optimization algorithm. Interactions with the local
    population and synchronization with the communication layer must occur
    through the contract provided by this class.
    """

    _algorithm: AlgorithmBase | None = None

    @abstractmethod
    def __init__(
        self,
        initial_iter: int,
        communication_processor: CommunicationProcessorBase,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the migration strategy configuration.

        The constructor must initialize only configuration and state that can
        safely be serialized.

        Non-serializable runtime resources must never be created here. They
        must be created in `start` when they belong to the migration runtime,
        or in `initialize_loop_context` when their lifetime is restricted to
        the optimization loop.

        Parameters
        ----------
        initial_iter:
            First iteration from which the migration strategy may operate.

        communication_processor:
            Communication processor used by the migration strategy to exchange
            migration messages with the driver.

        Notes
        -----
        Concrete implementations may define additional strategy-specific
        parameters, provided that construction remains safe for the
        serialization model of the execution backend.
        """
        raise NotImplementedError

    @abstractmethod
    def start(self) -> None:
        """
        Initialize the migration runtime after distributed serialization.

        This method is called when processor execution starts and is intended
        for creating runtime resources that must not exist when the migration
        processor is serialized.

        Resources created here are not tied to the optimization loop. They
        remain available throughout the processor execution, including before
        and after the loop context.

        Notes
        -----
        Any non-serializable resource created by this method must be destroyed
        in `stop`.

        This method must not create resources whose lifetime is restricted to
        the optimization loop; those belong in `initialize_loop_context`.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_loop_context(self, migration_signal: LocalEvent) -> None:
        """
        Initialize resources associated with the optimization loop and establish
        migration synchronization.

        This method is called immediately before the processor enters its
        optimization loop. It is intended for resources whose lifetime is
        explicitly tied to that loop. These resources may be non-serializable,
        such as a multiprocessing context manager and the shared objects created
        from it for communication between the processes participating in the
        loop.

        Parameters
        ----------
        migration_signal:
            Shared synchronization event used by the processor, communication
            layer, and migration strategy to coordinate execution.

            The signal is a fundamental part of the migration synchronization
            protocol. Its exact use depends on the migration strategy.

            In asynchronous strategies, it can coordinate the consumption and
            application of migration events so that changes to processor or
            algorithm state occur only during the appropriate migration-control
            phase.

            In synchronous strategies, it can be used to hold the optimization
            loop until the communication layer receives the command that allows
            the processor to continue. For example, a driver may instruct the
            processor to execute until iteration X and then wait for the next
            synchronization command.

            The migration strategy must therefore use the supplied shared signal
            as the synchronization mechanism between the processor and the
            communication layer. It must not replace it with an independent
            synchronization primitive for the same execution boundary.

        Notes
        -----
        Every resource created here must be destroyed in
        `finalize_loop_context`.

        The implementation must establish all loop-scoped synchronization state
        before the optimization loop starts.

        The signal may be used to coordinate either the application of migration
        events, the progression of the optimization loop, or both. The concrete
        strategy is responsible for defining the appropriate synchronization
        behavior while preserving the contract of the shared signal.
        """
        raise NotImplementedError

    @abstractmethod
    def finalize_loop_context(self) -> None:
        """
        Finalize the optimization loop context and release its resources.

        This method is called immediately after the processor leaves the
        optimization loop.

        Every resource created by `initialize_loop_context` must be explicitly
        released here. This includes non-serializable objects and shared
        multiprocessing resources created for the loop.

        Notes
        -----
        Loop-scoped resources must not be left alive until `stop`, because
        `stop` represents a different lifecycle boundary.

        Destruction should normally be explicit, such as assigning `None` to
        attributes holding loop-scoped resources.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """
        Finalize the migration runtime and release execution resources.

        This method is called when processor execution ends, after
        `finalize_loop_context` has released all loop-scoped resources.

        Every non-serializable resource created by `start` must be explicitly
        destroyed here.

        Notes
        -----
        Resources belonging to the optimization loop must already have been
        released by `finalize_loop_context` and must not be managed here.
        """
        raise NotImplementedError

    @abstractmethod
    def migration_control(
        self,
        actual_iter: int,
        population: dict[str, float],
        iter_best: str | None,
        insert_arrival_particle: Callable[[dict[str, Any]], None],
        departure_particle: Callable[[str], None],
    ) -> None:
        """
        Execute the migration strategy for the current iteration.

        This is the main extension point of a concrete migration strategy. The
        processor calls this method during the optimization loop, allowing the
        strategy to coordinate migration and synchronization with the
        communication layer.

        The `migration_signal` established by `initialize_loop_context` is the
        synchronization mechanism shared by the processor, communication layer,
        and migration strategy. Its exact use depends on the migration strategy.

        In asynchronous strategies, communication may receive migration data
        independently of the optimization loop. The migration strategy uses the
        synchronization protocol to determine when those events may be consumed
        and when their effects may be applied to the processor or algorithm state.

        In synchronous strategies, this method may participate in controlling the
        progression of the optimization loop. The processor may be required to
        wait until the communication layer receives authorization from the driver
        to continue, for example when the driver instructs the processor to
        execute until a specific iteration and then wait for the next command.

        Therefore, `migration_control` represents the migration and
        synchronization phase of an iteration. Implementations must preserve the
        synchronization semantics established through `migration_signal` and must
        not allow communication activity to modify processor or algorithm state
        outside the synchronization boundary defined by the migration strategy.

        Parameters
        ----------
        actual_iter:
            Current processor iteration.

        population:
            Current processor population represented as a dictionary mapping
            particle identifiers to their fitness values.

            The population is ordered by fitness in ascending order, with the
            best particle first and the worst particle last. Tyrannis minimizes
            objective functions, so lower fitness values represent better
            solutions.

            A fitness value of `None` indicates that the particle has not yet
            received a valid fitness evaluation.

        iter_best:
            Best particle found during the current optimization iteration.

            This value represents the best result of the current iteration and
            must not be confused with the algorithm's historical `local_best`.
            The latter represents the best solution accumulated over previous
            iterations, whereas `iter_best` represents the current iteration
            state used by migration control.

            `None` indicates that no valid iteration-best particle is currently
            available.

        insert_arrival_particle:
            Callback used to request insertion of a particle received through
            migration into the local processor population.

            The migration strategy must use this callback rather than directly
            modifying the processor's population. The callback is responsible
            for applying the insertion according to the processor's population
            management rules.

        departure_particle:
            Callback used to request removal of a particle selected for migration
            from the local processor population.

            The migration strategy must use this callback rather than directly
            modifying the processor's population.

        Notes
        -----
        The migration strategy must not directly access or modify the processor's
        population, algorithm state, or other processor internals. Population
        modifications must be requested exclusively through the supplied
        callbacks.

        Communication and migration processing must be separated from the direct
        modification of processor state. A communication thread or process may
        receive, prepare, or signal migration data asynchronously, but it must not
        directly modify the processor or algorithm state as a consequence of
        receiving that data.

        For asynchronous strategies, `migration_control` is the synchronization
        boundary at which pending migration effects can be applied to the local
        processor.

        For synchronous strategies, `migration_control` may also be responsible
        for waiting for a synchronization signal that authorizes the processor to
        proceed. In particular, an implementation may block the optimization loop
        after reaching a synchronization point until the driver provides the next
        execution command.

        The concrete implementation is responsible for defining the migration
        policy and synchronization behavior. The processor remains responsible
        for executing the requested population changes and maintaining the
        consistency of its own state.

        The implementation must not replace the shared `migration_signal` with an
        independent synchronization primitive when coordinating the same
        processor, communication, and migration state.
        """
        raise NotImplementedError


class MigrationDriverBase(ABC):
    """
    Base class for migration strategies configured by the user on the driver.

    This is the user-facing base class for a migration strategy. A concrete
    implementation is instantiated by the user to configure the behavior of
    migration for an optimization algorithm.

    The migration strategy is divided into two cooperating components:

    - the migration driver, which is created and managed by the user on the
      driver side and coordinates the migration strategy;
    - the migration processor, which is created by this class for each
      optimization processor and executes the processor-side portion of the
      migration protocol.

    Therefore, this class is also responsible for creating and configuring
    the `MigrationProcessorBase` instances that will operate alongside the
    optimization processors. Each processor receives an independent
    migration processor and communication processor.

    The constructor must require only configuration that is directly related
    to the migration algorithm and is supplied by the user. Configuration or
    runtime dependencies belonging to the backend, communication layer, or
    processor execution environment must not be required by the constructor.
    Those dependencies are provided later through `initialize_context` and
    `create_processor_module`.

    The constructor should establish only the serializable configuration of
    the migration strategy. Although the migration driver itself is not
    serialized, keeping its constructor free of non-serializable runtime
    resources is important because the migration processor configuration
    created from it may subsequently participate in parallel or distributed
    execution.

    Non-serializable resources required by the driver-side migration runtime
    must be created in `start` and released in `stop`. Non-serializable
    resources whose lifetime is tied to the optimization loop belong to the
    corresponding `MigrationProcessorBase` and must be created and released
    through its loop-context lifecycle.

    The driver-side lifecycle is:

        __init__
        -> initialize_context
        -> start
        -> create_processor_module (for each processor)
        -> [processor execution]
        -> stop

    `initialize_context` supplies the communication infrastructure required by
    the migration strategy after its user configuration has been established.

    `create_processor_module` creates the processor-side migration module and
    its corresponding communication processor for a specific optimization
    processor. The resulting module is independent from the other processor
    modules.

    `start` initializes resources required by the driver-side migration
    runtime, while `stop` terminates that runtime and releases its resources.

    The driver-side migration strategy must communicate with processors
    through the communication contract. It must not directly manipulate
    processor populations or depend on processor internals.

    Concrete migration strategies may implement different migration and
    synchronization models, including asynchronous strategies in which
    processors operate independently and synchronous strategies in which the
    driver coordinates when processors may proceed.
    """

    _processor_class: type[MigrationProcessorBase]
    _communication_driver: CommunicationDriverBase
    _communication_processor_class: type[CommunicationProcessorBase]
    _migration_processor_init_kargs: dict[str, Any]
    _history_buffer: list[str]

    @abstractmethod
    def __init__(self, initial_iter: int = 1, *args: Any, **kargs: Any) -> None:
        """
        Initialize the user-defined configuration of the migration strategy.

        The constructor is the public configuration point of a concrete migration
        strategy. Its arguments must represent only parameters that directly
        define the migration algorithm and that are meaningful for the user to
        configure.

        Backend dependencies, communication objects, processor-specific
        information, execution resources, and other infrastructure concerns must
        not be required as constructor arguments. These dependencies are supplied
        later through `initialize_context` and are used by
        `create_processor_module` when constructing the processor-side migration
        modules.

        Parameters
        ----------
        initial_iter:
            First iteration from which the migration strategy may operate.

        *args:
            Additional positional arguments containing only migration-algorithm
            configuration defined by the concrete strategy.

        **kargs:
            Additional keyword arguments containing only migration-algorithm
            configuration defined by the concrete strategy.

        Notes
        -----
        The migration driver itself is not serialized. Nevertheless, the
        constructor must not create non-serializable runtime resources. Keeping
        construction limited to user-provided migration configuration ensures
        that the configuration can safely be propagated when the corresponding
        `MigrationProcessorBase` instances are prepared for parallel or
        distributed execution.

        Non-serializable resources required by the driver-side migration runtime
        must be created in `start` and released in `stop`.

        Dependencies belonging to the communication infrastructure are provided
        after construction through `initialize_context` and must not be treated
        as user configuration.

        A concrete implementation should call the base constructor when the base
        class provides initialization required by the migration processor
        construction contract.
        """
        raise NotImplementedError

    def initialize_context(
        self,
        communication_driver: CommunicationDriverBase,
        communication_processor_class: type[CommunicationProcessorBase],
        communication_processor_kargs: dict[str, Any],
        history_config: HistoryConfig,
        seed: int | None,
    ) -> None:
        """
        Configure the communication context used by the migration strategy.

        This method connects the migration driver to the communication driver
        and stores the communication processor class and initialization
        arguments required to construct processor-side migration modules.

        Parameters
        ----------
        communication_driver:
            Driver-side communication backend used to exchange migration
            messages with the processors.

        communication_processor_class:
            Communication processor class that will be instantiated for each
            optimization processor.

        communication_processor_kargs:
            Serializable keyword arguments used to initialize each
            communication processor. The processor-specific `identifier`
            is added by `create_processor_module`.

        Notes
        -----
        This method configures the communication context; it does not start
        the communication runtime. Runtime resources must be started by
        `start`.

        The supplied communication processor arguments are copied before
        being stored, so later modifications to the original dictionary do
        not alter the migration configuration.

        Concrete migration strategies overriding this method must call
        `super().initialize_context(...)` unless they intentionally replace
        the base implementation in its entirety. The base implementation is
        responsible for storing the communication context required by
        `create_processor_module`.
        """
        self._communication_driver = communication_driver
        self._communication_processor_class = communication_processor_class
        self._history_config = history_config
        self._migration_processor_init_kargs["communication"] = (
            communication_processor_kargs.copy()
        )
        self._history_buffer = []
        self._seed = seed
        self._rng = np.random.default_rng(seed)

    def create_processor_module(self, identifier: str) -> MigrationProcessorBase:
        """
        Create the migration processor module for one optimization processor.

        This method creates both the communication processor and the
        migration processor associated with the specified processor.

        Parameters
        ----------
        identifier:
            Unique identifier of the optimization processor. It is passed
            to the communication processor so that communication endpoints
            can be associated with the correct processor.

        Returns
        -------
        MigrationProcessorBase
            A new migration processor configured with an independent
            communication processor for the specified processor.

        Notes
        -----
        Each call creates a new communication processor and a new migration
        processor. Processor modules must not share mutable processor-side
        runtime state with one another.

        The communication processor receives a copy of the communication
        initialization arguments and the processor identifier.

        Migration-specific initialization arguments are kept separate from
        communication arguments. The communication configuration is removed
        before constructing the migration processor itself.

        This method is responsible only for construction. Runtime resources
        must be initialized according to the lifecycle contract of the
        resulting migration processor.
        """
        communication_kargs = self._migration_processor_init_kargs[
            "communication"
        ].copy()

        communication_processor = self._communication_processor_class(
            **communication_kargs,
            identifier=identifier,
        )

        migration_processor_kargs = self._migration_processor_init_kargs.copy()
        migration_processor_kargs.pop("communication")

        return self._processor_class(
            communication_processor=communication_processor,
            **migration_processor_kargs,
        )

    @abstractmethod
    def start(self) -> None:
        """
        Start the driver-side migration runtime.

        This method is called after the migration context has been configured
        and when the migration strategy is ready to begin execution.

        It is the appropriate place to create non-serializable resources whose
        lifetime belongs to the driver-side migration runtime, such as
        threads, synchronization primitives, or runtime communication state
        that cannot safely exist during serialization.

        Notes
        -----
        Every non-serializable resource created by `start` must be explicitly
        released by `stop`, normally by stopping the resource and then
        assigning `None` to the corresponding attribute when appropriate.

        `start` must not be used to create resources that belong to the
        processor-side optimization loop. Those resources are managed by the
        corresponding `MigrationProcessorBase` lifecycle.
        """
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        """
        Stop the driver-side migration runtime and release its resources.

        This method is called when the migration execution is finished.

        Every runtime resource created by `start` must be stopped and released
        here. In particular, threads, communication resources, and other
        non-serializable objects owned by the driver-side migration runtime
        must not remain active after this method returns.

        Notes
        -----
        Resources belonging to processor-side loop contexts are not managed
        by this method. Their lifetime is controlled by the corresponding
        `MigrationProcessorBase` implementation.
        """
        raise NotImplementedError

    def migration_log(
        self, particle: dict[str, Any], origin: str, destination: str, iteration: int
    ) -> None:
        if not self._history_config.migration:
            return

        self._history_buffer.append(
            json.dumps(
                {
                    "iteration": iteration,
                    "event": self._history_config.get_event("migration"),
                    "origin": origin,
                    "destination": destination,
                    "particle": particle,
                }
            )
        )

    def consume_migration_history(self) -> list[str]:
        try:
            return self._history_buffer
        finally:
            self._history_buffer = []
