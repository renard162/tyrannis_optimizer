from abc import ABC, abstractmethod
from typing import Any

from ..backend.local import Local
from ..migration.island_isolation import IslandIsolation
from ..processor.serial import Serial
from .algorithm import AlgorithmBase
from .backend import BackendBase
from .backend_migration import MigrationDriverBase
from .processor import ProcessorBase
from .results import HistoryConfig, OptimizationHistory, ProcessorResult
from .space import SpaceBase


class OrchestratorBase(ABC):
    """
    Base class for configuring optimization components.

    An orchestrator connects the user-facing optimization interface to the
    components that represent and solve a minimization problem. It selects
    components, supplies their shared configuration, and decides when their
    contexts are ready for execution. This class provides preparation methods
    without prescribing a public execution method or an execution lifecycle.

    Component responsibilities
    --------------------------
    The space represents the problem domain, encodes its boundaries, and
    decodes solver variables when evaluating the user's cost function. The
    algorithm owns particle transitions and population-level search behavior.
    The processor supplies the iteration mechanism on an execution resource
    and can serve as a template for independent island processors. The backend
    determines the execution model and collects the complete optimization
    result; a standalone backend may execute without using the processor.
    The migration driver coordinates exchanges between islands and creates
    their processor-side migration modules when the backend uses migration.

    The orchestrator wires these components together. It must not incorporate
    search-space encoding, mathematical update rules, iteration scheduling,
    communication protocols, or migration algorithms into this shared base.
    Component initialization and execution remain governed by their own
    contracts. In particular, migration context initialization belongs to the
    backend that provides its communication infrastructure.

    Attributes
    ----------
    _space : SpaceBase
        Search space used as the algorithm's fitness function and source of
        encoded boundaries.
    _algorithm : AlgorithmBase
        Algorithm supplied to the processor and backend.
    _processor : ProcessorBase
        Processor template available to the selected backend.
    _backend : BackendBase
        Execution backend responsible for producing a `ProcessorResult`.
    _migration : MigrationDriverBase
        Selected migration driver. Local execution always uses isolation.
    _seed : int or None
        Seed forwarded unchanged to the space, processor, and backend.
    _fitness_failure_strategy : str
        Fitness-evaluation failure strategy forwarded to the components.
        Its validation remains the responsibility of those components.
    _history_config : HistoryConfig
        Event selection shared with the processor and backend.
    _history : OptimizationHistory
        Independent history object initialized and updated explicitly by the
        concrete orchestrator through the shared history helpers.
    _n_particles : int
        Initial population size, per island when the backend uses islands.
    _n_iterations : int
        Iteration count forwarded when preparing the component contexts.
    _result : ProcessorResult or None
        Result exposed by the concrete orchestrator, or ``None`` before it has
        acquired a result. The subclass initializes and updates this attribute.
    history : OptimizationHistory
        Concrete property returning ``_history`` directly, without copying,
        materializing, filtering, or decoding its records.

    Implementing an orchestrator
    ---------------------------
    Implement ``__init__`` with the public signature appropriate to the concrete
    orchestrator. The abstract constructor performs no initialization and does
    not need to be called with ``super().__init__()``. Attribute declarations
    above define names and types only: they create no default instance state
    and no shared mutable values.

    Use ``_validate_common_arguments`` to validate iteration and particle
    counts before configuring components. ``_configure_components`` assigns
    all shared configuration attributes except ``_result`` and ``_history``,
    selects default components, and converts the requested history events into
    `HistoryConfig`.
    It retains supplied component instances without copying or resetting them.
    ``_configure_particle_history`` is also available independently when only
    event selection needs to be converted; it does not assign instance state.

    Initialize ``_result`` to ``None`` before any result accessor or execution
    path can read it, and initialize any subclass-specific lifecycle state
    before that state is used. The shared helpers do not manage these values.
    Initialize ``_history`` through ``_configure_history`` after the space has
    been initialized and before exposing ``history`` or updating its records.
    All component configuration attributes must be assigned before explicitly
    calling ``_initialize_context``. The constructor and the helpers are
    independent: the subclass chooses when preparation occurs and when
    execution begins.

    Contexts must be initialized in this order: space, algorithm, processor,
    backend. The space first prepares the encoded boundaries consumed by the
    algorithm; the processor then receives the prepared algorithm, and the
    backend receives the configured components for its execution model.
    ``_initialize_context`` preserves this dependency order. It prepares
    component state but does not execute the backend. Calling it again invokes
    component initialization again and can reset execution state, so it is
    not a continuation operation.

    The subclass defines its execution interface, result publication, exception
    handling, and policies for repeated execution. It must respect the chosen
    backend's capabilities and manage its own lifecycle state. No ``fit``,
    ``execute``, or ``__call__`` implementation is required by this base, and
    execution-specific state such as ``Optimizer._fit_state`` belongs solely
    to the concrete orchestrator.

    Results and history
    -------------------
    `HistoryConfig` selects the events recorded by existing component logging
    mechanisms. `ProcessorResult` contains the best-result dictionary (which
    can be ``None``) and the recorded history entries. The backend owns its
    result container; the concrete orchestrator decides when to retain that
    container in ``_result`` and how to expose its contents to users.

    ``_configure_history`` constructs an empty `OptimizationHistory` using the
    configured event selection and an initialized space. It captures metadata
    and an independent decoding snapshot through the history object's existing
    contract; it does not initialize the space or enable recording. Even before
    execution or with recording disabled, ``history`` exposes the configuration
    and decoded and internal columns, and its generators yield no records.

    ``_update_history`` replaces the contents of the history's record list with
    the complete records explicitly supplied by the subclass. It copies only
    list references to the existing JSON strings, without parsing or decoding
    them. It does not append an already accumulated history, infer missing
    iterations, or inspect a backend or result container. The history object,
    its record-list identity, and its decoding state are preserved across updates,
    including when the backend publishes a different result container.

    The concrete orchestrator chooses when to configure, update, preserve, or
    replace history. For continuation with the same schema and configuration,
    update the existing object with the complete accumulated records. Calling
    ``_configure_history`` again creates a new empty object and is appropriate
    only when the concrete lifecycle requires replacing the previous history,
    for example for a new optimization or a changed variable schema. Previously
    returned history objects remain usable but no longer receive these updates.

    `OptimizationHistory` retains no reference to the orchestrator, execution
    components, or complete result containers. Its decoding snapshot excludes
    evaluation resources and does not advance the optimization's random state.
    It remains usable independently after the orchestrator is released. The
    history object owns lazy iteration, event filtering, and decoding; the base
    only constructs it and synchronizes its records. Concurrent updates during
    iteration are not supported by the history contract.

    Public best-result access, export, execution recovery, and stopping policies
    remain outside this base's responsibilities. Components decide which partial
    results to publish after interruptions; the concrete orchestrator decides
    when to synchronize those records without duplicating component recovery.
    """

    _space: SpaceBase
    _algorithm: AlgorithmBase
    _processor: ProcessorBase
    _backend: BackendBase
    _migration: MigrationDriverBase
    _seed: int | None
    _fitness_failure_strategy: str
    _history_config: HistoryConfig
    _history: OptimizationHistory
    _n_particles: int
    _n_iterations: int
    _result: ProcessorResult | None

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Define the concrete orchestrator's public construction interface.

        Each subclass must implement this method because orchestrators can
        accept different problem descriptions and define different execution
        lifecycles. There is no common public construction signature beyond
        these generic arguments; they do not prescribe concrete parameters.

        Parameters
        ----------
        *args:
            Positional configuration defined by the concrete orchestrator.
        **kwargs:
            Keyword configuration defined by the concrete orchestrator.

        Notes
        -----
        The implementation owns argument interpretation and the initialization
        of its lifecycle state, ``_result``, and ``_history``. It may use the shared
        validation and configuration helpers, and must establish their
        documented preconditions before initializing component contexts.
        Preparation may occur during construction or at another point chosen
        by the concrete interface, before the components are used.

        This abstract method has no configuration or initialization logic.
        Subclasses do not need to call ``super().__init__()``. Implementing the
        constructor does not require implementing any execution method in this
        base, and it does not implicitly invoke the preparation helpers.
        """

    @property
    def history(self) -> OptimizationHistory:
        """Return the initialized history object without processing its records.

        The concrete orchestrator must call ``_configure_history`` before this
        property is used. Empty histories retain their configuration and columns;
        both generator properties yield no rows and invoke no decoding.
        """
        return self._history

    def _configure_history(self) -> None:
        """
        Create an empty history with independent decoding state and metadata.

        Returns
        -------
        None
            A new `OptimizationHistory` is assigned to ``_history``.

        Notes
        -----
        ``_history_config`` must be assigned and ``_space`` must be initialized
        before this call. The history constructor obtains its decoding snapshot
        through ``SpaceBase.copy_for_decoding`` without reinitializing the space
        or modifying the optimization's decoding state. Snapshot errors propagate.

        This helper does not read ``_result`` or any execution component, change
        recording flags, or collect records. The subclass must call it before
        exposing ``history``, even when recording is disabled. Calling it again
        replaces the history object; use ``_update_history`` for continuation
        with the existing schema and configuration instead.
        """
        self._history = OptimizationHistory(
            records=[], config=self._history_config, space=self._space
        )

    def _update_history(self, records: list[str]) -> None:
        """
        Synchronize the existing history with a complete published record list.

        Parameters
        ----------
        records:
            All JSON records currently published for this optimization, including
            earlier executions when continuing. Records must use the schema and
            decoding configuration captured by ``_configure_history``. An empty
            list empties the history without changing its metadata.

        Returns
        -------
        None
            The existing record list contains the supplied entries in order.

        Notes
        -----
        ``_configure_history`` must have completed before this call. The subclass
        chooses when to publish records, including after an interrupted execution,
        and must supply the full current list rather than an incremental batch.

        Slice assignment preserves the history object, its list, and its decoder
        while replacing earlier contents instead of appending duplicate records.
        Only references to immutable JSON strings are copied; no source list or
        result container is retained, and no parsing or decoding occurs. Passing
        the history's own record list requires no update. This helper performs
        no recovery and must not run concurrently with history iteration.
        """
        if records is not self._history.records:
            self._history.records[:] = records

    @staticmethod
    def _validate_common_arguments(n_iterations: int, n_particles: int) -> None:
        """
        Validate the shared iteration and particle counts without changing state.

        Parameters
        ----------
        n_iterations:
            Iteration count, checked first with ``isinstance(value, int)`` and
            required to be greater than or equal to one.
        n_particles:
            Population size, checked with the same conditions after iterations.

        Returns
        -------
        None
            Both counts satisfy the existing validation conditions.

        Raises
        ------
        ValueError
            A count is not an integer or is less than one.

        Notes
        -----
        No initialized instance or component context is required. Call this
        helper before assigning the common configuration. It performs only
        these two checks, does not coerce values, and changes no instance state.
        """
        if not isinstance(n_iterations, int) or n_iterations < 1:
            raise ValueError(
                "n_iterations must be an integer greater than or equal to 1."
            )

        if not isinstance(n_particles, int) or n_particles < 1:
            raise ValueError(
                "n_particles must be an integer greater than or equal to 1."
            )

    def _configure_components(
        self,
        space: SpaceBase,
        algorithm: AlgorithmBase,
        n_iterations: int,
        n_particles: int,
        backend: BackendBase | None = None,
        processor: ProcessorBase | None = None,
        migration: MigrationDriverBase | None = None,
        seed: int | None = None,
        fitness_failure_strategy: str = "raise",
        history: str | list[str] | None = None,
    ) -> None:
        """
        Assign the shared configuration without initializing component contexts.

        Parameters
        ----------
        space:
            Constructed search space for the optimization problem.
        algorithm:
            Constructed optimization algorithm.
        n_iterations:
            Validated iteration count passed to the component contexts.
        n_particles:
            Validated population size, per island for island-based backends.
        backend:
            Execution backend. ``None`` creates a new ``Local()`` instance.
        processor:
            Processor template. ``None`` creates a new ``Serial()`` instance.
        migration:
            Migration driver. ``None`` creates ``IslandIsolation()``. An
            instance of ``Local`` always selects a new ``IslandIsolation()``,
            even when another driver is explicitly supplied.
        seed:
            Optional seed stored for subsequent context initialization.
        fitness_failure_strategy:
            Failure strategy stored unchanged for component initialization.
        history:
            Event selection accepted by ``_configure_particle_history``.

        Returns
        -------
        None
            Configuration is stored in the shared instance attributes.

        Notes
        -----
        Call after ``_validate_common_arguments`` and before
        ``_initialize_context``. No existing instance attributes are required.
        Supplied components must already have been constructed; their
        optimization contexts need not yet be initialized.

        This method assigns all shared attributes except ``_result`` and
        ``_history``. It does not validate counts again or validate the failure
        strategy. History validation occurs last and propagates the helper's
        exceptions; prior assignments are not rolled back if validation fails.

        Supplied components are retained by reference, except for a migration
        driver replaced by local isolation. No component is copied, rebuilt,
        reset, or context-initialized here. This is a configuration operation,
        not an execution restart mechanism; result and lifecycle state remain
        the responsibility of the subclass.
        """
        self._space = space
        self._algorithm = algorithm
        self._n_iterations = n_iterations
        self._n_particles = n_particles
        self._backend = backend if backend is not None else Local()
        self._processor = processor if processor is not None else Serial()
        if not isinstance(self._backend, Local):
            self._migration = migration if migration is not None else IslandIsolation()
        else:
            self._migration = IslandIsolation()
        self._seed = seed
        self._fitness_failure_strategy = fitness_failure_strategy
        self._history_config = self._configure_particle_history(history)

    @staticmethod
    def _configure_particle_history(history: str | list[str] | None) -> HistoryConfig:
        """
        Convert a history event selection into a new configuration value.

        Parameters
        ----------
        history:
            ``None`` disables recording; ``"all"`` enables every event. A
            single event name or a list can select ``"migration"``,
            ``"pre_iteration"``, ``"new_particle"``, ``"error"``,
            ``"iteration"``, ``"status"``, and ``"best"``. An empty list
            disables recording and repeated event names are accepted.
            ``"all"`` cannot appear inside a list.

        Returns
        -------
        HistoryConfig
            Configuration with the selected event flags enabled.

        Raises
        ------
        ValueError
            An event is unsupported or ``"all"`` appears inside a list.
        TypeError
            The selection is neither ``None``, a string, nor a list, or a
            list entry cannot be used as an event dictionary key.

        Notes
        -----
        No initialized instance or context is required. Use when preparing
        the history selection before context initialization. The method does
        not modify the input, assign ``_history_config``, or record history.
        Recording and result storage remain with the existing components.
        """
        history_config = {
            "migration": False,
            "pre_iteration": False,
            "new_particle": False,
            "error": False,
            "iteration": False,
            "status": False,
            "best": False,
        }

        if history is None:
            return HistoryConfig(**history_config)

        if isinstance(history, str):
            if history == "all":
                history_config = dict.fromkeys(history_config, True)

            elif history in history_config:
                history_config[history] = True

            else:
                raise ValueError(f"Invalid history event: {history!r}.")

            return HistoryConfig(**history_config)

        if isinstance(history, list):
            if "all" in history:
                raise ValueError(
                    "'all' cannot be used inside a history list. "
                    "Use history='all' instead."
                )

            for event in history:
                if event not in history_config:
                    raise ValueError(f"Invalid history event: {event!r}.")

                history_config[event] = True

            return HistoryConfig(**history_config)

        raise TypeError("history must be None, a string, or a list of strings.")

    def _initialize_context(self) -> None:
        """
        Initialize space, algorithm, processor, and backend contexts in order.

        Returns
        -------
        None
            The component contexts have been prepared for execution.

        Notes
        -----
        This method takes no configuration arguments. All shared attributes
        other than ``_result`` and ``_history`` must already be assigned, normally
        by a successful ``_configure_components`` call after count validation.
        Any subclass-specific state observed during initialization must also
        be ready. Neither ``_result`` nor ``_history`` is read or assigned here.

        Call explicitly when preparing an optimization, before executing its
        components. The space receives the seed and prepares the boundaries;
        the algorithm receives that space, its encoded boundaries, and the
        iteration and particle counts. The processor receives the algorithm,
        counts, migration driver, seed, failure strategy, and history
        configuration. The backend receives the same configuration together
        with the processor. This order must be preserved because later
        components depend on the contexts prepared earlier.

        Initialization mutates the supplied component instances according to
        their own contracts, including their execution and result state.
        Migration initialization remains delegated to the backend. This helper
        does not execute the backend, publish its result, or change subclass
        lifecycle state. It provides no guard against repeated initialization:
        calling it again can reset component state and is not appropriate for
        continuing an existing execution context.

        Exceptions propagate immediately. Earlier initialization effects are
        not rolled back, and later components are not initialized after a
        failure. The concrete orchestrator owns the resulting lifecycle policy.
        """
        self._space.initialize_context(seed=self._seed)

        self._algorithm.initialize_context(
            fitness_function=self._space,
            boundaries=self._space.encoded_boundaries,
            n_iter=self._n_iterations,
            n_particles=self._n_particles,
        )

        self._processor.initialize_context(
            algorithm=self._algorithm,
            n_iter=self._n_iterations,
            n_particles=self._n_particles,
            migration_driver=self._migration,
            seed=self._seed,
            fitness_failure_strategy=self._fitness_failure_strategy,
            history_config=self._history_config,
        )

        self._backend.initialize_context(
            algorithm=self._algorithm,
            n_iter=self._n_iterations,
            n_particles=self._n_particles,
            migration=self._migration,
            processor=self._processor,
            seed=self._seed,
            fitness_failure_strategy=self._fitness_failure_strategy,
            history_config=self._history_config,
        )
