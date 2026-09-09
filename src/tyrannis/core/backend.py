from abc import ABC, abstractmethod
from typing import Any

from .algorithm import AlgorithmBase, CostFunctionWrapperBase
from .backend_migration import MigrationDriverBase
from .processor import ProcessorBase


class BackendBase(ABC):
    """
    Base class for all optimization execution backends.

    A backend defines how an optimization is executed in a particular
    computational environment. It is the highest-level execution abstraction
    of Tyrannis and is responsible for connecting the optimization algorithm
    with the execution mechanism provided by the concrete backend.

    Different backends may implement fundamentally different execution models.
    A backend may operate standalone, such as a parallel backend that executes
    all particles within a single island and therefore has no migration
    concept, or it may use processors and migration to distribute the
    optimization across multiple islands.

    Consequently, `BackendBase` does not impose a particular population,
    processor, parallelization, distribution, or migration model. These are
    implementation details of concrete backends.

    The backend receives the optimization configuration through
    `initialize_context` and executes the configured optimization through
    `execute`. The concrete backend is responsible for translating the
    generic optimization context into its own execution model while
    preserving the contracts established by the core abstractions.

    The backend constructor defines the resources and configuration intrinsic
    to the execution environment. Optimization-specific configuration is
    provided later through `initialize_context`.

    The general lifecycle is:

        __init__
        -> initialize_context
        -> execute
        -> result

    `execute` may internally use completely different execution flows
    depending on the backend. For example, a standalone parallel backend may
    directly execute particles, whereas a distributed backend may coordinate
    processors, communication, and migration.

    The backend owns the final execution result, which becomes available
    through the `result` property after the optimization has completed.
    """

    _identifier: str
    _cost_function_wrapper: type[CostFunctionWrapperBase]

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the backend with resources intrinsic to its execution model.

        The constructor must receive only arguments that are specific to the
        concrete backend and to the computational environment in which it
        operates. Arguments that configure the optimization algorithm or a
        particular optimization execution must be provided later through
        `initialize_context`.

        A backend may require resources such as a Spark session, an MPI communicator,
        a Ray context, a Dask client, or other execution-environment-specific objects.
        The exact requirements are defined by the concrete backend.

        The constructor must also establish the backend identifier and the cost
        function wrapper class. The identifier must uniquely represent the
        backend instance within the optimization execution. The cost function
        wrapper must be assigned as a class derived from
        `CostFunctionWrapperBase`, rather than as an instance of that class.

        Notes
        -----
        The constructor must not require arguments that are specific to another
        backend execution model. In particular, the existence of processor or
        migration concepts must not be assumed by the base interface.

        A standalone backend may therefore require neither a processor nor a
        migration strategy, while a distributed backend may use both.
        """

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the configured optimization process.

        The concrete backend is responsible for executing the optimization
        according to its execution model. The implementation must use the
        optimization context configured through `initialize_context` and must
        preserve the behavioral contracts of the algorithm and other core
        components used by that backend.

        The internal execution flow is backend-specific. A standalone backend
        may execute the population directly, while a distributed backend may
        delegate execution to processors and coordinate multiple islands through
        communication and migration.

        The backend must not impose execution mechanisms that are not required by
        its own model. In particular, migration and processor-based execution are
        not requirements of this interface.

        After execution has completed, the backend must make the final
        optimization result available through the `result` property.
        """

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        migration: MigrationDriverBase,
        processor: ProcessorBase | None = None,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
    ) -> None:
        """
        Configure the optimization execution context.

        This method supplies the optimization algorithm and execution parameters
        required by the backend. The concrete backend may use only the components
        relevant to its execution model.

        Parameters
        ----------
        algorithm:
            Optimization algorithm to be executed.

        n_iter:
            Number of optimization iterations.

        n_particles:
            Number of initial particles allocated to each island.

            For execution models with a single island, such as serial or purely
            parallel backends, this corresponds to the initial population size.

            For distributed execution models, this corresponds to the initial
            population size of each island. The total initial number of particles
            is therefore determined by the number of islands and this value.

        migration:
            Migration strategy available to backends whose execution model uses
            migration between islands.

        processor:
            Optional processor used by backends whose execution model delegates
            particle execution to a processor. A standalone backend may not use
            this component.

        fitness_failure_strategy:
            Strategy used when evaluation of a particle's fitness fails.

        seed:
            Optional seed used to initialize the algorithm's random state.

        Notes
        -----
        This method separates optimization configuration from backend
        construction. The backend itself is constructed with resources intrinsic
        to its execution environment, while the optimization to be executed is
        supplied here.

        Not every argument is necessarily meaningful to every concrete backend.
        For example, a standalone backend may not use `migration` or `processor`,
        while a distributed backend may require both.

        The backend must configure the algorithm with the backend identifier and
        the cost-function wrapper associated with its execution model.
        """
        self._algorithm = algorithm
        self._n_iter = n_iter
        self._n_particles = n_particles
        self._processor = processor
        self._migration = migration
        self._fitness_failure_strategy = fitness_failure_strategy
        self._seed = seed

        self._result = None

        self._algorithm.configure(
            identifier=f"{self._identifier}|algorithm",
            cost_function_wrapper=self._cost_function_wrapper,
            seed=seed,
        )

    @property
    def identifier(self) -> str:
        return self._identifier

    @property
    def result(self) -> dict[str, str | float | dict[str, float]] | None:
        return self._result
