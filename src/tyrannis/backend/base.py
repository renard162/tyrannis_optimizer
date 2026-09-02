from abc import ABC, abstractmethod
from typing import Any

from ..algorithm.base import AlgorithmBase, CostFunctionWrapperBase
from .distributed.communication.base import CommunicationBase
from .distributed.migration.base import MigrationBase
from .processor.base import ProcessorBase


class BackendBase(ABC):
    """Base class for all backends."""

    _identifier: str
    _cost_function_wrapper: type[CostFunctionWrapperBase]

    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """
        Initialize the backend interface with the user.

        This method must receive only arguments that are specific to the
        backend implementation. Arguments related to the optimization
        algorithm or to the optimization process itself must not be defined
        here.

        The backend identifier and cost function wrapper class must also be
        set during initialization. The identifier must uniquely represent
        the backend instance within the optimization execution. The cost
        function wrapper must be assigned as a class derived from
        ``CostFunctionWrapperBase``, rather than as an instance of that
        class.

        For example, a Spark-based backend may receive a SparkSession,
        while other implementations may receive backend-specific resources
        such as an MPI communicator, a Ray context, or a Dask client.
        """

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the optimization process.

        The execution consists of an initialization stage followed by
        ``n_iter + 1`` execution cycles. Iteration zero is reserved for
        establishing the initial state of the population and therefore does
        not represent an iterative step of the optimization algorithm itself.

        At the beginning of each iteration, the current iteration number is
        registered in the backend state and the algorithm's ``pre_iteration``
        method is called. This stage is responsible for the pre-iteration
        processing of the algorithm, including, primarily, the creation and
        destruction of dynamic particles during the optimization process.

        After the pre-iteration processing, any new particles identified by
        the algorithm are initialized using the parallel execution mechanism
        and incorporated into the population. For iterations greater than
        zero, the particles belonging to the current population are then
        processed in parallel to update their state, and the resulting
        particles are incorporated into the population.

        Once the particle states have been updated, the algorithm's
        ``post_iteration`` method is called. This stage consolidates the
        results obtained from the particles during the iteration, including
        the evaluation of the cost-function values for their candidate
        solutions and the corresponding update of the optimization state.

        After all ``n_iter + 1`` iterations have been completed, the final
        optimization result is updated from the best solution obtained by
        the algorithm.
        """

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        processor: ProcessorBase | None = None,
        migration: MigrationBase | None = None,
        communication: CommunicationBase | None = None,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._n_iter = n_iter
        self._n_particles = n_particles
        self._processor = processor
        self._migration = migration
        self._communication = communication
        self._fitness_failure_strategy = fitness_failure_strategy
        self._seed = seed

        self._result: dict[str, float | None | dict[str, float]] | None = None
        self._algorithm.configure(
            identifier=f"{self._identifier}|algorithm",
            cost_function_wrapper=self._cost_function_wrapper,
            seed=seed,
        )

    @property
    def identifier(self):
        return self._identifier

    @property
    def result(self):
        return self._result
