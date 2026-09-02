import json
from abc import ABC, abstractmethod
from typing import Any

from ...algorithm.base import AlgorithmBase
from ..base import CostFunctionWrapperBase


class ParallelBackendBase(ABC):
    """Base class for parallel optimization backends."""

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

        For example, a Spark-based backend may receive a SparkSession,
        while other implementations may receive backend-specific resources
        such as an MPI communicator, a Ray context, or a Dask client.
        """

    @abstractmethod
    def execute(self) -> None:
        """
        Execute the optimization process.

        This method is responsible for performing the optimization using
        the execution model provided by the parallel backend.
        """

    def initialize_context(
        self,
        algorithm: AlgorithmBase,
        n_iter: int,
        n_particles: int,
        fitness_failure_strategy: str = "invalidate",
        seed: int | None = None,
    ) -> None:
        self._algorithm = algorithm
        self._n_iter = n_iter
        self._n_particles = n_particles
        self._fitness_failure_strategy = fitness_failure_strategy
        self._seed = seed

        self._result: dict[str, float | None | dict[str, float]] | None = None
        self._algorithm.configure(
            identifier=f"{self._identifier}|algorithm",
            cost_function_wrapper=self._cost_function_wrapper,
            seed=seed,
        )

    def update_result(self) -> None:
        if self._algorithm.local_best is None:
            return
        self._result = self._algorithm.local_best.get_result_data()

    @property
    def result(self):
        return self._result
