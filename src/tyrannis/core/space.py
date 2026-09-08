from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


class SpaceBase(ABC):
    """
    Base class for optimization search-space wrappers.

    A space is the interface between the user-defined domain of an optimization
    problem and the continuous numerical domain used internally by Tyrannis.
    It wraps the user-provided cost function and is responsible for translating
    values between these two representations.

    The user defines the optimization problem in an arbitrary domain supported
    by the concrete space implementation. During initialization, the space
    converts that domain into an encoded continuous search space and exposes
    its encoded boundaries to the optimization algorithm.

    During optimization, Tyrannis operates exclusively on the encoded
    continuous representation. When a candidate solution needs to be
    evaluated, the space decodes the continuous variables back into the
    representation expected by the user-defined cost function and evaluates
    that function using the decoded values.

    The complete evaluation flow is:

        encoded continuous variables
        -> __call__
        -> decode
        -> user-defined cost_function
        -> fitness

    Conversely, `decode` provides the conversion from the internal continuous
    representation to the original representation of the optimization
    problem.

    Because the space acts as a wrapper around the cost function, calling a
    space object is equivalent to evaluating the user-defined cost function
    through the space. The supplied continuous variables are first validated
    against the encoded `boundaries`; values outside those boundaries must be
    rejected rather than silently clipped, wrapped, or otherwise modified.

    The concrete space defines how the user's domain is encoded and decoded.
    `SpaceBase` therefore does not assume that the original variables are
    continuous, numerical, or represented in any particular way. The only
    invariant required by the optimization core is that the encoded
    representation exposed to Tyrannis is continuous and bounded.

    The space is also responsible for preserving the calling convention of
    the user-defined cost function. Depending on the concrete representation,
    decoded values may be supplied as positional or keyword arguments.

    The space must contain only the logic required to translate and evaluate
    the optimization problem. Optimization-algorithm behavior, population
    management, particle state, and interactions between particles do not
    belong in the space.
    """

    _cost_function: Callable[..., float] | None
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]
    _encoded_boundaries: dict[str, tuple[float, float]]

    @abstractmethod
    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the user-facing search-space interface.

        The constructor defines the public interface through which the user
        configures a search space and provides the cost function to be
        optimized. Each concrete space implementation is responsible for
        interpreting its own positional and keyword arguments and storing the
        configuration required to construct the corresponding encoded search
        space.

        The arguments following ``cost_function`` are intentionally generic
        because different search spaces require different configuration
        parameters. For example, a continuous space may receive a collection
        of lower and upper bounds for each cost-function input, while a binary
        space may receive the number of bits used to represent the search
        variables.

        The supplied ``cost_function`` represents the user-defined objective
        function. It must be preserved by the space because the space is
        responsible both for translating the solver representation back into
        the representation expected by the user and for evaluating the cost
        function using that representation.

        Implementations must store the constructor inputs required to create
        the encoded search-space representation during
        :meth:`initialize_context`. No execution-specific resources should be
        created by this method.

        Parameters
        ----------
        cost_function:
            User-defined cost function that receives the decoded search-space
            representation and returns its fitness value.
        *args:
            Positional arguments specific to the concrete search-space
            implementation.
        **kwargs:
            Keyword arguments specific to the concrete search-space
            implementation.
        """
        raise NotImplementedError

    @abstractmethod
    def initialize_context(self, seed: int | None = None) -> None:
        """
        Initialize the execution context of the search space.

        The implementation must use the configuration received by
        :meth:`__init__` to construct the encoded boundaries used by the
        optimization solver. These boundaries must be stored in the
        ``_encoded_boundaries`` attribute as a dictionary in the form
        ``dict[str, tuple[float, float]]``.

        The encoded boundaries define the continuous representation exposed
        to the solver. The implementation is responsible for converting the
        user-facing search-space configuration into this representation.

        ``seed`` is part of the search-space initialization contract even when
        the concrete search space does not require random-number generation.
        This guarantees a uniform interface for spaces whose encoding or
        initialization may depend on stochastic operations.

        Parameters
        ----------
        seed:
            Integer seed associated with the optimization execution. Concrete
            spaces may use it when random initialization or another
            seed-dependent operation is required.
        """
        raise NotImplementedError

    @abstractmethod
    def decode(self, float_inputs: dict[str, float]) -> Any:
        """
        Decode solver variables into the representation expected by the user.

        The input dictionary contains the continuous floating-point variables
        used internally by the optimization solver. The implementation must
        verify that every supplied value is within the corresponding bounds
        stored in ``_encoded_boundaries``. An input outside its encoded
        boundary must cause an exception to be raised rather than being
        silently clipped, wrapped, or otherwise modified.

        After validation, the implementation must convert the encoded values
        into the representation expected by the user-defined cost function.
        This representation is specific to the concrete search space. For
        example, a continuous space may return the decoded parameter values,
        while a binary space may convert floating-point solver variables into
        the corresponding bit representation.

        Parameters
        ----------
        float_inputs:
            Dictionary containing the solver representation of the search
            variables, with each value represented as a ``float``.

        Returns
        -------
        Any
            Values decoded into the representation required by the
            user-defined cost function.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def is_kwargs(self) -> bool:
        """
        Identify whether the user-provided search space can be used as keyword
        arguments when calling the cost function.

        Returns
        -------
        bool
            ``True`` when the search-space representation can be used as keyword
            arguments, or ``False`` otherwise.

        Raises
        ------
        NotImplementedError
            This property must be implemented by concrete search-space classes.
        """
        raise NotImplementedError

    @property
    def encoded_boundaries(self) -> dict[str, tuple[float, float]]:
        return self._encoded_boundaries.copy()

    def __call__(self, float_inputs: dict[str, float]) -> float:
        if self._cost_function is None:
            raise ValueError("cost_function cannot be None.")
        inputs = self.decode(float_inputs)

        if self.is_kwargs:
            return self._cost_function(**inputs)

        return self._cost_function(*inputs)
