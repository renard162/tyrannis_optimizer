from abc import ABC, abstractmethod
from collections.abc import Callable
from importlib.util import find_spec
from typing import Any, ClassVar

import numpy as np


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

    The space may optionally cache cost-function evaluations. When enabled,
    the cache operates on the representation expected by the cost function,
    using `encode_cache` and `decode_cache` to create a reversible canonical
    cache key.

    Runtime cache state is local to the process in which it is created and is
    excluded from serialization. This prevents cache contents from being
    transmitted when a space is distributed to another process or worker.
    """

    _cost_function: Callable[..., float] | None
    _args: tuple[Any, ...]
    _kwargs: dict[str, Any]
    _encoded_boundaries: dict[str, tuple[float, float]]
    _use_cache: bool
    _cache_type: str
    _cache_size: int
    _cached_cost_function: Callable[[Any], float] | None

    _CACHE_DEPENDENCIES: ClassVar[dict[str, str | None]] = {
        "lru": None,
        "lfu": "cachetools",
        "fifo": "cachetools",
        "rr": "cachetools",
        "disk": "joblib",
    }

    _CACHE_FACTORIES: ClassVar[dict[str, str]] = {
        "lru": "_create_lru_cache",
        "lfu": "_create_lfu_cache",
        "fifo": "_create_fifo_cache",
        "rr": "_create_rr_cache",
        "disk": "_create_disk_cache",
    }

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        *args: Any,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the user-facing search-space interface.

        Concrete space implementations must call this method through
        ``super().__init__``. The base initialization configures the common
        search-space state and the optional cost-function cache, while the
        concrete implementation remains responsible for interpreting and
        storing its own space-specific configuration.

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
        use_cache:
            Whether cost-function evaluations should be cached. When ``False``,
            no cache is created or used, regardless of ``cache_type``.
        cache_type:
            Cache strategy to use when caching is enabled. Supported
            strategies are ``"lru"``, ``"lfu"``, ``"fifo"``, ``"rr"``, and
            ``"disk"``. The default ``"lru"`` uses the Python standard library.
        cache_size:
            Maximum cache size. For in-memory caches, this represents the
            maximum number of cached records. For the disk cache, this
            represents the maximum size in megabytes.
        **kwargs:
            Keyword arguments specific to the concrete search-space
            implementation.
        """
        self._cost_function = cost_function
        self._args = args
        self._kwargs = kwargs
        self._use_cache = use_cache
        self._cache_type = cache_type
        self._cache_size = cache_size
        self._cached_cost_function = None

        self._validate_cache()

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

    def encode(self, inputs: Any) -> Any:
        """
        Encode values into the representation used by the search space.

        Spaces that do not require an explicit encoding operation may use the
        default implementation, which returns the supplied values unchanged.
        Concrete spaces may override this method when their user-facing
        representation requires an encoding operation.
        """
        return inputs

    @abstractmethod
    def encode_cache(self, inputs: Any) -> Any:
        """
        Encode cost-function inputs into a canonical cache key.

        The cache representation is independent from the solver encoding and
        decoding performed by the search space. The resulting value must be
        suitable for use as a cache key and must preserve enough information
        for :meth:`decode_cache` to reconstruct the representation expected by
        the cost function.

        Parameters
        ----------
        inputs:
            Representation expected by the user-defined cost function.

        Returns
        -------
        Any
            Canonical, hashable representation used as the cache key.
        """
        raise NotImplementedError

    @abstractmethod
    def decode_cache(self, inputs: Any) -> Any:
        """
        Decode a canonical cache key into the representation expected by the
        cost function.

        This operation must be compatible with :meth:`encode_cache`, such
        that a valid cost-function input can be reconstructed from its cache
        key.

        Parameters
        ----------
        inputs:
            Canonical cache key produced by :meth:`encode_cache`.

        Returns
        -------
        Any
            Representation expected by the user-defined cost function.
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

    def __call__(self, float_inputs: dict[str, float]) -> np.float64:
        """
        Evaluate the cost function using the supplied encoded variables.

        The encoded variables are first decoded into the representation
        expected by the user-defined cost function. When caching is disabled,
        the cost function is evaluated directly. When caching is enabled, the
        decoded representation is converted into a canonical cache key and
        the cached evaluation is used.
        """
        if self._cost_function is None:
            raise ValueError("cost_function cannot be None.")

        inputs = self.decode(float_inputs)

        if not self._use_cache:
            return np.float64(self._evaluate(inputs))

        cache_key = self.encode_cache(inputs)
        cached_cost_function = self._get_cached_cost_function()

        return np.float64(cached_cost_function(cache_key))

    def _evaluate(self, inputs: Any) -> float:
        """
        Evaluate the user-defined cost function using the configured calling
        convention.
        """
        if self._cost_function is None:
            raise RuntimeError("self._cost_function cannot be None.")
        if self.is_kwargs:
            return self._cost_function(**inputs)

        return self._cost_function(*inputs)

    def _evaluate_cache_key(self, cache_key: Any) -> float:
        """
        Decode a cache key and evaluate the user-defined cost function.
        """
        inputs = self.decode_cache(cache_key)
        return self._evaluate(inputs)

    def _get_cached_cost_function(self) -> Callable[[Any], float]:
        """
        Return the runtime cache wrapper, creating it on first use.
        """
        if self._cached_cost_function is None:
            self._cached_cost_function = self._create_cached_cost_function()

        return self._cached_cost_function

    def _create_cached_cost_function(self) -> Callable[[Any], float]:
        """
        Create the configured cache wrapper for the cost function.
        """
        factory_name = self._CACHE_FACTORIES[self._cache_type]
        factory = getattr(self, factory_name)

        return factory(self._evaluate_cache_key, self._cache_size)

    def _validate_cache(self) -> None:
        """
        Validate the configured cache type, size, and optional dependency.

        Cache configuration is only validated when caching is enabled. When
        ``use_cache`` is ``False``, the cache configuration has no effect and
        optional cache dependencies are not inspected.
        """
        if not self._use_cache:
            return

        if not isinstance(self._cache_type, str):
            raise TypeError("cache_type must be a string.")

        if self._cache_type not in self._CACHE_DEPENDENCIES:
            raise ValueError(
                f"Invalid cache_type: {self._cache_type!r}. "
                f"Expected one of: {tuple(self._CACHE_DEPENDENCIES)}."
            )

        if not isinstance(self._cache_size, int) or isinstance(self._cache_size, bool):
            raise TypeError("cache_size must be an integer.")

        if self._cache_size <= 0:
            raise ValueError("cache_size must be greater than zero.")

        dependency = self._CACHE_DEPENDENCIES[self._cache_type]

        if dependency is not None and find_spec(dependency) is None:
            raise ImportError(
                f"cache_type={self._cache_type!r} requires the {dependency!r} package."
            )

    @staticmethod
    def _create_lru_cache(
        function: Callable[[Any], float],
        cache_size: int,
    ) -> Callable[[Any], float]:
        """
        Create an LRU cache using the Python standard library.

        The standard-library LRU cache is thread-safe and does not require
        additional synchronization for access to its internal state.
        """
        from functools import lru_cache

        return lru_cache(maxsize=cache_size)(function)

    @staticmethod
    def _create_lfu_cache(
        function: Callable[[Any], float],
        cache_size: int,
    ) -> Callable[[Any], float]:
        """
        Create an LFU cache using cachetools.

        A reentrant lock protects concurrent access to the cache. When the
        installed cachetools version supports ``condition``, it is also used
        to prevent concurrent evaluations of the same cache key.
        """
        from threading import Condition, RLock

        import cachetools
        from cachetools import LFUCache, cached

        lock = RLock()
        cache = LFUCache(maxsize=cache_size)
        cached_kwargs: dict[str, Any] = {"lock": lock}

        major_version = int(cachetools.__version__.split(".", maxsplit=1)[0])

        if major_version >= 6:
            cached_kwargs["condition"] = Condition(lock)

        return cached(cache, **cached_kwargs)(function)

    @staticmethod
    def _create_fifo_cache(
        function: Callable[[Any], float],
        cache_size: int,
    ) -> Callable[[Any], float]:
        """
        Create a FIFO cache using cachetools.

        A reentrant lock protects concurrent access to the cache. When the
        installed cachetools version supports ``condition``, it is also used
        to prevent concurrent evaluations of the same cache key.
        """
        from threading import Condition, RLock

        import cachetools
        from cachetools import FIFOCache, cached

        lock = RLock()
        cache = FIFOCache(maxsize=cache_size)
        cached_kwargs: dict[str, Any] = {"lock": lock}

        major_version = int(cachetools.__version__.split(".", maxsplit=1)[0])

        if major_version >= 6:
            cached_kwargs["condition"] = Condition(lock)

        return cached(cache, **cached_kwargs)(function)

    @staticmethod
    def _create_rr_cache(
        function: Callable[[Any], float],
        cache_size: int,
    ) -> Callable[[Any], float]:
        """
        Create a random-replacement cache using cachetools.

        A reentrant lock protects concurrent access to the cache. When the
        installed cachetools version supports ``condition``, it is also used
        to prevent concurrent evaluations of the same cache key.
        """
        from threading import Condition, RLock

        import cachetools
        from cachetools import RRCache, cached

        lock = RLock()
        cache = RRCache(maxsize=cache_size)
        cached_kwargs: dict[str, Any] = {"lock": lock}

        major_version = int(cachetools.__version__.split(".", maxsplit=1)[0])

        if major_version >= 6:
            cached_kwargs["condition"] = Condition(lock)

        return cached(cache, **cached_kwargs)(function)

    @staticmethod
    def _create_disk_cache(
        function: Callable[[Any], float],
        cache_size: int,
    ) -> Callable[[Any], float]:
        """
        Create a disk-backed cache using joblib.

        ``cache_size`` is currently reserved for future disk-cache size
        management and is not used by this implementation.
        """
        from tempfile import TemporaryDirectory
        from typing import cast

        from joblib import Memory

        temporary_directory = TemporaryDirectory(prefix="tyrannis_cache_")
        memory = Memory(location=temporary_directory.name, verbose=0)

        def evaluate(cache_key: Any) -> float:
            return function(cache_key)

        cached = memory.cache(evaluate)

        def cached_function(cache_key: Any) -> float:
            _ = temporary_directory
            return cast(float, cached(cache_key))

        return cached_function

    def __getstate__(self) -> dict[str, Any]:
        """
        Return the serializable state of the search space.

        The runtime cache is deliberately excluded from the serialized state.
        A deserialized space therefore starts with an empty cache in the
        process where it is evaluated.
        """
        state = self.__dict__.copy()
        state["_cached_cost_function"] = None
        return state
