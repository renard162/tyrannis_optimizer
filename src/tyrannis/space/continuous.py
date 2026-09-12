from collections.abc import Callable
from typing import TypeAlias, cast

from tyrannis.core.space import SpaceBase

Boundary: TypeAlias = tuple[float, float]
Boundaries: TypeAlias = list[Boundary] | dict[str, Boundary]


class Continuous(SpaceBase):
    """Continuous optimization search space."""

    _boundaries: Boundaries

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        boundaries: Boundaries,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing continuous search space.

        The cost function is supplied by the user and is evaluated after the
        solver representation has been decoded. The search-space boundaries
        can be provided either as a list of lower and upper limits or as a
        dictionary associating each variable with its limits.

        When a list is provided, each tuple represents the lower and upper
        limits of one positional input of the cost function. When a
        dictionary is provided, each key identifies one input of the cost
        function and its associated tuple contains its lower and upper
        limits.

        Parameters
        ----------
        cost_function:
            User-defined cost function to be evaluated after decoding the
            solver inputs. It may be ``None`` during construction, but
            ``initialize_context`` requires a valid cost function.
        boundaries:
            Search-space boundaries. A list contains one ``(lower, upper)``
            tuple for each positional input. A dictionary maps each input
            name to its ``(lower, upper)`` tuple.
        use_cache:
            Whether cost-function evaluations should be cached. When
            ``False``, no cache is created or used.
        cache_type:
            Cache strategy to use when caching is enabled. Supported
            strategies are ``"lru"``, ``"lfu"``, ``"fifo"``, ``"rr"``, and
            ``"disk"``. The default ``"lru"`` uses the Python standard
            library.
        cache_size:
            Maximum cache size. For in-memory caches, this represents the
            maximum number of cached records. For the disk cache, this
            represents the maximum size in megabytes.
        """
        super().__init__(
            cost_function,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
        )
        self._boundaries = boundaries
        self._is_kwargs = isinstance(boundaries, dict)

    def initialize_context(self, seed: int | None = None) -> None:
        if isinstance(self._boundaries, dict):
            self._encoded_boundaries = self._boundaries.copy()
            return

        self._encoded_boundaries = {
            str(index): boundary for index, boundary in enumerate(self._boundaries)
        }

    def decode(self, float_inputs: dict[str, float]) -> list[float] | dict[str, float]:
        self._check_input_bounds(float_inputs)

        if isinstance(self._boundaries, dict):
            return float_inputs

        return [float_inputs[str(index)] for index in range(len(self._boundaries))]

    def encode_cache(
        self,
        inputs: list[float] | dict[str, float],
    ) -> tuple[float, ...] | tuple[tuple[str, float], ...]:
        if isinstance(inputs, dict):
            return tuple(sorted(inputs.items()))

        return tuple(inputs)

    def decode_cache(
        self,
        inputs: tuple[float, ...] | tuple[tuple[str, float], ...],
    ) -> list[float] | dict[str, float]:
        if self.is_kwargs:
            return dict(cast(tuple[tuple[str, float], ...], inputs))

        return list(cast(tuple[float, ...], inputs))
