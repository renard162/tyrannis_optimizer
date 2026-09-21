from collections.abc import Callable
from typing import TypeAlias, cast

from ..core.space import SpaceBase
from . import register_space

Boundary: TypeAlias = tuple[float, float]
Boundaries: TypeAlias = list[Boundary] | dict[str, Boundary]


class Continuous(SpaceBase):
    """Continuous optimization search space."""

    _boundaries: Boundaries

    def __init__(
        self,
        boundaries: Boundaries | Boundary,
        cost_function: Callable[..., float] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing continuous search space.

        The continuous search space represents optimization variables directly
        as real-valued inputs. Each variable is defined by a lower and upper
        boundary that constrains the continuous representation used by the
        optimization algorithm.

        Boundaries can be provided either as a list of intervals or as a
        dictionary associating each variable with its interval. A list is
        interpreted as positional inputs of the cost function, while a
        dictionary is interpreted as keyword inputs.

        Parameters
        ----------
        boundaries:
            Search-space boundaries. A list contains one ``(lower, upper)``
            tuple for each positional input. A dictionary maps each input
            name to its ``(lower, upper)`` tuple. The lower boundary must be
            smaller than the upper boundary.

        cost_function:
            User-defined cost function evaluated after decoding the solver
            inputs into their user-facing representation. This argument is
            required when the space is used independently. It does not need
            to be provided when the space is used as a component of a
            ``Mixed`` space, because in that case the cost function is
            provided to the ``Mixed`` space itself.

        use_cache:
            Whether cost-function evaluations should be cached. When
            ``False``, no cache is created or used.

        cache_type:
            Cache strategy to use when caching is enabled. Supported strategies are:

            - ``"lru"``: Least Recently Used cache. (default)
            - ``"disk"``: Persistent disk-based cache.
            - ``"lfu"``: Least Frequently Used cache. Requires the optional
            dependencies for advanced caching.
            - ``"fifo"``: First In, First Out cache. Requires the optional
            dependencies for advanced caching.
            - ``"rr"``: Random Replacement cache. Requires the optional dependencies
            for advanced caching.

        cache_size:
            Maximum cache size for in-memory caches, expressed as the maximum
            number of cached records. This parameter has no effect when
            ``cache_type="disk"``.
        """
        super().__init__(
            cost_function,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
        )
        if isinstance(boundaries, dict):
            self._boundaries = boundaries
            self._is_kwargs = True
        elif self._is_interval(boundaries):
            self._boundaries = [cast(Boundary, boundaries)]
            self._is_kwargs = False
        else:
            self._boundaries = cast(Boundaries, boundaries)
            self._is_kwargs = False

        self._type = "continuous"
        self._configs = {}
        register_space(name=self._type, space_class=Continuous)

    def initialize_context(self, seed: int | None = None) -> None:
        if isinstance(self._boundaries, dict):
            self._encoded_boundaries = self._boundaries.copy()
        else:
            self._encoded_boundaries = {
                str(index): boundary for index, boundary in enumerate(self._boundaries)
            }

        self._variable_names = list(self._encoded_boundaries.keys())

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
        if self._is_kwargs:
            return dict(cast(tuple[tuple[str, float], ...], inputs))

        return list(cast(tuple[float, ...], inputs))
