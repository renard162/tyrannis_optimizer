import warnings
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
from scipy.special import expit

from tyrannis.core.space import SpaceBase

Boundary: TypeAlias = tuple[int, int]
Boundaries: TypeAlias = list[Boundary] | dict[str, Boundary]
CacheKey: TypeAlias = tuple[int, ...] | tuple[tuple[str, int], ...]
IntegerInput: TypeAlias = list[int] | dict[str, int]


class Integer(SpaceBase):
    """Integer optimization search space."""

    _boundaries: Boundaries
    _decoded_boundaries: dict[str, tuple[float, float]]
    _decoder: str
    _rng: np.random.Generator

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        boundaries: Boundaries,
        decoder: str = "round",
        custom_bounds: tuple[float, float] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing integer search space.

        Parameters
        ----------
        cost_function:
            User-defined cost function to be evaluated after decoding.
        boundaries:
            Integer search-space boundaries. A list contains one
            ``(lower, upper)`` tuple for each positional input. A dictionary maps
            each input name to its ``(lower, upper)`` tuple.
        decoder:
            Decoder used to convert the continuous solver representation into
            integer values. Supported decoders are ``"round"``, ``"scaling"``,
            ``"stochastic_round"``, and ``"transfer_function"``.
        custom_bounds:
            Continuous interval considered during optimization when
            ``decoder="transfer_function"``. If ``None``, the interval
            ``(-6.0, 6.0)`` is used. This default follows the interval commonly
            used with sigmoid transfer functions in meta-heuristic optimization.
        use_cache:
            Whether cost-function evaluations should be cached.
        cache_type:
            Cache strategy to use when caching is enabled.
        cache_size:
            Maximum cache size.
        """
        super().__init__(
            cost_function=cost_function,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
        )

        if decoder not in {"round", "scaling", "stochastic_round", "transfer_function"}:
            raise ValueError(
                f"Invalid decoder: {decoder!r}. "
                "Expected one of: 'round', 'scaling', "
                "'stochastic_round', 'transfer_function'."
            )

        self._boundaries = boundaries
        self._decoder = decoder
        self._custom_bounds = custom_bounds

    def initialize_context(self, seed: int | None = None) -> None:
        if isinstance(self._boundaries, dict):
            self._decoded_boundaries = {
                key: (float(lower), float(upper))
                for key, (lower, upper) in self._boundaries.items()
            }
        else:
            self._decoded_boundaries = {
                str(index): (float(lower), float(upper))
                for index, (lower, upper) in enumerate(self._boundaries)
            }

        transfer_function_bounds = (
            (-6.0, 6.0) if (self._custom_bounds is None) else self._custom_bounds
        )

        default_boundaries = {
            "scaling": (0.0, 1.0),
            "transfer_function": transfer_function_bounds,
        }
        boundaries = default_boundaries.get(self._decoder)

        if boundaries is not None:
            self._encoded_boundaries = {
                key: boundaries for key in self._decoded_boundaries
            }
        else:
            self._encoded_boundaries = self._decoded_boundaries.copy()

        self._rng = np.random.default_rng(seed)

        if self._decoder in {"stochastic_round"} and (seed is None):
            warnings.warn(
                "A stochastic decoding method was selected without a seed. "
                "Different evaluations may produce different results",
                RuntimeWarning,
                stacklevel=2,
            )

    def decode(self, float_inputs: dict[str, float]) -> list[int] | dict[str, int]:
        self._check_input_bounds(float_inputs)

        decoders = {
            "round": self._decode_round,
            "scaling": self._decode_scaling,
            "stochastic_round": self._decode_stochastic_round,
            "transfer_function": self._decode_transfer_function,
        }

        decoder = decoders.get(self._decoder)

        if decoder is None:
            raise RuntimeError(f"Decoder {self._decoder!r} is not available.")

        decoded = decoder(float_inputs)

        if self.is_kwargs:
            return decoded

        return [decoded[str(index)] for index in range(len(self._boundaries))]

    def encode_cache(self, inputs: IntegerInput) -> CacheKey:
        if isinstance(inputs, dict):
            return tuple(sorted(inputs.items()))

        return tuple(inputs)

    def decode_cache(self, inputs: CacheKey) -> IntegerInput:
        if self.is_kwargs:
            return dict(cast(tuple[tuple[str, int], ...], inputs))

        return list(cast(tuple[int, ...], inputs))

    def _decode_round(self, float_inputs: dict[str, float]) -> dict[str, int]:
        return {key: int(np.round(value)) for key, value in float_inputs.items()}

    def _decode_stochastic_round(
        self, float_inputs: dict[str, float]
    ) -> dict[str, int]:
        return {
            key: int(np.floor(value) + (self._rng.random() < (value % 1)))
            for key, value in float_inputs.items()
        }

    def _decode_scaling(self, float_inputs: dict[str, float]) -> dict[str, int]:
        decoded = {}
        for key, value in float_inputs.items():
            lower, upper = self._decoded_boundaries[key]
            scaled = lower + (value * (upper - lower))
            decoded[key] = int(np.round(scaled))

        return decoded

    def _decode_transfer_function(
        self, float_inputs: dict[str, float]
    ) -> dict[str, int]:
        decoded = {}
        for key, value in float_inputs.items():
            lower, upper = self._decoded_boundaries[key]
            scaled = lower + (expit(value) * (upper - lower))
            decoded[key] = int(np.round(scaled))

        return decoded
