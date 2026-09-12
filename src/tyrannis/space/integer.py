from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np
import scipy as sp

from tyrannis.core.space import SpaceBase

Boundary: TypeAlias = tuple[int, int]
Boundaries: TypeAlias = list[Boundary] | dict[str, Boundary]
CacheKey: TypeAlias = tuple[int, ...] | tuple[tuple[str, int], ...]
IntegerInput: TypeAlias = list[int] | dict[str, int]


class Integer(SpaceBase):
    """Integer optimization search space."""

    _boundaries: Boundaries
    _decoded_boundaries: list[tuple[float, float]] | dict[str, tuple[float, float]]
    _decoder: str
    _rng: np.random.Generator

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        boundaries: Boundaries,
        decoder: str = "round",
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing integer search space.
        """
        super().__init__(
            cost_function,
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

    def initialize_context(self, seed: int | None = None) -> None:
        """
        Initialize the execution context of the integer search space.
        """
        if isinstance(self._boundaries, dict):
            self._decoded_boundaries = {
                key: (float(lower), float(upper))
                for key, (lower, upper) in self._boundaries.items()
            }
        else:
            self._decoded_boundaries = [
                (float(lower), float(upper)) for lower, upper in self._boundaries
            ]

        if self._decoder in {"scaling", "transfer_function"}:
            boundary = (0.0, 1.0)

            if isinstance(self._boundaries, dict):
                self._encoded_boundaries = {key: boundary for key in self._boundaries}
            else:
                self._encoded_boundaries = {
                    str(index): boundary for index in range(len(self._boundaries))
                }
        elif isinstance(self._boundaries, dict):
            self._encoded_boundaries = self._boundaries.copy()
        else:
            self._encoded_boundaries = {
                str(index): boundary for index, boundary in enumerate(self._boundaries)
            }

        self._rng = np.random.default_rng(seed)

    def decode(self, float_inputs: dict[str, float]) -> list[int] | dict[str, int]:
        """
        Decode the continuous solver representation into integer variables.
        """
        for key, value in float_inputs.items():
            if key not in self._encoded_boundaries:
                raise KeyError(f"Unknown integer-space variable: {key!r}.")

            lower, upper = self._encoded_boundaries[key]

            if not lower <= value <= upper:
                raise ValueError(
                    f"Value {value} for variable {key!r} is outside "
                    f"the boundaries ({lower}, {upper})."
                )

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
        """
        Encode the cost-function inputs into a canonical cache key.
        """
        if isinstance(inputs, dict):
            return tuple(sorted(inputs.items()))

        return tuple(inputs)

    def decode_cache(self, inputs: CacheKey) -> IntegerInput:
        """
        Decode a canonical cache key into the cost-function representation.
        """
        if self.is_kwargs:
            return dict(cast(tuple[tuple[str, int], ...], inputs))

        return list(cast(tuple[int, ...], inputs))

    def _decode_round(self, float_inputs: dict[str, float]) -> dict[str, int]:
        """
        Decode variables using nearest-integer rounding and clipping.
        """
        return {
            key: int(np.clip(np.round(value), *self._decoded_boundaries[key]))
            for key, value in float_inputs.items()
        }

    def _decode_scaling(self, float_inputs: dict[str, float]) -> dict[str, int]:
        """
        Decode normalized variables by scaling them to the integer bounds.
        """
        return {
            key: int(np.round(lower + value * (upper - lower)))
            for key, value in float_inputs.items()
            for lower, upper in [self._decoded_boundaries[key]]
        }

    def _decode_stochastic_round(
        self, float_inputs: dict[str, float]
    ) -> dict[str, int]:
        """
        Decode variables using stochastic rounding.
        """
        return {
            key: int(np.floor(value) + (self._rng.random() < value % 1))
            for key, value in float_inputs.items()
        }

    def _decode_transfer_function(
        self, float_inputs: dict[str, float]
    ) -> dict[str, int]:
        """
        Decode normalized variables using a sigmoid transfer function.
        """
        return {
            key: int(np.floor(lower + sp.special.expit(value) * (upper - lower + 1)))
            for key, value in float_inputs.items()
            for lower, upper in [self._decoded_boundaries[key]]
        }

    @property
    def is_kwargs(self) -> bool:
        """Return whether the integer space uses keyword arguments."""
        return isinstance(self._boundaries, dict)
