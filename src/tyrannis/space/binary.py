import warnings
from collections.abc import Callable
from typing import TypeAlias, cast

import numpy as np

from tyrannis.core.space import SpaceBase

CacheKey: TypeAlias = tuple[bool, ...] | tuple[tuple[str, bool], ...]
BinaryInput: TypeAlias = list[bool] | dict[str, bool]


class Binary(SpaceBase):
    """Binary optimization search space."""

    _boundaries: int | list[str]
    _deterministic: bool
    _rng: np.random.Generator

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        bits: int | list[str],
        deterministic: bool = True,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing binary search space.

        Parameters
        ----------
        cost_function:
            User-defined cost function to be evaluated after decoding the
            solver representation.

        bits:
            Number of positional binary variables or names of keyword
            arguments.

            When an integer is provided, the cost function receives a list
            containing one boolean value for each bit.

            When a list of strings is provided, the cost function receives a
            dictionary using those strings as keys.

        deterministic:
            Select the binary decoding method.

            When True, angle modulation is used.

            When False, a sigmoid transfer function is used to determine the
            probability of each bit being one, followed by stochastic
            sampling.

        use_cache:
            Whether cost-function evaluations should be cached.

        cache_type:
            Cache strategy to use when caching is enabled.

        cache_size:
            Maximum cache size.
        """
        super().__init__(
            cost_function,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
        )

        if isinstance(bits, bool):
            raise TypeError("bits must be an int or a list of strings.")

        if isinstance(bits, int):
            if bits <= 0:
                raise ValueError("bits must be greater than 0.")
        elif isinstance(bits, list):
            if not all(isinstance(bit, str) for bit in bits):
                raise TypeError("bits must be a list of strings.")

            if len(set(bits)) != len(bits):
                raise ValueError("bits must contain unique strings.")
        else:
            raise TypeError("bits must be an int or a list of strings.")

        if not isinstance(deterministic, bool):
            raise TypeError("deterministic must be a bool.")

        self._boundaries = bits
        self._deterministic = deterministic

    def initialize_context(self, seed: int | None = None) -> None:
        """
        Initialize the execution context of the binary search space.

        Angle modulation uses the continuous domain [-2, 2]. The sigmoid
        transfer-function representation uses [-10, 10].
        """
        boundary = (-2.0, 2.0) if self._deterministic else (-10.0, 10.0)

        if isinstance(self._boundaries, int):
            self._encoded_boundaries = {
                str(index): boundary for index in range(self._boundaries)
            }
        else:
            self._encoded_boundaries = {bit: boundary for bit in self._boundaries}

        self._rng = np.random.default_rng(seed)

        if not self._deterministic and seed is None:
            warnings.warn(
                "A non-deterministic binary decoding method was selected "
                "without a seed. Each bit will be sampled according to the "
                "probability determined by the transfer function, so "
                "different evaluations may produce different bit sequences.",
                RuntimeWarning,
                stacklevel=2,
            )

    def decode(self, float_inputs: dict[str, float]) -> list[bool] | dict[str, bool]:
        """
        Decode the continuous solver representation into binary variables.
        """
        for key, value in float_inputs.items():
            if key not in self._encoded_boundaries:
                raise KeyError(f"Unknown binary-space variable: {key!r}.")

            lower, upper = self._encoded_boundaries[key]

            if not lower <= value <= upper:
                raise ValueError(
                    f"Value {value} for variable {key!r} is outside "
                    f"the boundaries ({lower}, {upper})."
                )

        if self._deterministic:
            decoded = self._decode_angle_modulation(float_inputs)
        else:
            decoded = self._decode_transfer_function(float_inputs)

        if self.is_kwargs:
            return decoded

        return [decoded[str(index)] for index in range(cast(int, self._boundaries))]

    def encode_cache(self, inputs: BinaryInput) -> CacheKey:
        """
        Encode the cost-function inputs into a canonical cache key.

        Boolean values are already hashable primitives, so no value
        transformation is required.
        """
        if isinstance(inputs, dict):
            return tuple(sorted(inputs.items()))

        return tuple(inputs)

    def decode_cache(self, inputs: CacheKey) -> BinaryInput:
        """
        Decode a canonical cache key into the cost-function representation.
        """
        if self.is_kwargs:
            return dict(cast(tuple[tuple[str, bool], ...], inputs))

        return list(cast(tuple[bool, ...], inputs))

    def _decode_angle_modulation(
        self, float_inputs: dict[str, float]
    ) -> dict[str, bool]:
        """
        Decode variables using angle modulation.

        The generating function is:

            g(x) = sin(2πx cos(2πx))

        Positive values represent one and non-positive values represent zero.
        """
        pi_2 = 2.0 * np.pi
        return {
            key: bool(np.sin(pi_2 * value * np.cos(pi_2 * value)) > 0.0)
            for key, value in float_inputs.items()
        }

    def _decode_transfer_function(
        self, float_inputs: dict[str, float]
    ) -> dict[str, bool]:
        """
        Decode variables using the sigmoid transfer function.

        The probability of a bit being one is given by:

            S(x) = 1 / (1 + exp(-x))

        Each bit is independently sampled using the RNG initialized by
        initialize_context.
        """
        return {
            key: bool(self._rng.random() < (1.0 / (1.0 + np.exp(-value))))
            for key, value in float_inputs.items()
        }

    @property
    def is_kwargs(self) -> bool:
        """Return whether the binary space uses keyword arguments."""
        return isinstance(self._boundaries, list)
