import warnings
from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
from scipy.special import expit

from ..core.space import SpaceBase
from . import register_space

CacheKey: TypeAlias = tuple[bool, ...] | tuple[tuple[str, bool], ...]
BinaryInput: TypeAlias = list[bool] | dict[str, bool]
Limits: TypeAlias = tuple[float, float]


class Binary(SpaceBase):
    """Binary optimization search space."""

    _boundaries: int | list[str]
    _limits: Limits | None
    _decoder: str
    _params: dict[str, Any]
    _rng: np.random.Generator

    def __init__(
        self,
        bits: int | list[str] | None = None,
        cost_function: Callable[..., float] | None = None,
        decoder: str = "angle_modulation",
        bounds: Limits | None = None,
        params: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing binary search space.

        The binary search space represents optimization variables whose
        decoded values are Boolean values. The optimization algorithm
        operates on a continuous representation, which is converted into
        binary values according to the selected decoder.

        Binary variables can be defined either by their number, producing
        positional inputs, or by a list of names, producing keyword inputs.
        The selected decoder determines how the continuous representation is
        mapped to the resulting Boolean values.

        Parameters
        ----------
        bits:
            Number of binary variables or names of keyword arguments. An
            integer creates that many positional Boolean inputs, while a list
            of strings creates keyword inputs with the specified names. If
            ``None``, ``bits=1`` is assumed.

        cost_function:
            User-defined cost function to be evaluated after decoding the
            solver inputs. It receives the variables as Boolean values in
            their user-facing representation. It may be ``None`` during
            construction, but ``initialize_context`` requires a valid cost
            function.

        decoder:
            Method used to decode the continuous solver representation into
            binary variables. ``"angle_modulation"`` determines each bit
            through the sign of an angle-modulation function, while
            ``"s-shape"`` uses a sigmoid transfer function to obtain the
            probability of each bit being ``True`` and samples the resulting
            Boolean value.

        bounds:
            Lower and upper limits of the continuous encoded representation.
            When ``None``, decoder-specific default bounds are used:
            ``(-2.0, 2.0)`` for ``"angle_modulation"`` and
            ``(-6.0, 6.0)`` for ``"s-shape"``.

        params:
            Parameters used by the selected decoding method. Parameters are
            provided as a dictionary where each key is the name of a
            parameter and its value is the corresponding parameter value.
            Supported parameters are:

            ``alpha``:
                Controls the inclination of the logistic transfer function
                used by the ``"s-shape"`` decoder. Higher values produce a
                steeper transition around the center of the logistic
                function, while lower values produce a smoother transition
                over a wider range of continuous inputs. Defaults to
                ``1.0``.

        use_cache:
            Whether cost-function evaluations should be cached. When
            ``False``, no cache is created or used.

        cache_type:
            Cache strategy to use when caching is enabled. Supported
            strategies are ``"lru"``, ``"lfu"``, ``"fifo"``, ``"rr"``, and
            ``"disk"``.

        cache_size:
            Maximum cache size for in-memory caches, expressed as the maximum
            number of cached records. This parameter has no effect when
            ``cache_type="disk"``.

        References
        ----------
        Kennedy, J., & Eberhart, R. C. (1997). A discrete binary version of
        the particle swarm algorithm. Proceedings of the 1997 IEEE
        International Conference on Systems, Man, and Cybernetics,
        4104-4108. https://doi.org/10.1109/ICSMC.1997.637339

        Pampara, G., Franken, N., & Engelbrecht, A. P. (2005). Combining
        particle swarm optimisation with angle modulation to solve binary
        problems. Proceedings of the IEEE Congress on Evolutionary
        Computation, 89-96. https://doi.org/10.1109/CEC.2005.1554671
        """
        super().__init__(
            cost_function,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
        )

        if bits is None:
            bits = 1  # No arguments, to use with mixed inputs

        elif isinstance(bits, int):
            if bits <= 0:
                raise ValueError("bits must be greater than 0.")

        elif isinstance(bits, list):
            if not all(isinstance(bit, str) for bit in bits):
                raise TypeError("bits must be a list of strings.")

            if len(set(bits)) != len(bits):
                raise ValueError("bits must contain unique strings.")

        else:
            raise TypeError("bits must be an int or a list of strings.")

        if decoder not in {"angle_modulation", "s-shape"}:
            raise ValueError(
                f"Invalid decoder: {decoder!r}. "
                "Expected one of: 'angle_modulation', 's-shape'."
            )

        if bounds is not None:
            if not isinstance(bounds, tuple) or len(bounds) != 2:
                raise TypeError("bounds must be a tuple with two numeric values.")

            if not all(
                isinstance(value, (float, int, np.floating, np.integer))
                for value in bounds
            ):
                raise TypeError("bounds values must be numeric.")

            if bounds[0] >= bounds[1]:
                raise ValueError(
                    "The lower bound must be smaller than the upper bound."
                )

        self._boundaries = bits
        self._decoder = decoder
        self._limits = bounds
        self._params = {} if params is None else params
        self._params = dict(sorted(self._params.items()))
        self._is_kwargs = isinstance(bits, list)

        self._type = "binary"
        self._configs = {"decoder": decoder, "bounds": bounds, "params": self._params}
        register_space(name=self._type, space_class=Binary)

    def initialize_context(self, seed: int | None = None) -> None:
        default_bounds = {"angle_modulation": (-2.0, 2.0), "s-shape": (-6.0, 6.0)}

        if self._limits is not None:
            boundary = self._limits
        else:
            boundary = default_bounds.get(self._decoder)

        if boundary is None:
            raise RuntimeError("boundary cannot be None")

        if isinstance(self._boundaries, int):
            self._encoded_boundaries = {
                str(index): boundary for index in range(self._boundaries)
            }
        else:
            self._encoded_boundaries = {bit: boundary for bit in self._boundaries}

        self._variable_names = list(self._encoded_boundaries.keys())
        self._rng = np.random.default_rng(seed)

        if self._decoder in {"s-shape"} and (seed is None):
            warnings.warn(
                "A stochastic decoding method was selected without "
                "a seed. Different evaluations may produce different bit "
                "sequences.",
                RuntimeWarning,
                stacklevel=2,
            )

    def decode(self, float_inputs: dict[str, float]) -> list[bool] | dict[str, bool]:
        self._check_input_bounds(float_inputs)

        decoders = {
            "angle_modulation": self._decode_angle_modulation,
            "s-shape": self._decode_s_shape,
        }

        decoder = decoders.get(self._decoder)

        if decoder is None:
            raise RuntimeError(f"Decoder {self._decoder!r} is not available.")

        decoded = decoder(float_inputs)

        if self._is_kwargs:
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
        if self._is_kwargs:
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

    def _decode_s_shape(self, float_inputs: dict[str, float]) -> dict[str, bool]:
        """
        Decode variables using the S-shaped transfer function.

        The transfer function is the sigmoid:

            S(x) = 1 / (1 + exp(-x))

        The resulting value represents the probability of the bit being one.
        """
        alpha = self._params.get("alpha", 1.0)
        return {
            key: bool(self._rng.random() < expit(alpha * value))
            for key, value in float_inputs.items()
        }
