import warnings
from collections.abc import Callable
from typing import Any, TypeAlias, cast

import numpy as np
from scipy.special import expit

from ..core.space import SpaceBase
from . import register_space

Boundary: TypeAlias = tuple[int, int]
Boundaries: TypeAlias = list[Boundary] | dict[str, Boundary]
CacheKey: TypeAlias = tuple[int, ...] | tuple[tuple[str, int], ...]
IntegerInput: TypeAlias = list[int] | dict[str, int]


class Integer(SpaceBase):
    """Integer optimization search space."""

    _boundaries: Boundaries
    _decoded_boundaries: dict[str, tuple[float, float]]
    _decoder: str
    _params: dict[str, Any]
    _rng: np.random.Generator

    def __init__(
        self,
        boundaries: Boundaries,
        cost_function: Callable[..., float] | None = None,
        decoder: str = "round",
        custom_bounds: tuple[float, float] | None = None,
        params: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing integer search space.

        The integer search space represents optimization variables whose
        decoded values are restricted to integers. The optimization algorithm
        operates on a continuous representation, which is converted into
        integer values according to the selected decoder.

        Boundaries define the valid integer range of each variable. They can
        be provided either as a list of intervals for positional inputs or as
        a dictionary associating each variable with its interval.

        Parameters
        ----------
        boundaries:
            Integer search-space boundaries. A list contains one
            ``(lower, upper)`` tuple for each positional input. A dictionary
            maps each input name to its ``(lower, upper)`` tuple. The bounds
            define the valid integer range of each variable.

        cost_function:
            User-defined cost function to be evaluated after decoding the
            solver inputs. It receives the variables as integers in their
            user-facing representation. It may be ``None`` during
            construction, but ``initialize_context`` requires a valid cost
            function.

        decoder:
            Method used to convert the continuous solver representation into
            integer values. ``"round"`` rounds the continuous value to the
            nearest integer. ``"scaling"`` maps the encoded value from
            ``[0.0, 1.0]`` to the corresponding integer boundary range.
            ``"stochastic_round"`` performs probabilistic rounding based on
            the fractional part of the value. ``"transfer_function"`` uses
            a logistic transfer function to map the encoded value to the
            integer boundary range.

        custom_bounds:
            Continuous interval used by the optimization algorithm when
            ``decoder="transfer_function"``. When ``None``, ``(-6.0, 6.0)``
            is used. This parameter has no effect for the other decoders.

        params:
            Parameters used by the selected decoding method. Parameters are
            provided as a dictionary where each key is the name of a
            parameter and its value is the corresponding parameter value.
            Supported parameters are:

            ``alpha``:
                Controls the inclination of the logistic transfer function
                used by the ``"transfer_function"`` decoder. Higher values
                produce a steeper transition around the center of the
                logistic function, while lower values produce a smoother
                transition over a wider portion of the continuous search
                interval. Defaults to ``1.0``.

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
        Hereford, J. M., & Gerlach, H. (2008). Integer-valued Particle Swarm
        Optimization applied to Sudoku puzzles. Proceedings of the 2008 IEEE
        Swarm Intelligence Symposium, 1-7.
        https://doi.org/10.1109/SIS.2008.4668293

        Pampara, G., Franken, N., & Engelbrecht, A. P. (2009). Novel
        Mutative Particle Swarm Optimization Algorithm for Discrete
        Optimization. Proceedings of the 2009 IEEE Congress on Evolutionary
        Computation.

        Connolly, M. P., Higham, N. J., & Mary, T. (2021). Stochastic rounding
        and its probabilistic backward error analysis. SIAM Journal on
        Scientific Computing, 43(5), C259-C281.
        https://doi.org/10.1137/20M1334796

        Chen, C.-H. (2010). Hierarchical Swarm Model: A New Approach to
        Optimization. Discrete Dynamics in Nature and Society, 2010,
        379649. https://doi.org/10.1155/2010/379649
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

        if isinstance(boundaries, dict):
            self._boundaries = boundaries
            self._is_kwargs = True
        elif self._is_interval(boundaries):
            self._boundaries = [cast(Boundary, boundaries)]
            self._is_kwargs = False
        else:
            self._boundaries = cast(Boundaries, boundaries)
            self._is_kwargs = False

        self._decoder = decoder
        self._custom_bounds = custom_bounds
        self._params = {} if params is None else params
        self._params = dict(sorted(self._params.items()))
        self._type = "integer"
        self._configs = {
            "decoder": decoder,
            "custom_bounds": custom_bounds,
            "params": self._params,
        }
        register_space(name=self._type, space_class=Integer)

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

        self._variable_names = list(self._encoded_boundaries.keys())
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

        if self._is_kwargs:
            return decoded

        return [decoded[str(index)] for index in range(len(self._boundaries))]

    def encode_cache(self, inputs: IntegerInput) -> CacheKey:
        if isinstance(inputs, dict):
            return tuple(sorted(inputs.items()))

        return tuple(inputs)

    def decode_cache(self, inputs: CacheKey) -> IntegerInput:
        if self._is_kwargs:
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
        alpha = self._params.get("alpha", 1.0)
        decoded = {}
        for key, value in float_inputs.items():
            lower, upper = self._decoded_boundaries[key]
            scaled = lower + (expit(alpha * value) * (upper - lower))
            decoded[key] = int(np.round(scaled))

        return decoded
