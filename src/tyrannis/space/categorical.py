import warnings
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias, cast

import numpy as np
from scipy.special import softmax
from scipy.stats import gumbel_r

from ..core.space import SpaceBase
from . import register_space

Choice: TypeAlias = Any
Choices: TypeAlias = Iterable[Choice]
Boundaries: TypeAlias = list[Choices] | dict[str, Choices]
CategoricalInput: TypeAlias = list[Any] | dict[str, Any]
CacheKey: TypeAlias = tuple[Any, ...] | tuple[tuple[str, Any], ...]


class Categorical(SpaceBase):
    """Categorical optimization search space."""

    _boundaries: Boundaries
    _bounds: tuple[float, float] | None
    _decoder: str
    _gumbel_temperature: float
    _rng: np.random.Generator

    def __init__(
        self,
        choices: Choices | Boundaries,
        cost_function: Callable[..., float] | None = None,
        decoder: str = "one-hot",
        bounds: tuple[float, float] | None = None,
        gumbel_temperature: float | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing categorical search space.

        The categorical choices can be provided as a single collection, a
        collection of collections, or a dictionary associating each variable
        with its collection of possible choices.

        When a single collection is provided, it represents one categorical
        input. This form is also useful when the space is used as a component
        of a mixed search space.

        When a collection of collections is provided, each inner collection
        represents one positional input of the cost function.

        When a dictionary is provided, each key identifies one input of the
        cost function and its associated collection contains the possible
        choices for that input.

        The concrete collection type is not relevant to the search-space
        interface. Lists, tuples, NumPy arrays, and other iterable
        collections can be used to represent the available choices.

        Parameters
        ----------
        choices:
            Categories available to each optimization variable. A single
            collection represents one categorical input. A collection of
            collections represents multiple positional inputs. A dictionary
            maps each input name to its collection of possible choices.

        cost_function:
            User-defined cost function to be evaluated after decoding the
            solver inputs. It may be ``None`` during construction, but
            ``initialize_context`` requires a valid cost function.

        decoder:
            Method used to decode the continuous solver representation.
            ``"one-hot"`` selects the category with the highest encoded
            value. ``"softmax"`` interprets the encoded values as logits,
            converts them into probabilities with softmax, and samples a
            category according to those probabilities. ``"gumbel_softmax"``
            adds Gumbel noise to the encoded values, applies the configured
            temperature and softmax, and samples a category from the
            resulting probability distribution. ``"scalar"`` represents
            each categorical variable with a single continuous value.

        bounds:
            Continuous search interval exposed to the optimization
            algorithm. When ``None``, decoder-specific default bounds are
            used. The ``"scalar"`` decoder defaults to ``(0.0, 1.0)`` while
            the remaining decoders default to ``(-1.0, 1.0)``.

        gumbel_temperature:
            Temperature factor used by the ``"gumbel_softmax"`` decoder to
            control the concentration of the categorical probability
            distribution. Lower temperatures produce increasingly
            concentrated distributions, making the decoder more likely to
            select the category with the highest perturbed value, while
            higher temperatures produce increasingly uniform distributions,
            increasing exploration among categories.

            The mathematical domain is ``(0, +inf)``. In practice, the
            recommended range is ``[0.1, 10.0]``, as values outside this
            range provide increasingly limited practical benefit due to
            excessive concentration or near-uniformity of the resulting
            distribution.

            As a practical guideline, temperatures can be interpreted as:
            values below ``0.5`` are low and favor exploitation, values from
            ``0.5`` to ``2.0`` are moderate and provide a balance between
            exploration and exploitation, and values above ``2.0`` are high
            and favor exploration. A temperature of ``1.0`` is used by
            default.

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

        if isinstance(choices, dict):
            self._boundaries = {
                key: self._to_choices(value) for key, value in choices.items()
            }
            self._is_kwargs = True
        elif self._is_choices_collection(choices):
            self._boundaries = [self._to_choices(choices)]
            self._is_kwargs = False
        else:
            self._boundaries = [self._to_choices(value) for value in choices]
            self._is_kwargs = False

        if not self._boundaries:
            raise ValueError("choices cannot be empty.")

        if isinstance(self._boundaries, dict):
            if any(not choices for choices in self._boundaries.values()):
                raise ValueError(
                    "Each categorical variable must have at least one choice."
                )
        elif any(not choices for choices in self._boundaries):
            raise ValueError("Each categorical variable must have at least one choice.")

        if decoder not in {"one-hot", "softmax", "gumbel_softmax", "scalar"}:
            raise ValueError(
                f"Invalid decoder: {decoder!r}. "
                "Expected one of: 'one-hot', 'softmax', "
                "'gumbel_softmax', 'scalar'."
            )

        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError("bounds must contain exactly two values.")

            if bounds[0] >= bounds[1]:
                raise ValueError(
                    "The lower bound must be smaller than the upper bound."
                )

        if gumbel_temperature is None:
            gumbel_temperature = 1.0

        if gumbel_temperature <= 0:
            raise ValueError("gumbel_temperature must be greater than zero.")

        self._bounds = bounds
        self._decoder = decoder
        self._gumbel_temperature = float(gumbel_temperature)

        self._type = "categorical"
        self._configs = {
            "decoder": decoder,
            "bounds": bounds,
            "gumbel_temperature": gumbel_temperature,
        }

        register_space(name=self._type, space_class=Categorical)

    def initialize_context(self, seed: int | None = None) -> None:
        bounds = self._bounds

        if self._bounds is None:
            default_bounds = {
                "one-hot": (-1.0, 1.0),
                "softmax": (-1.0, 1.0),
                "gumbel_softmax": (-1.0, 1.0),
                "scalar": (0.0, 1.0),
            }
            bounds = default_bounds.get(self._decoder)

        if bounds is None:
            raise RuntimeError("bounds cannot be None")

        if isinstance(self._boundaries, dict):
            if self._decoder == "scalar":
                self._encoded_boundaries = {key: bounds for key in self._boundaries}
            else:
                self._encoded_boundaries = {}
                for key, choices in self._boundaries.items():
                    for choice in choices:
                        choice_str = str(choice)
                        encoded_key = f"{key}-{choice_str}"
                        self._encoded_boundaries[encoded_key] = bounds
        else:
            if self._decoder == "scalar":
                self._encoded_boundaries = {
                    str(index): bounds for index in range(len(self._boundaries))
                }
            else:
                self._encoded_boundaries = {}
                for index, choices in enumerate(self._boundaries):
                    for choice in choices:
                        choice_str = str(choice)
                        encoded_key = f"{index}-{choice_str}"
                        self._encoded_boundaries[encoded_key] = bounds

        self._rng = np.random.default_rng(seed)

        if self._decoder in {"softmax", "gumbel_softmax"} and seed is None:
            warnings.warn(
                "A stochastic decoding method was selected without a seed. "
                "Different evaluations may produce different results",
                RuntimeWarning,
                stacklevel=2,
            )

    def decode(self, float_inputs: dict[str, float]) -> list[Any] | dict[str, Any]:
        self._check_input_bounds(float_inputs)

        decoders = {
            "one-hot": self._decode_one_hot,
            "softmax": self._decode_softmax,
            "gumbel_softmax": self._decode_gumbel_softmax,
            "scalar": self._decode_scalar,
        }

        decoder = decoders.get(self._decoder)

        if decoder is None:
            raise RuntimeError(f"Decoder {self._decoder!r} is not available.")

        decoded = decoder(float_inputs)

        if self._is_kwargs:
            return decoded

        return [decoded[str(index)] for index in range(len(self._boundaries))]

    def encode_cache(self, inputs: CategoricalInput) -> CacheKey:
        if isinstance(inputs, dict):
            key = tuple(sorted(inputs.items()))
        else:
            key = tuple(inputs)

        if self._use_cache:
            try:
                hash(key)
            except TypeError as error:
                raise TypeError(
                    "Categorical values must be hashable when caching is enabled."
                ) from error

        return key

    def decode_cache(self, inputs: CacheKey) -> CategoricalInput:
        if self._is_kwargs:
            return dict(cast(tuple[tuple[str, Any], ...], inputs))

        return list(cast(tuple[Any, ...], inputs))

    def _decode_one_hot(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}

        for key, choices in self._iter_choices():
            values = []

            for choice in choices:
                choice_str = str(choice)
                values.append(float_inputs[f"{key}-{choice_str}"])

            index = int(np.argmax(values))
            decoded[key] = choices[index]

        return decoded

    def _decode_softmax(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}

        for key, choices in self._iter_choices():
            values = []

            for choice in choices:
                choice_str = str(choice)
                values.append(float_inputs[f"{key}-{choice_str}"])

            probabilities = softmax(values)

            index = int(self._rng.choice(len(choices), p=probabilities))

            decoded[key] = choices[index]

        return decoded

    def _decode_gumbel_softmax(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}
        for key, choices in self._iter_choices():
            values = []
            for choice in choices:
                choice_str = str(choice)
                values.append(float_inputs[f"{key}-{choice_str}"])

            gumbel_gi = gumbel_r.rvs(size=len(choices), random_state=self._rng)
            gumbel = (np.asarray(values) + gumbel_gi) / self._gumbel_temperature
            probabilities = softmax(gumbel)
            index = int(self._rng.choice(len(choices), p=probabilities))
            decoded[key] = choices[index]

        return decoded

    def _decode_scalar(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}

        if self._bounds is None:
            lower, upper = 0.0, 1.0
        else:
            lower, upper = self._bounds

        for key, choices in self._iter_choices():
            value = float_inputs[key]
            normalized = (value - lower) / (upper - lower)
            index = min(int(normalized * len(choices)), len(choices) - 1)
            decoded[key] = choices[index]

        return decoded

    def _iter_choices(self) -> list[tuple[str, list[Any]]]:
        if isinstance(self._boundaries, dict):
            return [(key, list(choices)) for key, choices in self._boundaries.items()]

        return [
            (str(index), list(choices))
            for index, choices in enumerate(self._boundaries)
        ]

    @staticmethod
    def _is_choices_collection(value: object) -> bool:
        if isinstance(value, (str, bytes)):
            return False

        if isinstance(value, np.ndarray):
            return value.ndim == 1

        if not isinstance(value, Iterable):
            return False

        iterator = iter(value)

        try:
            first = next(iterator)
        except StopIteration:
            return True

        return not Categorical._is_nested_collection(first)

    @staticmethod
    def _is_nested_collection(value: object) -> bool:
        if isinstance(value, (str, bytes)):
            return False

        if isinstance(value, np.ndarray):
            return value.ndim > 0

        return isinstance(value, Iterable)

    @staticmethod
    def _to_choices(value: object) -> list[Any]:
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError(
                    "Each categorical variable must be represented by "
                    "a one-dimensional collection of choices."
                )

            return value.tolist()

        if not isinstance(value, Iterable):
            raise TypeError(
                "Each categorical variable must be represented by "
                "an iterable collection of choices."
            )

        return list(value)
