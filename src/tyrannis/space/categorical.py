from collections.abc import Callable, Iterable
from typing import Any, TypeAlias, cast

import numpy as np
from scipy.special import softmax
from scipy.stats import gumbel_r

from ..core.space import SpaceBase
from . import register_space

Choice: TypeAlias = Any
Choices: TypeAlias = list[Choice] | tuple[Choice, ...] | np.ndarray
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

        The choices can be supplied as a single collection, a collection of
        collections for positional inputs, or a dictionary of collections
        for named inputs.
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
        elif self._is_choice_collection(choices):
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
        default_bounds = {
            "one-hot": (-1.0, 1.0),
            "softmax": (-1.0, 1.0),
            "gumbel_softmax": (-1.0, 1.0),
            "scalar": (0.0, 1.0),
        }

        bounds = self._bounds

        if bounds is None:
            bounds = default_bounds[self._decoder]

        if isinstance(self._boundaries, dict):
            if self._decoder == "scalar":
                self._encoded_boundaries = {key: bounds for key in self._boundaries}
            else:
                self._encoded_boundaries = {
                    f"{key}-{str(choice)}": bounds
                    for key, choices in self._boundaries.items()
                    for choice in choices
                }
        else:
            if self._decoder == "scalar":
                self._encoded_boundaries = {
                    str(index): bounds for index in range(len(self._boundaries))
                }
            else:
                self._encoded_boundaries = {
                    f"{index}-{str(choice)}": bounds
                    for index, choices in enumerate(self._boundaries)
                    for choice in choices
                }

        self._rng = np.random.default_rng(seed)

    def decode(self, float_inputs: dict[str, float]) -> list[Any] | dict[str, Any]:
        self._check_input_bounds(float_inputs)

        decoders = {
            "one-hot": self._decode_one_hot,
            "softmax": self._decode_softmax,
            "gumbel_softmax": self._decode_gumbel_softmax,
            "scalar": self._decode_scalar,
        }

        decoder = decoders[self._decoder]
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
                    "Categorical choices must be hashable when caching is enabled."
                ) from error

        return key

    def decode_cache(self, inputs: CacheKey) -> CategoricalInput:
        if self._is_kwargs:
            return dict(cast(tuple[tuple[str, Any], ...], inputs))

        return list(cast(tuple[Any, ...], inputs))

    def _decode_one_hot(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}

        for key, choices in self._iter_choices():
            values = np.asarray(
                [float_inputs[f"{key}-{str(choice)}"] for choice in choices]
            )

            index = int(np.argmax(values))
            decoded[key] = choices[index]

        return decoded

    def _decode_softmax(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}

        for key, choices in self._iter_choices():
            values = np.asarray(
                [float_inputs[f"{key}-{str(choice)}"] for choice in choices]
            )

            probabilities = softmax(values)

            index = int(self._rng.choice(len(choices), p=probabilities))

            decoded[key] = choices[index]

        return decoded

    def _decode_gumbel_softmax(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}

        for key, choices in self._iter_choices():
            values = np.asarray(
                [float_inputs[f"{key}-{str(choice)}"] for choice in choices]
            )

            gumbel = gumbel_r.rvs(size=len(choices), random_state=self._rng)

            probabilities = softmax((values + gumbel) / self._gumbel_temperature)

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

    def _iter_choices(self) -> Iterable[tuple[str, Choices]]:
        if isinstance(self._boundaries, dict):
            return self._boundaries.items()

        return ((str(index), choices) for index, choices in enumerate(self._boundaries))

    @staticmethod
    def _is_choice_collection(value: object) -> bool:
        if isinstance(value, (str, bytes, dict)):
            return False

        if isinstance(value, np.ndarray):
            return value.ndim == 1

        if not isinstance(value, Iterable):
            return False

        values = list(value)

        if not values:
            return True

        return not all(Categorical._is_nested_collection(item) for item in values)

    @staticmethod
    def _is_nested_collection(value: object) -> bool:
        if isinstance(value, (str, bytes, dict)):
            return False

        if isinstance(value, np.ndarray):
            return value.ndim > 0

        return isinstance(value, Iterable)

    @staticmethod
    def _to_choices(value: object) -> Choices:
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError(
                    "Each categorical variable must be represented by "
                    "a one-dimensional collection of choices."
                )

            return value

        if isinstance(value, (str, bytes)):
            raise TypeError(
                "A categorical variable must contain a collection "
                "of choices, not a string."
            )

        try:
            choices = list(value)
        except TypeError as error:
            raise TypeError(
                "Each categorical variable must be represented by "
                "an iterable collection of choices."
            ) from error

        return choices
