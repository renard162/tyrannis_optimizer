import json
import warnings
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias, cast

import numpy as np
from scipy.special import expit, ndtr, softmax

from ..core.space import SpaceBase
from . import register_space

Choice: TypeAlias = Any
Choices: TypeAlias = Iterable[Choice]
Boundaries: TypeAlias = list[Choices] | dict[str, Choices]
Positions: TypeAlias = Iterable[float]
PositionBoundaries: TypeAlias = list[Positions] | dict[str, Positions]
OrdinalInput: TypeAlias = list[Any] | dict[str, Any]
CacheKey: TypeAlias = str


class Ordinal(SpaceBase):
    """Ordinal optimization search space."""

    _boundaries: Boundaries
    _positions: list[list[float]] | dict[str, list[float]]
    _positions_config: list[float] | list[list[float]] | dict[str, list[float]] | None
    _decoder: str
    _params: dict[str, Any]
    _rng: np.random.Generator

    def __init__(
        self,
        choices: Choices | Boundaries,
        cost_function: Callable[..., float] | None = None,
        positions: Positions | PositionBoundaries | None = None,
        decoder: str = "rank",
        params: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing ordinal search space.

        The ordinal choices can be provided as a single collection, a
        collection of collections, or a dictionary associating each variable
        with its collection of ordered choices.

        When a single collection is provided, it represents one ordinal
        input. This form is also useful when the space is used as a component
        of a mixed search space.

        When a collection of collections is provided, each inner collection
        represents one positional input of the cost function, with the
        elements defining the ordered choices for that input.

        When a dictionary is provided, each key identifies one input of the
        cost function and its associated collection contains the ordered
        choices available to that variable.

        The concrete collection type is not relevant to the search-space
        interface. Lists, tuples, NumPy arrays, and other iterable
        collections can be used to represent the available choices.

        Optional positions can be associated with the choices to define their
        relative distances in the continuous search space. When omitted,
        equally spaced positions are used.

        Parameters
        ----------
        choices:
            Ordered elements available to each ordinal variable. A single
            collection represents one ordinal input. A collection of
            collections represents multiple positional ordinal inputs.
            A dictionary maps each input name to the ordered collection of
            available values.

        cost_function:
            User-defined cost function evaluated after decoding the solver
            inputs into their user-facing representation. This argument is
            required when the space is used independently. It does not need
            to be provided when the space is used as a component of a
            ``Mixed`` space, because in that case the cost function is
            provided to the ``Mixed`` space itself.

        positions:
            Continuous positions associated one-to-one with the ordinal
            choices. The representation must match ``choices``: a single
            collection of positions for one ordinal input, a collection of
            position collections for multiple positional inputs, or a
            dictionary with exactly the same keys as ``choices`` for keyword
            inputs. Each position must be finite and positions within each
            variable must be strictly increasing. When ``None``, equally
            spaced positions ``0.0, 1.0, ..., n - 1`` are generated
            automatically.

        decoder:
            Method used to decode the continuous solver representation into
            ordinal values. Supported methods include:

            - ``"rank"``: Selects the ordinal choice whose continuous position
            is nearest to the solver value. This provides deterministic
            decoding while preserving the ordering and spacing represented
            by the configured positions. Ties are resolved in favor of the
            lower ordinal level. (default)
            - ``"nearest-stochastic"``: Stochastically selects between the two
            ordinal positions adjacent to the solver value. Selection
            probabilities are proportional to the relative distance from
            the solver value to each adjacent position, following the
            stochastic-rounding principle. Values outside the first and
            last positions decode to the corresponding extreme level.
            - ``"cumulative-logit"``: Stochastically samples an ordinal level
            using a cumulative logistic model. Midpoints between adjacent
            positions define the ordered cut points, and ``temperature``
            controls the scale of the logistic distribution.
            - ``"cumulative-probit"``: Stochastically samples an ordinal level
            using a cumulative normal model. Midpoints between adjacent
            positions define the ordered cut points, and ``temperature``
            controls the scale of the normal distribution.
            - ``"distance-softmax"``: Stochastically samples among all ordinal
            levels using a softmax distribution over the negative absolute
            distance between the solver value and each configured position.
            ``temperature`` controls how strongly the distribution favors
            nearby positions.

        params:
            Parameters used by the selected decoding method. Parameters are
            provided as a dictionary where each key is the name of a parameter
            and its value is the corresponding parameter value. Supported
            parameters are:

            ``temperature``:
                Controls the stochasticity of the ``"cumulative-logit"``,
                ``"cumulative-probit"``, and ``"distance-softmax"`` decoders.
                Lower values concentrate probability more strongly around
                nearby ordinal levels, while higher values produce broader
                probability distributions. The value must be finite and
                greater than zero. Defaults to ``1.0``.

        use_cache:
            Whether cost-function evaluations should be cached. When
            ``False``, no cache is created or used.

        cache_type:
            Cache strategy to use when caching is enabled. Supported strategies are:

            - ``"lru"``: Least Recently Used cache. (default)
            - ``"disk"``: Temporary disk-backed runtime cache. Its contents are
            local to the current runtime and are not preserved through serialization.
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

        References
        ----------
        Bradley, R. A., Katti, S. K., & Coons, I. J. (1962). Optimal Scaling
        for Ordered Categories. Psychometrika, 27(4), 355-374.
        https://doi.org/10.1007/BF02289644

        Croci, M., Fasi, M., Higham, N. J., Mary, T., & Mikaitis, M. (2022).
        Stochastic Rounding: Implementation, Error Analysis and Applications.
        Royal Society Open Science, 9(3), 211631.
        https://doi.org/10.1098/rsos.211631

        McCullagh, P. (1980). Regression Models for Ordinal Data. Journal of
        the Royal Statistical Society: Series B (Methodological), 42(2),
        109-127. https://doi.org/10.1111/j.2517-6161.1980.tb01109.x

        Aitchison, J., & Silvey, S. D. (1957). The Generalization of Probit
        Analysis to the Case of Multiple Responses. Biometrika, 44(1-2),
        131-140. https://doi.org/10.1093/biomet/44.1-2.131

        Snell, J., Swersky, K., & Zemel, R. S. (2017). Prototypical Networks
        for Few-shot Learning. Advances in Neural Information Processing
        Systems, 30, 4077-4087.
        https://doi.org/10.48550/arXiv.1703.05175
        """
        super().__init__(
            cost_function,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
        )

        choices_are_single = False

        if isinstance(choices, dict):
            self._boundaries = {
                key: self._to_choices(value) for key, value in choices.items()
            }
            self._is_kwargs = True
        else:
            materialized_choices = choices

            if isinstance(choices, Iterable) and not isinstance(
                choices, (str, bytes, np.ndarray)
            ):
                materialized_choices = list(choices)

            choices_are_single = self._is_choices_collection(materialized_choices)

            if choices_are_single:
                self._boundaries = [self._to_choices(materialized_choices)]
            else:
                self._boundaries = [
                    self._to_choices(value) for value in materialized_choices
                ]

            self._is_kwargs = False

        if not self._boundaries:
            raise ValueError("choices cannot be empty.")

        if isinstance(self._boundaries, dict):
            if any(not choices for choices in self._boundaries.values()):
                raise ValueError("Each ordinal variable must have at least one choice.")
        elif any(not choices for choices in self._boundaries):
            raise ValueError("Each ordinal variable must have at least one choice.")

        self._positions, self._positions_config = self._prepare_positions(
            positions, choices_are_single=choices_are_single
        )
        self._validate_positions()

        if decoder not in {
            "rank",
            "nearest-stochastic",
            "cumulative-logit",
            "cumulative-probit",
            "distance-softmax",
        }:
            raise ValueError(
                f"Invalid decoder: {decoder!r}. "
                "Expected one of: 'rank', 'nearest-stochastic', "
                "'cumulative-logit', 'cumulative-probit', 'distance-softmax'."
            )

        self._decoder = decoder
        self._params = {} if params is None else params
        self._params = dict(sorted(self._params.items()))

        if self._decoder in {
            "cumulative-logit",
            "cumulative-probit",
            "distance-softmax",
        }:
            temperature = self._params.get("temperature", 1.0)

            if isinstance(temperature, (bool, np.bool_)) or not isinstance(
                temperature, (int, float, np.integer, np.floating)
            ):
                raise ValueError(
                    "temperature must be a finite number greater than zero."
                )

            try:
                numeric_temperature = float(temperature)
            except (OverflowError, TypeError, ValueError) as error:
                raise ValueError(
                    "temperature must be a finite number greater than zero."
                ) from error

            if not np.isfinite(numeric_temperature) or numeric_temperature <= 0.0:
                raise ValueError(
                    "temperature must be a finite number greater than zero."
                )

        self._type = "ordinal"
        self._configs = {
            "positions": self._positions_config,
            "decoder": decoder,
            "params": self._params,
        }

        register_space(name=self._type, space_class=Ordinal)

    def initialize_context(self, seed: int | None = None) -> None:
        self._encoded_boundaries = {}

        for key, positions in self._iter_positions():
            self._encoded_boundaries[key] = self._position_bounds(positions)

        self._variable_names = [key for key, _ in self._iter_choices()]
        self._rng = np.random.default_rng(seed)

        if self._decoder != "rank" and seed is None:
            warnings.warn(
                "A stochastic decoding method was selected without a seed. "
                "Different evaluations may produce different results",
                RuntimeWarning,
                stacklevel=2,
            )

    def decode(self, float_inputs: dict[str, float]) -> OrdinalInput:
        self._check_input_bounds(float_inputs)

        decoders = {
            "rank": self._decode_rank,
            "nearest-stochastic": self._decode_nearest_stochastic,
            "cumulative-logit": self._decode_cumulative_logit,
            "cumulative-probit": self._decode_cumulative_probit,
            "distance-softmax": self._decode_distance_softmax,
        }

        decoder = decoders.get(self._decoder)

        if decoder is None:
            raise RuntimeError(f"Decoder {self._decoder!r} is not available.")

        decoded = decoder(float_inputs)

        if self._is_kwargs:
            return decoded

        return [decoded[str(index)] for index in range(len(self._boundaries))]

    def encode_cache(self, inputs: OrdinalInput) -> CacheKey:
        if isinstance(inputs, dict):
            if not self._is_kwargs:
                raise TypeError("Ordinal positional inputs must be provided as a list.")

            expected_keys = {key for key, _ in self._iter_choices()}

            if set(inputs) != expected_keys:
                raise ValueError(
                    "Ordinal cache inputs must contain exactly the configured "
                    "variables."
                )

            encoded = {
                key: self._choice_index(key, inputs[key]) for key in sorted(inputs)
            }
        else:
            if self._is_kwargs:
                raise TypeError(
                    "Ordinal keyword inputs must be provided as a dictionary."
                )

            values = list(inputs)

            if len(values) != len(self._boundaries):
                raise ValueError(
                    "Ordinal cache inputs must contain exactly one value per variable."
                )

            encoded = [
                self._choice_index(str(index), value)
                for index, value in enumerate(values)
            ]

        return json.dumps(encoded, sort_keys=True, separators=(",", ":"))

    def decode_cache(self, inputs: CacheKey) -> OrdinalInput:
        decoded_indices = json.loads(inputs)

        if self._is_kwargs:
            if not isinstance(decoded_indices, dict):
                raise TypeError("Invalid ordinal cache key for keyword inputs.")

            choices_by_key = dict(self._iter_choices())

            if set(decoded_indices) != set(choices_by_key):
                raise ValueError(
                    "Invalid ordinal cache key variables for keyword inputs."
                )

            return {
                key: choices_by_key[key][self._validate_cache_index(key, index)]
                for key, index in decoded_indices.items()
            }

        if not isinstance(decoded_indices, list):
            raise TypeError("Invalid ordinal cache key for positional inputs.")

        choices_by_key = dict(self._iter_choices())

        if len(decoded_indices) != len(choices_by_key):
            raise ValueError("Invalid ordinal cache key length.")

        return [
            choices_by_key[str(index)][
                self._validate_cache_index(str(index), choice_index)
            ]
            for index, choice_index in enumerate(decoded_indices)
        ]

    def _decode_rank(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        decoded = {}
        choices_by_key = dict(self._iter_choices())

        for key, positions in self._iter_positions():
            distances = np.abs(np.asarray(positions) - float_inputs[key])
            index = int(np.argmin(distances))
            decoded[key] = choices_by_key[key][index]

        return decoded

    def _decode_nearest_stochastic(
        self, float_inputs: dict[str, float]
    ) -> dict[str, Any]:
        decoded = {}
        choices_by_key = dict(self._iter_choices())

        for key, positions in self._iter_positions():
            value = float_inputs[key]
            choices = choices_by_key[key]

            if len(positions) == 1 or value <= positions[0]:
                decoded[key] = choices[0]
                continue

            if value >= positions[-1]:
                decoded[key] = choices[-1]
                continue

            upper_index = int(np.searchsorted(positions, value, side="right"))
            lower_index = upper_index - 1

            lower = positions[lower_index]
            upper = positions[upper_index]

            probability_upper = (value - lower) / (upper - lower)

            if self._rng.random() < probability_upper:
                decoded[key] = choices[upper_index]
            else:
                decoded[key] = choices[lower_index]

        return decoded

    def _decode_cumulative_logit(
        self, float_inputs: dict[str, float]
    ) -> dict[str, Any]:
        temperature = float(self._params.get("temperature", 1.0))

        return self._decode_cumulative(float_inputs, temperature, expit)

    def _decode_cumulative_probit(
        self, float_inputs: dict[str, float]
    ) -> dict[str, Any]:
        temperature = float(self._params.get("temperature", 1.0))

        return self._decode_cumulative(float_inputs, temperature, ndtr)

    def _decode_cumulative(
        self,
        float_inputs: dict[str, float],
        temperature: float,
        cumulative_function: Callable[[np.ndarray], np.ndarray],
    ) -> dict[str, Any]:
        decoded = {}
        choices_by_key = dict(self._iter_choices())

        for key, positions in self._iter_positions():
            choices = choices_by_key[key]

            if len(positions) == 1:
                decoded[key] = choices[0]
                continue

            thresholds = np.asarray(self._thresholds(positions), dtype=float)

            cumulative = cumulative_function(
                (thresholds - float_inputs[key]) / temperature
            )

            probabilities = np.empty(len(positions), dtype=float)

            probabilities[0] = cumulative[0]
            probabilities[1:-1] = np.diff(cumulative)
            probabilities[-1] = 1.0 - cumulative[-1]

            probabilities = np.clip(probabilities, 0.0, 1.0)
            probabilities /= probabilities.sum()

            index = int(self._rng.choice(len(choices), p=probabilities))

            decoded[key] = choices[index]

        return decoded

    def _decode_distance_softmax(
        self, float_inputs: dict[str, float]
    ) -> dict[str, Any]:
        temperature = float(self._params.get("temperature", 1.0))

        decoded = {}
        choices_by_key = dict(self._iter_choices())

        for key, positions in self._iter_positions():
            choices = choices_by_key[key]

            distances = np.abs(np.asarray(positions) - float_inputs[key])

            probabilities = softmax(-distances / temperature)

            index = int(self._rng.choice(len(choices), p=probabilities))

            decoded[key] = choices[index]

        return decoded

    def _prepare_positions(
        self,
        positions: Positions | PositionBoundaries | None,
        *,
        choices_are_single: bool,
    ) -> tuple[
        list[list[float]] | dict[str, list[float]],
        list[float] | list[list[float]] | dict[str, list[float]] | None,
    ]:
        if positions is None:
            if isinstance(self._boundaries, dict):
                generated = {
                    key: self._equidistant_positions(len(choices))
                    for key, choices in self._boundaries.items()
                }
            else:
                generated = [
                    self._equidistant_positions(len(choices))
                    for choices in self._boundaries
                ]

            return generated, None

        if isinstance(self._boundaries, dict):
            if not isinstance(positions, dict):
                raise TypeError(
                    "positions must be a dictionary when choices is a dictionary."
                )

            if set(positions) != set(self._boundaries):
                raise ValueError(
                    "positions must contain exactly the same keys as choices."
                )

            materialized = {
                key: self._to_positions(positions[key]) for key in self._boundaries
            }

            return materialized, materialized.copy()

        if isinstance(positions, dict):
            raise TypeError(
                "positions cannot be a dictionary when choices is not a dictionary."
            )

        materialized_positions: object = positions

        if isinstance(positions, Iterable) and not isinstance(
            positions, (str, bytes, np.ndarray)
        ):
            materialized_positions = list(positions)

        positions_are_single = self._is_positions_collection(materialized_positions)

        if choices_are_single != positions_are_single:
            raise ValueError("positions must have the same structure as choices.")

        if choices_are_single:
            single = self._to_positions(materialized_positions)

            return [single], single.copy()

        nested = [
            self._to_positions(value)
            for value in cast(Iterable[object], materialized_positions)
        ]

        return nested, [values.copy() for values in nested]

    def _validate_positions(self) -> None:
        choices_by_key = dict(self._iter_choices())
        positions_by_key = dict(self._iter_positions())

        if set(choices_by_key) != set(positions_by_key):
            raise ValueError("positions must define one collection for each variable.")

        for key, choices in choices_by_key.items():
            positions = positions_by_key[key]

            if len(positions) != len(choices):
                raise ValueError(
                    f"positions for variable {key!r} must contain exactly "
                    "one value per choice."
                )

            if len(positions) > 1:
                differences = np.diff(np.asarray(positions, dtype=float))

                if np.any(~np.isfinite(differences)) or np.any(differences <= 0.0):
                    raise ValueError(
                        f"positions for variable {key!r} must be strictly increasing."
                    )

            self._position_bounds(positions)

    def _iter_choices(self) -> list[tuple[str, list[Any]]]:
        if isinstance(self._boundaries, dict):
            return [(key, list(choices)) for key, choices in self._boundaries.items()]

        return [
            (str(index), list(choices))
            for index, choices in enumerate(self._boundaries)
        ]

    def _iter_positions(self) -> list[tuple[str, list[float]]]:
        if isinstance(self._positions, dict):
            return [
                (key, list(positions)) for key, positions in self._positions.items()
            ]

        return [
            (str(index), list(positions))
            for index, positions in enumerate(self._positions)
        ]

    def _choice_index(self, key: str, value: Any) -> int:
        choices_by_key = dict(self._iter_choices())

        if key not in choices_by_key:
            raise KeyError(f"Unknown ordinal variable: {key!r}.")

        choices = choices_by_key[key]

        for index, choice in enumerate(choices):
            if value is choice:
                return index

        for index, choice in enumerate(choices):
            try:
                equality = value == choice
            except Exception:
                continue

            if isinstance(equality, (bool, np.bool_)) and bool(equality):
                return index

            if isinstance(equality, np.ndarray) and equality.shape != ():
                if bool(np.all(equality)):
                    return index

        raise ValueError(
            f"Value {value!r} is not a configured choice for variable {key!r}."
        )

    def _validate_cache_index(self, key: str, index: Any) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Invalid ordinal cache index.")

        choices_by_key = dict(self._iter_choices())

        if key not in choices_by_key:
            raise KeyError(f"Unknown ordinal variable in cache key: {key!r}.")

        if not 0 <= index < len(choices_by_key[key]):
            raise ValueError("Ordinal cache index is outside the configured choices.")

        return index

    @staticmethod
    def _equidistant_positions(size: int) -> list[float]:
        return [float(index) for index in range(size)]

    @staticmethod
    def _thresholds(positions: list[float]) -> list[float]:
        return [
            lower + ((upper - lower) / 2.0)
            for lower, upper in zip(positions[:-1], positions[1:], strict=True)
        ]

    @staticmethod
    def _position_bounds(positions: list[float]) -> tuple[float, float]:
        if len(positions) == 1:
            lower = positions[0] - 0.5
            upper = positions[0] + 0.5
        else:
            lower_gap = positions[1] - positions[0]
            upper_gap = positions[-1] - positions[-2]

            lower = positions[0] - (lower_gap / 2.0)
            upper = positions[-1] + (upper_gap / 2.0)

        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            raise ValueError(
                "positions produce invalid continuous boundaries for the solver."
            )

        return float(lower), float(upper)

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

        return not Ordinal._is_nested_collection(first)

    @staticmethod
    def _is_positions_collection(value: object) -> bool:
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

        return not Ordinal._is_nested_collection(first)

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
                    "Each ordinal variable must be represented by "
                    "a one-dimensional collection of choices."
                )

            return value.tolist()

        if not isinstance(value, Iterable):
            raise TypeError(
                "Each ordinal variable must be represented by "
                "an iterable collection of choices."
            )

        return list(value)

    @staticmethod
    def _to_positions(value: object) -> list[float]:
        if isinstance(value, np.ndarray):
            if value.ndim != 1:
                raise ValueError(
                    "Each ordinal variable must be represented by "
                    "a one-dimensional collection of positions."
                )

            values = value.tolist()
        else:
            if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
                raise TypeError(
                    "Each ordinal variable must be represented by "
                    "an iterable collection of positions."
                )

            values = list(value)

        positions = []

        for position in values:
            if isinstance(position, (bool, np.bool_)) or not isinstance(
                position, (int, float, np.integer, np.floating)
            ):
                raise TypeError("Each ordinal position must be a finite number.")

            try:
                numeric_position = float(position)
            except (OverflowError, TypeError, ValueError) as error:
                raise TypeError(
                    "Each ordinal position must be a finite number."
                ) from error

            if not np.isfinite(numeric_position):
                raise TypeError("Each ordinal position must be a finite number.")

            positions.append(numeric_position)

        return positions
