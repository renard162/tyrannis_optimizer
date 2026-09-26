from collections.abc import Callable, Collection, Iterable, Mapping
from enum import Enum
from typing import Any, ClassVar, TypeAlias

import numpy as np

from ..core.space import SpaceBase
from . import register_space
from .categorical import Categorical
from .ordinal import Ordinal

Choice: TypeAlias = Any
Choices: TypeAlias = Collection[Choice]
Boundaries: TypeAlias = Choices | dict[str, Choices]
Positions: TypeAlias = Collection[float]
PositionBoundaries: TypeAlias = list[Iterable[float]]
StopPositions: TypeAlias = float | Collection[float]
SequenceInput: TypeAlias = list[list[Any]] | dict[str, list[Any]]
CacheKey: TypeAlias = tuple[Any, ...] | str


class _SequenceToken(Enum):
    STOP = "stop"


class Sequence(SpaceBase):
    """Variable-length ordered sequence optimization search space."""

    _CATEGORICAL_DECODERS: ClassVar[set[str]] = {
        "one-hot",
        "softmax",
        "gumbel-softmax",
        "scalar",
    }
    _ORDINAL_DECODERS: ClassVar[set[str]] = {
        "rank",
        "nearest-stochastic",
        "cumulative-logit",
        "cumulative-probit",
        "distance-softmax",
    }

    _boundaries: list[list[Any]] | dict[str, list[Any]]
    _choices: list[Any]
    _input_name: str | None
    _max_choices: int
    _min_choices: int
    _positions_config: list[float] | None
    _stop_positions_config: float | list[float] | None
    _decoder: str
    _params: dict[str, Any]
    _base_space: Categorical | Ordinal

    def __init__(
        self,
        choices: Boundaries,
        max_choices: int,
        min_choices: int,
        positions: Positions | None = None,
        stop_positions: StopPositions | None = None,
        decoder: str | None = None,
        params: dict[str, Any] | None = None,
        cost_function: Callable[..., float] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        super().__init__(
            cost_function,
            use_cache=use_cache,
            cache_type=cache_type,
            cache_size=cache_size,
        )

        self._choices, self._input_name = self._prepare_choices(choices)
        self._validate_choice_limits(min_choices=min_choices, max_choices=max_choices)

        self._min_choices = min_choices
        self._max_choices = max_choices
        self._is_kwargs = self._input_name is not None

        space_type, base_decoder = self._resolve_decoder(
            decoder=decoder, positions=positions
        )
        self._decoder = f"{space_type}-{base_decoder}"

        self._params = {} if params is None else dict(params)
        self._params = dict(sorted(self._params.items()))

        if space_type == "categorical":
            if positions is not None or stop_positions is not None:
                raise ValueError(
                    "positions and stop_positions must be None when using a "
                    "categorical sequence decoder."
                )

            self._positions_config = None
            self._stop_positions_config = None
            self._base_space = self._create_categorical_space(base_decoder)
        else:
            (
                ordinal_choices,
                ordinal_positions,
                positions_config,
                stop_positions_config,
            ) = self._prepare_ordinal_space(
                positions=positions, stop_positions=stop_positions
            )

            self._positions_config = positions_config
            self._stop_positions_config = stop_positions_config
            self._base_space = self._create_ordinal_space(
                choices=ordinal_choices,
                positions=ordinal_positions,
                decoder=base_decoder,
            )

        if self._input_name is None:
            self._boundaries = [self._choices.copy()]
        else:
            self._boundaries = {self._input_name: self._choices.copy()}

        self._type = "sequence"
        self._configs = {
            "max_choices": self._max_choices,
            "min_choices": self._min_choices,
            "positions": self._positions_config,
            "stop_positions": self._stop_positions_config,
            "decoder": self._decoder,
            "params": self._params,
            "groupable": False,
        }

        register_space(name=self._type, space_class=Sequence)

    def initialize_context(self, seed: int | None = None) -> None:
        self._base_space.initialize_context(seed)
        self._encoded_boundaries = self._base_space.encoded_boundaries

        if self._input_name is None:
            self._variable_names = ["0"]
        else:
            self._variable_names = [self._input_name]

    def decode(self, float_inputs: dict[str, float]) -> SequenceInput:
        self._check_input_bounds(float_inputs)
        decoded = self._base_space.decode(float_inputs)

        if not isinstance(decoded, list):
            raise TypeError("Sequence base space must decode positional inputs.")

        sequence = self._truncate_at_stop(decoded)

        if self._input_name is None:
            return [sequence]

        return {self._input_name: sequence}

    def encode_cache(self, inputs: SequenceInput) -> CacheKey:
        sequence = self._validate_sequence_inputs(inputs)
        base_inputs = self._pad_with_stop(sequence)

        return self._base_space.encode_cache(base_inputs)

    def decode_cache(self, inputs: CacheKey) -> SequenceInput:
        if isinstance(self._base_space, Categorical):
            if not isinstance(inputs, tuple):
                raise TypeError("Invalid categorical Sequence cache key.")

            decoded = self._base_space.decode_cache(inputs)
        else:
            if not isinstance(inputs, str):
                raise TypeError("Invalid ordinal Sequence cache key.")

            decoded = self._base_space.decode_cache(inputs)

        if not isinstance(decoded, list):
            raise TypeError("Sequence base space must decode positional cache inputs.")

        sequence = self._truncate_at_stop(decoded)

        if self._input_name is None:
            return [sequence]

        return {self._input_name: sequence}

    def _create_categorical_space(self, decoder: str) -> Categorical:
        base_params = self._params.copy()
        bounds = base_params.pop("bounds", None)
        boundaries = self._categorical_boundaries()

        return Categorical(
            choices=boundaries,
            decoder=decoder,
            bounds=bounds,
            params=base_params,
            use_cache=self._use_cache,
            cache_type=self._cache_type,
            cache_size=self._cache_size,
        )

    def _create_ordinal_space(
        self,
        choices: list[list[Any]],
        positions: PositionBoundaries | None,
        decoder: str,
    ) -> Ordinal:
        if "bounds" in self._params:
            raise ValueError(
                "params['bounds'] is only supported by categorical sequence decoders."
            )

        return Ordinal(
            choices=choices,
            positions=positions,
            decoder=decoder,
            params=self._params,
            use_cache=self._use_cache,
            cache_type=self._cache_type,
            cache_size=self._cache_size,
        )

    def _categorical_boundaries(self) -> list[list[Any]]:
        boundaries = []

        for index in range(self._max_choices):
            step_choices = self._choices.copy()

            if index >= self._min_choices:
                step_choices.append(_SequenceToken.STOP)

            boundaries.append(step_choices)

        return boundaries

    def _prepare_ordinal_space(
        self, positions: Positions | None, stop_positions: StopPositions | None
    ) -> tuple[
        list[list[Any]],
        PositionBoundaries | None,
        list[float] | None,
        float | list[float] | None,
    ]:
        optional_choices = self._max_choices - self._min_choices

        if positions is None:
            if stop_positions is not None:
                raise ValueError(
                    "stop_positions must be None when ordinal positions are "
                    "generated automatically."
                )

            boundaries = self._ordinal_boundaries_without_positions()
            return boundaries, None, None, None

        base_positions = self._materialize_positions(positions)

        if len(base_positions) != len(self._choices):
            raise ValueError(
                "positions must contain exactly one value per sequence choice."
            )

        if optional_choices == 0:
            if stop_positions is not None:
                raise ValueError(
                    "stop_positions must be None when min_choices equals max_choices."
                )

            boundaries = [self._choices.copy() for _ in range(self._max_choices)]
            position_boundaries: PositionBoundaries = [
                base_positions.copy() for _ in range(self._max_choices)
            ]
            return boundaries, position_boundaries, base_positions.copy(), None

        materialized_stops, stop_positions_config = self._materialize_stop_positions(
            stop_positions,
            expected_size=optional_choices,
            choice_positions=base_positions,
        )

        boundaries: list[list[Any]] = []
        position_boundaries: PositionBoundaries = []

        for index in range(self._max_choices):
            if index < self._min_choices:
                boundaries.append(self._choices.copy())
                position_boundaries.append(base_positions.copy())
                continue

            stop_position = materialized_stops[index - self._min_choices]
            ordered = list(zip(base_positions, self._choices, strict=True))
            ordered.append((stop_position, _SequenceToken.STOP))
            ordered.sort(key=lambda item: item[0])

            position_boundaries.append([position for position, _ in ordered])
            boundaries.append([choice for _, choice in ordered])

        return (
            boundaries,
            position_boundaries,
            base_positions.copy(),
            stop_positions_config,
        )

    def _ordinal_boundaries_without_positions(self) -> list[list[Any]]:
        boundaries = []

        for index in range(self._max_choices):
            step_choices = self._choices.copy()

            if index >= self._min_choices:
                step_choices.append(_SequenceToken.STOP)

            boundaries.append(step_choices)

        return boundaries

    def _materialize_stop_positions(
        self,
        stop_positions: StopPositions | None,
        *,
        expected_size: int,
        choice_positions: list[float],
    ) -> tuple[list[float], float | list[float]]:
        if stop_positions is None:
            raise ValueError(
                "stop_positions must be provided when positions is defined."
            )

        if self._is_numeric(stop_positions):
            stop_position = self._to_finite_float(stop_positions, name="stop_positions")
            self._validate_stop_collision(
                stop_position, choice_positions=choice_positions
            )
            return [stop_position] * expected_size, stop_position

        if isinstance(stop_positions, (str, bytes, Mapping)) or not isinstance(
            stop_positions, Collection
        ):
            raise TypeError(
                "stop_positions must be a finite number or a collection of "
                "finite numbers."
            )

        materialized = [
            self._to_finite_float(value, name="stop_positions")
            for value in stop_positions
        ]

        if len(materialized) != expected_size:
            raise ValueError(
                "stop_positions must contain exactly one position for each "
                "optional sequence choice."
            )

        for stop_position in materialized:
            self._validate_stop_collision(
                stop_position, choice_positions=choice_positions
            )

        return materialized, materialized.copy()

    def _validate_stop_collision(
        self, stop_position: float, *, choice_positions: list[float]
    ) -> None:
        if stop_position in choice_positions:
            raise ValueError(
                f"stop position {stop_position} collides with a configured "
                "sequence choice position."
            )

    def _materialize_positions(self, positions: Positions) -> list[float]:
        if isinstance(positions, (str, bytes, Mapping)) or not isinstance(
            positions, Collection
        ):
            raise TypeError("positions must be a collection of finite numbers.")

        if isinstance(positions, np.ndarray) and positions.ndim != 1:
            raise ValueError("positions must be a one-dimensional collection.")

        materialized = [
            self._to_finite_float(position, name="positions") for position in positions
        ]

        if len(materialized) > 1:
            differences = np.diff(np.asarray(materialized, dtype=float))

            if np.any(~np.isfinite(differences)) or np.any(differences <= 0.0):
                raise ValueError("positions must be strictly increasing.")

        return materialized

    @classmethod
    def _prepare_choices(cls, choices: Boundaries) -> tuple[list[Any], str | None]:
        if isinstance(choices, dict):
            if len(choices) != 1:
                raise ValueError(
                    "Sequence keyword choices must contain exactly one variable."
                )

            input_name, input_choices = next(iter(choices.items()))
            return cls._materialize_choices(input_choices), input_name

        return cls._materialize_choices(choices), None

    @staticmethod
    def _materialize_choices(choices: Choices) -> list[Any]:
        if isinstance(choices, (str, bytes, Mapping)) or not isinstance(
            choices, Collection
        ):
            raise TypeError("choices must be a collection of sequence options.")

        if isinstance(choices, np.ndarray) and choices.ndim != 1:
            raise ValueError("choices must be a one-dimensional collection.")

        materialized = list(choices)

        if not materialized:
            raise ValueError("choices cannot be empty.")

        return materialized

    @staticmethod
    def _validate_choice_limits(*, min_choices: int, max_choices: int) -> None:
        if isinstance(min_choices, bool) or not isinstance(min_choices, int):
            raise TypeError("min_choices must be an integer.")

        if isinstance(max_choices, bool) or not isinstance(max_choices, int):
            raise TypeError("max_choices must be an integer.")

        if min_choices < 0:
            raise ValueError("min_choices must be greater than or equal to zero.")

        if max_choices < 1:
            raise ValueError("max_choices must be greater than or equal to one.")

        if min_choices > max_choices:
            raise ValueError("min_choices cannot be greater than max_choices.")

    @classmethod
    def _resolve_decoder(
        cls, *, decoder: str | None, positions: Positions | None
    ) -> tuple[str, str]:
        if decoder is None:
            if positions is None:
                return "categorical", "one-hot"

            return "ordinal", "rank"

        if not isinstance(decoder, str):
            raise TypeError("decoder must be a string or None.")

        if decoder.startswith("categorical-"):
            base_decoder = decoder.removeprefix("categorical-")

            if base_decoder not in cls._CATEGORICAL_DECODERS:
                raise ValueError(f"Invalid categorical sequence decoder: {decoder!r}.")

            return "categorical", base_decoder

        if decoder.startswith("ordinal-"):
            base_decoder = decoder.removeprefix("ordinal-")

            if base_decoder not in cls._ORDINAL_DECODERS:
                raise ValueError(f"Invalid ordinal sequence decoder: {decoder!r}.")

            return "ordinal", base_decoder

        raise ValueError(
            "decoder must use the '<space>-<decoder>' format with either the "
            "'categorical-' or 'ordinal-' prefix."
        )

    def _validate_sequence_inputs(self, inputs: SequenceInput) -> list[Any]:
        if self._input_name is None:
            if not isinstance(inputs, list) or len(inputs) != 1:
                raise TypeError(
                    "Sequence positional inputs must contain exactly one sequence."
                )

            sequence = inputs[0]
        else:
            if not isinstance(inputs, dict):
                raise TypeError(
                    "Sequence keyword inputs must be provided as a dictionary."
                )

            if set(inputs) != {self._input_name}:
                raise ValueError(
                    "Sequence keyword inputs must contain exactly the "
                    "configured variable."
                )

            sequence = inputs[self._input_name]

        if not isinstance(sequence, list):
            raise TypeError("The Sequence input must be represented as a list.")

        if not self._min_choices <= len(sequence) <= self._max_choices:
            raise ValueError(
                "Sequence cache inputs must contain between min_choices and "
                "max_choices elements."
            )

        return sequence.copy()

    def _pad_with_stop(self, sequence: list[Any]) -> list[Any]:
        padded = sequence.copy()

        while len(padded) < self._max_choices:
            padded.append(_SequenceToken.STOP)

        return padded

    def _truncate_at_stop(self, values: list[Any]) -> list[Any]:
        sequence = []

        for value in values:
            if value is _SequenceToken.STOP:
                break

            sequence.append(value)

        if not self._min_choices <= len(sequence) <= self._max_choices:
            raise RuntimeError(
                "Decoded Sequence length is outside the configured limits."
            )

        return sequence

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return not isinstance(value, (bool, np.bool_)) and isinstance(
            value, (int, float, np.integer, np.floating)
        )

    @classmethod
    def _to_finite_float(cls, value: Any, *, name: str) -> float:
        if not cls._is_numeric(value):
            raise TypeError(f"{name} must contain only finite numbers.")

        try:
            numeric = float(value)
        except (OverflowError, TypeError, ValueError) as error:
            raise TypeError(f"{name} must contain only finite numbers.") from error

        if not np.isfinite(numeric):
            raise TypeError(f"{name} must contain only finite numbers.")

        return numeric
