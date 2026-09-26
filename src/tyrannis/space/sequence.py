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
BaseChoices: TypeAlias = list[list[Any]] | dict[str, Iterable[Any]]
BasePositions: TypeAlias = PositionBoundaries | dict[str, Iterable[float]] | None
BaseInput: TypeAlias = list[Any] | dict[str, Any]
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
        """
        Initialize the user-facing sequence search space.

        The sequence space represents an ordered collection of choices whose length
        may vary between configured minimum and maximum limits. Elements can appear
        multiple times in the decoded sequence, and their order is significant.

        The available choices can be provided as a single collection or as a
        dictionary containing exactly one named collection. A single collection
        represents one positional input of the cost function, while a dictionary
        represents one keyword input.

        The concrete collection type is not relevant to the search-space interface.
        Lists, tuples, NumPy arrays, and other collection types can be used to
        represent the available choices.

        Sequences can use either categorical or ordinal decoding. Categorical
        decoding treats the available choices as distinct nominal alternatives,
        without assuming any ordering or distance relationship between them.
        Ordinal decoding instead places the choices on an ordered continuous scale,
        so their relative order and spacing become part of the search-space
        geometry. Although an ordinal sequence with equally spaced positions may
        look similar to a categorical sequence from the public API, the two
        represent mathematically different optimization domains and can therefore
        lead to substantially different search behavior.

        Parameters
        ----------
        choices:
            Elements available at every position of the sequence. A single
            collection represents one positional sequence input. A dictionary may
            be used to define a keyword input, but it must contain exactly one key
            mapping the input name to its collection of available choices. Elements
            may occur multiple times in the decoded sequence.

            When multiple independent sequence inputs are required in the same cost
            function, use a ``Mixed`` space containing one ``Sequence`` instance for
            each input instead of providing multiple keys to a single ``Sequence``.

        max_choices:
            Maximum number of elements in the decoded sequence. The value must be
            an integer greater than or equal to one and greater than or equal to
            ``min_choices``.

        min_choices:
            Minimum number of elements in the decoded sequence. The value must be
            a non-negative integer no greater than ``max_choices``. Sequence
            termination is not allowed before this number of elements has been
            selected.

        positions:
            Continuous positions associated one-to-one with ``choices`` when using
            an ordinal decoder. Positions are always provided as a single
            collection, including when ``choices`` is provided as a dictionary.
            Each position must be finite and the positions must be strictly
            increasing.

            When ``None`` and an ordinal decoder is explicitly selected, equally
            spaced positions are generated automatically. In this case,
            ``stop_positions`` must also be ``None``. This argument must be
            ``None`` when using a categorical decoder.

        stop_positions:
            Continuous position or positions associated with sequence termination
            when explicit ordinal ``positions`` are provided. A single finite value
            assigns the same termination position to every optional sequence
            position. A collection assigns an individual termination position to
            each optional sequence position and must therefore contain exactly
            ``max_choices - min_choices`` values.

            Each termination position must be finite and must not coincide with a
            position assigned to an element in ``choices``. This argument is
            required when explicit ``positions`` are provided and
            ``min_choices < max_choices``. It must be ``None`` when positions are
            generated automatically, when using a categorical decoder, or when
            ``min_choices == max_choices``.

        decoder:
            Method used to decode the continuous solver representation into the
            sequence. The decoder name follows the ``"<space>-<decoder>"`` format,
            where the prefix selects either categorical or ordinal decoding.
            Supported methods include:

            - ``"categorical-one-hot"``: Selects the category with the highest
            encoded value at each sequence position.
            - ``"categorical-softmax"``: Stochastically samples each sequence
            position according to softmax probabilities.
            - ``"categorical-gumbel-softmax"``: Stochastically samples each
            sequence position using Gumbel noise and softmax.
            - ``"categorical-scalar"``: Represents each sequence position with a
            single continuous value.
            - ``"ordinal-rank"``: Selects the ordinal choice whose continuous
            position is nearest to the solver value.
            - ``"ordinal-nearest-stochastic"``: Stochastically selects between the
            two ordinal positions adjacent to the solver value.
            - ``"ordinal-cumulative-logit"``: Stochastically samples an ordinal
            level using a cumulative logistic model.
            - ``"ordinal-cumulative-probit"``: Stochastically samples an ordinal
            level using a cumulative normal model.
            - ``"ordinal-distance-softmax"``: Stochastically samples among all
            ordinal levels using a softmax distribution over their distances
            from the solver value.

            When ``None``, ``"categorical-one-hot"`` is used if ``positions`` is
            ``None``. When explicit ``positions`` are provided,
            ``"ordinal-rank"`` is used instead. To use ordinal decoding with
            automatically generated positions, an ordinal decoder must be selected
            explicitly.

        params:
            Parameters used by the selected decoding method. Parameters are
            provided as a dictionary where each key is the name of a parameter and
            its value is the corresponding parameter value. Supported parameters
            depend on the selected categorical or ordinal decoder and include:

            ``bounds``:
                Continuous search interval exposed to the optimization algorithm
                when using a categorical decoder. When omitted, decoder-specific
                default bounds are used. ``"categorical-scalar"`` defaults to
                ``(0.0, 1.0)`` while the remaining categorical decoders default to
                ``(-1.0, 1.0)``. This parameter is not supported by ordinal
                decoders.

            ``temperature``:
                Controls the stochasticity of the
                ``"categorical-gumbel-softmax"``,
                ``"ordinal-cumulative-logit"``,
                ``"ordinal-cumulative-probit"``, and
                ``"ordinal-distance-softmax"`` decoders. Lower values concentrate
                probability more strongly around favored choices, while higher
                values produce broader probability distributions. The value must
                be finite and greater than zero. Defaults to ``1.0``.

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
            - ``"disk"``: Temporary disk-backed runtime cache. Its contents are
            local to the current runtime and are not preserved through serialization.
            - ``"lfu"``: Least Frequently Used cache. Requires the optional
            dependencies for advanced caching.
            - ``"fifo"``: First In, First Out cache. Requires the optional dependencies
            for advanced caching.
            - ``"rr"``: Random Replacement cache. Requires the optional dependencies
            for advanced caching.

        cache_size:
            Maximum cache size for in-memory caches, expressed as the maximum
            number of cached records. This parameter has no effect when
            ``cache_type="disk"``.

        Notes
        -----
        Let ``C`` denote the collection of available choices, ``m`` the configured
        ``min_choices``, and ``M`` the configured ``max_choices``. The user-facing
        search domain is the set of all finite ordered sequences over ``C`` whose
        length is between these limits:

        ``S(C, m, M) = union(C^k, k=m,...,M)``.

        Unlike a permutation, membership in ``C^k`` does not impose uniqueness.
        Consequently, the same element may occupy multiple positions and sequences
        such as ``(A, B, A, C)`` are valid whenever their length satisfies the
        configured limits.

        A bounded variable-length sequence can be represented by ``M`` ordered
        decision positions together with a distinguished termination symbol
        ``STOP``. For positions up to ``m``, the admissible domain is ``C``. For
        the remaining optional positions, the admissible domain is
        ``C union {STOP}``:

        ``D_i = C`` for ``i <= m``,

        ``D_i = C union {STOP}`` for ``m < i <= M``.

        For a decoded vector ``z = (z_1, ..., z_M)``, the first occurrence of
        ``STOP`` determines the sequence length. If ``STOP`` occurs at position
        ``j``, the resulting sequence is ``(z_1, ..., z_(j-1))``. If no
        termination symbol occurs, all ``M`` elements form the resulting sequence.
        Because ``STOP`` is unavailable in the first ``m`` positions, the decoded
        length is always contained in ``[m, M]``.

        Under categorical decoding, each decision position represents a categorical
        choice, with ``STOP`` acting as an additional category only in optional
        positions.

        Under ordinal decoding, each element ``c_j`` in ``C`` is associated with a
        continuous position ``p_j``. Optional sequence positions additionally
        associate ``STOP`` with a termination position. A scalar
        ``stop_positions`` value defines the same termination position for every
        optional decision, while a collection defines positions
        ``s_1, ..., s_(M-m)`` independently. This allows the continuous geometry of
        the termination level to vary along the sequence without changing the
        discrete sequence domain.

        When ordinal positions are generated automatically, each decision position
        uses equally spaced ordinal levels. Optional positions therefore contain
        one additional equally spaced level corresponding to ``STOP``.

        This formulation combines a fixed-dimensional continuous optimization
        representation with a variable-length discrete decoded domain. It follows
        ideas used in variable-length chromosome representations, random-key
        sequence encodings, and fixed-length encodings containing inactive or
        do-nothing alternatives; the particular formulation above is a unified
        search-space definition rather than a single representation adopted
        verbatim from one reference.

        References
        ----------
        Bean, J. C. (1994). Genetic Algorithms and Random Keys for Sequencing and
        Optimization. ORSA Journal on Computing, 6(2), 154-160.
        https://doi.org/10.1287/ijoc.6.2.154

        Zhang, Q., & Ding, L. (2016). A New Crossover Mechanism for Genetic
        Algorithms with Variable-Length Chromosomes for Path Optimization Problems.
        Expert Systems with Applications, 60, 183-189.
        https://doi.org/10.1016/j.eswa.2016.04.005

        Beke, L., Weiszer, M., & Chen, J. (2021). A Comparison of Genetic
        Representations and Initialisation Methods for the Multi-objective Shortest
        Path Problem on Multigraphs. SN Computer Science, 2, 176.
        https://doi.org/10.1007/s42979-021-00512-z

        Ni, Y., Du, X., Ye, P., Minku, L. L., Yao, X., Harman, M., & Xiao, R.
        (2021). Multi-objective Software Performance Optimisation at the
        Architecture Level Using Randomised Search Rules. Information and Software
        Technology, 135, 106565.
        https://doi.org/10.1016/j.infsof.2021.106565
        """
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
        sequence = self._truncate_at_stop(self._base_values(decoded))

        if self._input_name is None:
            return [sequence]

        return {self._input_name: sequence}

    def encode_cache(self, inputs: SequenceInput) -> CacheKey:
        sequence = self._validate_sequence_inputs(inputs)
        padded = self._pad_with_stop(sequence)
        base_inputs = self._base_inputs(padded)

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

        sequence = self._truncate_at_stop(self._base_values(decoded))

        if self._input_name is None:
            return [sequence]

        return {self._input_name: sequence}

    def _create_categorical_space(self, decoder: str) -> Categorical:
        base_params = self._params.copy()
        bounds = base_params.pop("bounds", None)
        boundaries = self._base_choices(self._categorical_boundaries())

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
            choices=self._base_choices(choices),
            positions=self._base_positions(positions),
            decoder=decoder,
            params=self._params,
            use_cache=self._use_cache,
            cache_type=self._cache_type,
            cache_size=self._cache_size,
        )

    def _base_choices(self, choices: list[list[Any]]) -> BaseChoices:
        if self._input_name is None:
            return choices

        boundaries: dict[str, Iterable[Any]] = {}

        for index, step_choices in enumerate(choices):
            boundaries[self._step_key(index)] = step_choices

        return boundaries

    def _base_positions(self, positions: PositionBoundaries | None) -> BasePositions:
        if self._input_name is None or positions is None:
            return positions

        boundaries: dict[str, Iterable[float]] = {}

        for index, step_positions in enumerate(positions):
            boundaries[self._step_key(index)] = step_positions

        return boundaries

    def _base_inputs(self, values: list[Any]) -> BaseInput:
        if self._input_name is None:
            return values

        return {self._step_key(index): value for index, value in enumerate(values)}

    def _base_values(self, values: BaseInput) -> list[Any]:
        if self._input_name is None:
            if not isinstance(values, list):
                raise TypeError("Sequence base space must decode positional inputs.")

            return values

        if not isinstance(values, dict):
            raise TypeError("Sequence base space must decode keyword inputs.")

        step_keys = [self._step_key(index) for index in range(self._max_choices)]

        if set(values) != set(step_keys):
            raise ValueError(
                "Sequence base space decoded unexpected internal variables."
            )

        return [values[key] for key in step_keys]

    def _step_key(self, index: int) -> str:
        if self._input_name is None:
            return str(index)

        return f"{self._input_name}-{index}"

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
