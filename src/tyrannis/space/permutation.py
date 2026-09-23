import warnings
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias, cast

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.special import softmax
from scipy.stats import gumbel_r

from ..core.space import SpaceBase
from . import register_space

Choice: TypeAlias = Any
Choices: TypeAlias = Iterable[Choice]
Boundaries: TypeAlias = list[Choices] | dict[str, Choices]
PermutationInput: TypeAlias = list[list[Any]] | dict[str, list[Any]]
CacheKey: TypeAlias = tuple[Any, ...] | tuple[tuple[str, Any], ...]


class Permutation(SpaceBase):
    """Permutation optimization search space."""

    _boundaries: Boundaries
    _bounds: tuple[float, float] | None
    _decoder: str
    _params: dict[str, Any]
    _rng: np.random.Generator

    def __init__(
        self,
        choices: Choices | Boundaries,
        cost_function: Callable[..., float] | None = None,
        decoder: str = "random-keys",
        bounds: tuple[float, float] | None = None,
        params: dict[str, Any] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing permutation search space.

        The permutation choices can be provided as a single collection, a
        collection of collections, or a dictionary associating each variable
        with its collection of possible choices.

        When a single collection is provided, it represents one permutation
        input. This form is also useful when the space is used as a component
        of a mixed search space.

        When a collection of collections is provided, each inner collection
        represents one positional input of the cost function, with the
        elements of each collection being permuted independently.

        When a dictionary is provided, each key identifies one input of the
        cost function and its associated collection contains the elements
        whose order is optimized.

        The concrete collection type is not relevant to the search-space
        interface. Lists, tuples, NumPy arrays, and other iterable
        collections can be used to represent the available choices.

        Parameters
        ----------
        choices:
            Elements available to each permutation variable. A single
            collection represents one permutation input. A collection of
            collections represents multiple positional permutation inputs.
            A dictionary maps each input name to the collection whose order
            will be optimized.

        cost_function:
            User-defined cost function evaluated after decoding the solver
            inputs into their user-facing representation. This argument is
            required when the space is used independently. It does not need
            to be provided when the space is used as a component of a
            ``Mixed`` space, because in that case the cost function is
            provided to the ``Mixed`` space itself.

        decoder:
            Method used to decode the continuous solver representation into
            permutations. Supported methods include:

            - ``"random-keys"``: Sorts continuous values to determine the permutation.
            (default)
            - ``"gumbel-random-keys"``: Stochastically perturbs values before sorting.
            - ``"plackett-luce"``: Stochastically samples permutations using the
            Plackett-Luce model.
            - ``"gumbel-sinkhorn"``: Stochastically approximates permutations using
            Gumbel noise and Sinkhorn normalization.

        bounds:
            Continuous search interval exposed to the optimization
            algorithm. When ``None``, ``(0.0, 1.0)`` is used.

        params:
            Parameters used by the selected decoding method. Parameters are
            provided as a dictionary where each key is the name of a parameter
            and its value is the corresponding parameter value. Supported
            parameters are:

            ``temperature``:
                Controls the stochasticity of the ``"gumbel-random-keys"``,
                ``"plackett-luce"``, and ``"gumbel-sinkhorn"`` decoders.
                Higher values increase randomness, while lower values make
                decoding more deterministic. Recommended range: 0.1–10,
                with low values (0.1–1), medium values (1–5), and high values
                (5–10). Values below 0.1 or above 10 are generally ineffective
                in practice. Defaults to ``1.0``.

            ``sinkhorn_iterations``:
                Number of normalization iterations performed by the
                Sinkhorn algorithm in the ``"gumbel-sinkhorn"`` decoder.
                Higher values produce a closer approximation to a
                doubly stochastic matrix at increased computational cost.
                Defaults to ``20``.

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
        Bean, J. C. (1994). Genetic Algorithms and Random Keys for Sequencing
        and Optimization. ORSA Journal on Computing, 6(2), 154-160.
        https://doi.org/10.1287/ijoc.6.2.154

        Yellott, J. I., Jr. (1977). The Relationship Between Luce's Choice
        Axiom, Thurstone's Theory of Comparative Judgment, and the Double
        Exponential Distribution. Journal of Mathematical Psychology, 15(2),
        109-144. https://doi.org/10.1016/0022-2496(77)90026-8

        Plackett, R. L. (1975). The Analysis of Permutations. Journal of the
        Royal Statistical Society: Series C (Applied Statistics), 24(2),
        193-202. https://doi.org/10.2307/2346567

        Mena, G., Belanger, D., Linderman, S., & Snoek, J. (2018). Learning
        Latent Permutations with Gumbel-Sinkhorn Networks. International
        Conference on Learning Representations.
        https://doi.org/10.48550/arXiv.1802.08665
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
        else:
            materialized_choices = choices

            if isinstance(choices, Iterable) and not isinstance(
                choices, (str, bytes, np.ndarray)
            ):
                materialized_choices = list(choices)

            if self._is_choices_collection(materialized_choices):
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
                raise ValueError(
                    "Each permutation variable must have at least one choice."
                )
        elif any(not choices for choices in self._boundaries):
            raise ValueError("Each permutation variable must have at least one choice.")

        if decoder not in {
            "random-keys",
            "gumbel-random-keys",
            "plackett-luce",
            "gumbel-sinkhorn",
        }:
            raise ValueError(
                f"Invalid decoder: {decoder!r}. "
                "Expected one of: 'random-keys', 'gumbel-random-keys', "
                "'plackett-luce', 'gumbel-sinkhorn'."
            )

        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError("bounds must contain exactly two values.")

            if bounds[0] >= bounds[1]:
                raise ValueError(
                    "The lower bound must be smaller than the upper bound."
                )

        self._bounds = bounds
        self._decoder = decoder
        self._params = {} if params is None else params
        self._params = dict(sorted(self._params.items()))
        self._type = "permutation"
        self._configs = {"decoder": decoder, "bounds": bounds, "params": self._params}

        register_space(name=self._type, space_class=Permutation)

    def initialize_context(self, seed: int | None = None) -> None:
        bounds = self._bounds

        if bounds is None:
            bounds = (0.0, 1.0)

        self._encoded_boundaries = {}

        if self._decoder == "gumbel-sinkhorn":
            encoded_matrix_keys = self._encoded_matrix_keys()

            for matrix_keys in encoded_matrix_keys.values():
                for row_keys in matrix_keys:
                    for encoded_key in row_keys:
                        self._encoded_boundaries[encoded_key] = bounds
        else:
            encoded_choice_keys = self._encoded_choice_keys()

            for keys in encoded_choice_keys.values():
                for encoded_key in keys:
                    self._encoded_boundaries[encoded_key] = bounds

        self._variable_names = [key for key, _ in self._iter_choices()]
        self._rng = np.random.default_rng(seed)

        if (
            self._decoder in {"gumbel-random-keys", "plackett-luce", "gumbel-sinkhorn"}
            and seed is None
        ):
            warnings.warn(
                "A stochastic decoding method was selected without a seed. "
                "Different evaluations may produce different results",
                RuntimeWarning,
                stacklevel=2,
            )

    def decode(
        self, float_inputs: dict[str, float]
    ) -> list[list[Any]] | dict[str, list[Any]]:
        self._check_input_bounds(float_inputs)

        decoders = {
            "random-keys": self._decode_random_keys,
            "gumbel-random-keys": self._decode_gumbel_random_keys,
            "plackett-luce": self._decode_plackett_luce,
            "gumbel-sinkhorn": self._decode_gumbel_sinkhorn,
        }

        decoder = decoders.get(self._decoder)

        if decoder is None:
            raise RuntimeError(f"Decoder {self._decoder!r} is not available.")

        decoded = decoder(float_inputs)

        if self._is_kwargs:
            return decoded

        return [decoded[str(index)] for index in range(len(self._boundaries))]

    def encode_cache(self, inputs: PermutationInput) -> CacheKey:
        if isinstance(inputs, dict):
            key = tuple(
                (name, tuple(values)) for name, values in sorted(inputs.items())
            )
        else:
            key = tuple(tuple(values) for values in inputs)

        if self._use_cache:
            try:
                hash(key)
            except TypeError as error:
                raise TypeError(
                    "Permutation values must be hashable when caching is enabled."
                ) from error

        return key

    def decode_cache(self, inputs: CacheKey) -> PermutationInput:
        if self._is_kwargs:
            return {
                name: list(values)
                for name, values in cast(
                    tuple[tuple[str, tuple[Any, ...]], ...], inputs
                )
            }

        return [list(values) for values in cast(tuple[tuple[Any, ...], ...], inputs)]

    def _decode_random_keys(
        self, float_inputs: dict[str, float]
    ) -> dict[str, list[Any]]:
        decoded = {}
        encoded_choice_keys = self._encoded_choice_keys()

        for key, choices in self._iter_choices():
            values = []

            for encoded_key in encoded_choice_keys[key]:
                values.append(float_inputs[encoded_key])

            order = np.argsort(values, kind="stable")
            decoded[key] = [choices[index] for index in order]

        return decoded

    def _decode_gumbel_random_keys(
        self, float_inputs: dict[str, float]
    ) -> dict[str, list[Any]]:
        temperature = self._params.get("temperature", 1.0)
        decoded = {}
        encoded_choice_keys = self._encoded_choice_keys()

        for key, choices in self._iter_choices():
            values = []

            for encoded_key in encoded_choice_keys[key]:
                values.append(float_inputs[encoded_key])

            gumbel_g = gumbel_r.rvs(size=len(choices), random_state=self._rng)
            scores = np.asarray(values) + temperature * gumbel_g
            order = np.argsort(scores, kind="stable")
            decoded[key] = [choices[index] for index in order]

        return decoded

    def _decode_plackett_luce(
        self, float_inputs: dict[str, float]
    ) -> dict[str, list[Any]]:
        temperature = self._params.get("temperature", 1.0)
        decoded = {}
        encoded_choice_keys = self._encoded_choice_keys()

        for key, choices in self._iter_choices():
            values = []

            for encoded_key in encoded_choice_keys[key]:
                values.append(float_inputs[encoded_key])

            remaining = list(range(len(choices)))
            permutation = []

            while remaining:
                remaining_values = np.asarray([values[index] for index in remaining])
                probabilities = softmax(remaining_values / temperature)
                selected = int(self._rng.choice(len(remaining), p=probabilities))
                permutation.append(remaining.pop(selected))

            decoded[key] = [choices[index] for index in permutation]

        return decoded

    def _decode_gumbel_sinkhorn(
        self, float_inputs: dict[str, float]
    ) -> dict[str, list[Any]]:
        temperature = self._params.get("temperature", 1.0)
        iterations = self._params.get("sinkhorn_iterations", 20)
        decoded = {}
        encoded_matrix_keys = self._encoded_matrix_keys()

        for key, choices in self._iter_choices():
            size = len(choices)
            values = np.empty((size, size), dtype=float)
            matrix_keys = encoded_matrix_keys[key]

            for row_index, row_keys in enumerate(matrix_keys):
                for column_index, matrix_key in enumerate(row_keys):
                    values[row_index, column_index] = float_inputs[matrix_key]

            gumbel_g = gumbel_r.rvs(size=(size, size), random_state=self._rng)
            matrix = (values + gumbel_g) / temperature
            matrix = self._sinkhorn(matrix, iterations)
            row_indices, column_indices = linear_sum_assignment(-matrix)
            order = np.argsort(column_indices, kind="stable")
            permutation = row_indices[order]
            decoded[key] = [choices[index] for index in permutation]

        return decoded

    @staticmethod
    def _sinkhorn(matrix: np.ndarray, iterations: int) -> np.ndarray:
        matrix = matrix - np.max(matrix)
        matrix = np.exp(matrix)

        for _ in range(iterations):
            row_sums = matrix.sum(axis=1, keepdims=True)
            matrix /= row_sums
            column_sums = matrix.sum(axis=0, keepdims=True)
            matrix /= column_sums

        return matrix

    def _encoded_choice_keys(self) -> dict[str, list[str]]:
        encoded_choice_keys = {}
        used_keys = set()

        for key, choices in self._iter_choices():
            keys = []

            for choice in choices:
                base_key = f"{key}-{choice}"
                encoded_key = self._unique_encoded_key(base_key, used_keys)
                used_keys.add(encoded_key)
                keys.append(encoded_key)

            encoded_choice_keys[key] = keys

        return encoded_choice_keys

    def _encoded_matrix_keys(self) -> dict[str, list[list[str]]]:
        encoded_matrix_keys = {}
        used_keys = set()

        for key, choices in self._iter_choices():
            matrix_keys = []

            for row in choices:
                row_keys = []

                for column in choices:
                    base_key = f"{key}-{row}-{column}"
                    encoded_key = self._unique_encoded_key(base_key, used_keys)
                    used_keys.add(encoded_key)
                    row_keys.append(encoded_key)

                matrix_keys.append(row_keys)

            encoded_matrix_keys[key] = matrix_keys

        return encoded_matrix_keys

    @staticmethod
    def _unique_encoded_key(base_key: str, used_keys: set[str]) -> str:
        encoded_key = base_key
        suffix = 1

        while encoded_key in used_keys:
            encoded_key = f"{base_key}-{suffix}"
            suffix += 1

        return encoded_key

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

        return not Permutation._is_nested_collection(first)

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
                    "Each permutation variable must be represented by "
                    "a one-dimensional collection of choices."
                )

            return value.tolist()

        if not isinstance(value, Iterable):
            raise TypeError(
                "Each permutation variable must be represented by "
                "an iterable collection of choices."
            )

        return list(value)
