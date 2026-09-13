from collections.abc import Callable
from typing import Any

from ..core.space import SpaceBase
from . import SPACE_CLASSES


class Mixed(SpaceBase):
    """Mixed optimization search space."""

    _spaces: list[SpaceBase]
    _space_variables: list[list[str]]

    def __init__(
        self,
        spaces: dict[str, SpaceBase],
        cost_function: Callable[..., float] | None = None,
        use_cache: bool = False,
        cache_type: str = "lru",
        cache_size: int = 100_000,
    ) -> None:
        """
        Initialize the user-facing mixed search space.

        Parameters
        ----------
        spaces:
            Dictionary associating each input name with its corresponding
            search-space instance.
        cost_function:
            User-defined cost function to be evaluated after decoding the
            solver representation.
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

        self._type = "mixed"
        self._configs = {}
        self._spaces = []
        self._space_variables = []

        grouped_spaces: dict[tuple[str, Any], dict[str, Any]] = {}

        for key, space in spaces.items():
            arguments = space.input_arguments
            space_type = arguments["space"]
            configs = arguments["configs"]

            group_key = (space_type, tuple(sorted(configs.items())))

            if group_key not in grouped_spaces:
                grouped_spaces[group_key] = {
                    "space": space_type,
                    "configs": configs.copy(),
                    "boundaries": [],
                    "variables": [],
                }

            group = grouped_spaces[group_key]
            group["variables"].append(key)

            if space_type == "binary":
                group["boundaries"].append(key)
            else:
                group["boundaries"].append((key, arguments["boundaries"][key]))

        for group in grouped_spaces.values():
            space_class = SPACE_CLASSES[group["space"]]

            if group["space"] == "binary":
                boundaries = group["boundaries"]
            else:
                boundaries = dict(group["boundaries"])

            space = space_class(
                boundaries, cost_function=cost_function, **group["configs"]
            )

            self._spaces.append(space)
            self._space_variables.append(group["variables"])

    def initialize_context(self, seed: int | None = None) -> None:
        """
        Initialize the execution context of all component spaces.
        """
        self._encoded_boundaries = {}

        for space in self._spaces:
            space.initialize_context(seed)
            self._encoded_boundaries.update(space.encoded_boundaries)

    def decode(self, float_inputs: dict[str, float]) -> dict[str, Any]:
        """
        Decode the mixed solver representation into the user representation.
        """
        decoded = {}

        for space, variables in zip(self._spaces, self._space_variables, strict=True):
            space_inputs = {key: float_inputs[key] for key in variables}
            decoded.update(space.decode(space_inputs))

        return decoded

    def encode_cache(self, inputs: dict[str, Any]) -> tuple[Any, ...]:
        """
        Encode the decoded mixed representation into a canonical cache key.
        """
        return tuple(
            space.encode_cache({key: inputs[key] for key in variables})
            for space, variables in zip(
                self._spaces, self._space_variables, strict=True
            )
        )

    def decode_cache(self, inputs: tuple[Any, ...]) -> dict[str, Any]:
        """
        Decode a canonical mixed cache key into the user representation.
        """
        decoded = {}

        for space, cache_key in zip(self._spaces, inputs, strict=True):
            decoded.update(space.decode_cache(cache_key))

        return decoded
