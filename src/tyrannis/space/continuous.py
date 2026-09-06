from collections.abc import Callable
from typing import TypeAlias

from tyrannis.core.space import SpaceBase

Boundary: TypeAlias = tuple[float, float]
Boundaries: TypeAlias = list[Boundary] | dict[str, Boundary]


class Continuous(SpaceBase):
    """Continuous optimization search space."""

    _boundaries: Boundaries

    def __init__(
        self,
        cost_function: Callable[..., float] | None,
        boundaries: Boundaries,
    ) -> None:
        """
        Initialize the user-facing continuous search space.

        The cost function is supplied by the user and is evaluated after the
        solver representation has been decoded. The search-space boundaries
        can be provided either as a list of lower and upper limits or as a
        dictionary associating each variable with its limits.

        When a list is provided, each tuple represents the lower and upper
        limits of one positional input of the cost function. When a
        dictionary is provided, each key identifies one input of the cost
        function and its associated tuple contains its lower and upper
        limits.

        Parameters
        ----------
        cost_function:
            User-defined cost function to be evaluated after decoding the
            solver inputs. It may be ``None`` during construction, but
            ``initialize_context`` requires a valid cost function.
        boundaries:
            Search-space boundaries. A list contains one ``(lower, upper)``
            tuple for each positional input. A dictionary maps each input
            name to its ``(lower, upper)`` tuple.
        """
        self._cost_function = cost_function
        self._boundaries = boundaries

    def initialize_context(self, seed: int) -> None:
        if isinstance(self._boundaries, dict):
            self._encoded_boundaries = self._boundaries.copy()
            return

        self._encoded_boundaries = {
            str(index): boundary for index, boundary in enumerate(self._boundaries)
        }

    def decode(self, float_inputs: dict[str, float]) -> list[float] | dict[str, float]:
        for key, value in float_inputs.items():
            if key not in self._encoded_boundaries:
                raise KeyError(f"Unknown continuous-space variable: {key!r}.")

            lower, upper = self._encoded_boundaries[key]

            if not lower <= value <= upper:
                raise ValueError(
                    f"Value {value} for variable {key!r} is outside "
                    f"the boundaries ({lower}, {upper})."
                )

        if isinstance(self._boundaries, dict):
            return float_inputs

        return [float_inputs[str(index)] for index in range(len(self._boundaries))]

    @property
    def is_kargs(self) -> bool:
        return isinstance(self._boundaries, dict)
