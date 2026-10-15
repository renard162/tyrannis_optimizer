"""Public Spaces preserve decoded inputs and results through a real optimization."""

import json
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeGuard

import pytest
from _support.assertions import (
    assert_optimizer_result_consistent,
    assert_valid_permutation,
    assert_variables_within_bounds,
)

# The shared factory omits AlgorithmBase's type parameter; its implementation
# and the concrete probe are reused without weakening their runtime contracts.
from _support.factories import (
    make_optimizer,  # pyright: ignore[reportUnknownVariableType]
)
from _support.integration_probe import IntegrationProbeAlgorithm
from _support.numerics import BASE_SEED
from _support.objectives import sphere

import tyrannis.space as space_module
from tyrannis import Optimizer
from tyrannis.core.algorithm import Serializable
from tyrannis.core.space import SpaceBase

CONTINUOUS_BOUNDS = {"x": (1.0, 3.0)}
INTEGER_BOUNDS = {"count": (1, 4)}
COLORS = ("red", "green", "blue")
LEVELS = ("low", "medium", "high")
MIN_CHOICES = 1
MAX_CHOICES = 3
N_PARTICLES = 2


@dataclass(frozen=True, kw_only=True)
class SpaceSmokeCase:
    """An explicit constructor and an objective validating inputs AND output.

    The same strict public objective checks best_solution and recomputes its
    fitness. Constructors keep their native, heterogeneous configuration.
    """

    identifier: str
    space_class: type[SpaceBase]
    build_space: Callable[[Callable[..., float]], SpaceBase]
    objective: Callable[..., float]


def _continuous_cost(*, x: object) -> float:
    assert isinstance(x, float)
    assert_variables_within_bounds({"x": x}, CONTINUOUS_BOUNDS)
    return sphere(x=x)


def _integer_cost(*, count: object) -> float:
    assert isinstance(count, int) and not isinstance(count, bool)
    assert_variables_within_bounds({"count": count}, INTEGER_BOUNDS)
    return float(count)


def _binary_cost(*, enabled: object) -> float:
    assert isinstance(enabled, bool)
    return float(enabled)


def _categorical_cost(*, color: object) -> float:
    assert isinstance(color, str)
    assert color in COLORS
    return float(COLORS.index(color))


def _is_public_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


def _permutation_cost(*, order: object) -> float:
    assert _is_public_list(order)
    assert_valid_permutation(order, COLORS)
    return _categorical_cost(color=order[0])


def _ordinal_cost(*, level: object) -> float:
    assert isinstance(level, str)
    assert level in LEVELS
    return float(LEVELS.index(level))


def _sequence_cost(*, sequence: object) -> float:
    assert _is_public_list(sequence)
    assert MIN_CHOICES <= len(sequence) <= MAX_CHOICES
    return sum(_categorical_cost(color=value) for value in sequence)


def _mixed_cost(*, x: object, count: object, color: object) -> float:
    return (
        _continuous_cost(x=x)
        + _integer_cost(count=count)
        + _categorical_cost(color=color)
    )


# All current default decoders are deterministic; no decoder matrix is needed.
SPACE_CASES = (
    SpaceSmokeCase(
        identifier="Continuous",
        space_class=space_module.Continuous,
        build_space=lambda objective: space_module.Continuous(
            boundaries=CONTINUOUS_BOUNDS.copy(),
            cost_function=objective,
            use_cache=False,
        ),
        objective=_continuous_cost,
    ),
    SpaceSmokeCase(
        identifier="Integer",
        space_class=space_module.Integer,
        build_space=lambda objective: space_module.Integer(
            boundaries=INTEGER_BOUNDS.copy(),
            cost_function=objective,
            use_cache=False,
        ),
        objective=_integer_cost,
    ),
    SpaceSmokeCase(
        identifier="Binary",
        space_class=space_module.Binary,
        build_space=lambda objective: space_module.Binary(
            bits=["enabled"], cost_function=objective, use_cache=False
        ),
        objective=_binary_cost,
    ),
    SpaceSmokeCase(
        identifier="Categorical",
        space_class=space_module.Categorical,
        build_space=lambda objective: space_module.Categorical(
            choices={"color": COLORS}, cost_function=objective, use_cache=False
        ),
        objective=_categorical_cost,
    ),
    SpaceSmokeCase(
        identifier="Permutation",
        space_class=space_module.Permutation,
        build_space=lambda objective: space_module.Permutation(
            choices={"order": COLORS}, cost_function=objective, use_cache=False
        ),
        objective=_permutation_cost,
    ),
    SpaceSmokeCase(
        identifier="Ordinal",
        space_class=space_module.Ordinal,
        build_space=lambda objective: space_module.Ordinal(
            choices={"level": LEVELS}, cost_function=objective, use_cache=False
        ),
        objective=_ordinal_cost,
    ),
    SpaceSmokeCase(
        identifier="Sequence",
        space_class=space_module.Sequence,
        build_space=lambda objective: space_module.Sequence(
            choices={"sequence": COLORS},
            min_choices=MIN_CHOICES,
            max_choices=MAX_CHOICES,
            cost_function=objective,
            use_cache=False,
        ),
        objective=_sequence_cost,
    ),
    SpaceSmokeCase(
        identifier="Mixed",
        space_class=space_module.Mixed,
        build_space=lambda objective: space_module.Mixed(
            spaces={
                "x": space_module.Continuous(
                    boundaries=CONTINUOUS_BOUNDS.copy(), use_cache=False
                ),
                "count": space_module.Integer(
                    boundaries=INTEGER_BOUNDS.copy(), use_cache=False
                ),
                "color": space_module.Categorical(choices=COLORS, use_cache=False),
            },
            cost_function=objective,
            use_cache=False,
        ),
        objective=_mixed_cost,
    ),
)


def _check_public_space_coverage() -> None:
    """Fail collection clearly when the public API and smoke cases diverge."""
    public_names = set(space_module.__all__)
    case_counts = Counter(case.identifier for case in SPACE_CASES)
    missing = public_names - case_counts.keys()
    obsolete = case_counts.keys() - public_names
    assert not missing, (
        f"Public Spaces {sorted(missing)} need a minimal integration case in SPACE_CASES."
    )
    assert not obsolete, f"Smoke cases reference non-public Spaces: {sorted(obsolete)}"
    duplicates = [name for name, count in case_counts.items() if count != 1]
    assert not duplicates, f"Duplicate Space smoke cases: {duplicates}"
    assert len({case.space_class for case in SPACE_CASES}) == len(SPACE_CASES), (
        "Each Space class must have exactly one main smoke case."
    )
    for case in SPACE_CASES:
        public_class = getattr(space_module, case.identifier, None)
        assert isinstance(public_class, type), (
            f"Public Space {case.identifier!r} must exist and be a class."
        )
        assert issubclass(public_class, SpaceBase), (
            f"Public Space {case.identifier!r} must derive from SpaceBase."
        )
        assert public_class is case.space_class


_check_public_space_coverage()


def _run_smoke(space: SpaceBase) -> Optimizer:
    # Defaults are Local + Serial. One iteration includes initialization and
    # one real ParticleBase.update followed by a new objective evaluation.
    optimizer = make_optimizer(
        space=space,
        algorithm=IntegrationProbeAlgorithm(
            double_eval=False, force_stop_at_iteration=None
        ),
        n_particles=N_PARTICLES,
        n_iterations=1,
        seed=BASE_SEED,
        history="iteration",
        fitness_failure_strategy="raise",
    ).fit()
    assert_optimizer_result_consistent(optimizer)

    # Local runs a replica: observe the returned population, not the template.
    snapshots: dict[tuple[int, str], dict[str, float]] = {}
    fitnesses: list[float] = []
    for values in optimizer.internal_history_generator_:
        row: dict[str, object] = dict(
            zip(optimizer.internal_history_columns_, values, strict=True)
        )
        iteration, identifier = row["iteration"], row["identifier"]
        assert isinstance(iteration, int)
        assert isinstance(identifier, str)
        key = (iteration, identifier)
        assert key not in snapshots
        variables: dict[str, float] = {}
        for name in space.encoded_boundaries:
            value = row[name]
            assert isinstance(value, float)
            variables[name] = value
        assert_variables_within_bounds(variables, space.encoded_boundaries)
        snapshots[key] = variables
        fitness = row["fitness"]
        assert isinstance(fitness, float)
        fitnesses.append(fitness)

        internal_state = row["internal_state"]
        assert isinstance(internal_state, str)
        # json.loads is untyped; narrow the probe's documented serializable state.
        state: dict[str, Serializable] = json.loads(internal_state)  # pyright: ignore[reportAny]
        assert state["evaluation_count"] == iteration + 1
        trace = state["trace"]
        assert isinstance(trace, list)
        phases: list[Serializable] = []
        for event in trace:
            assert isinstance(event, dict)
            phases.append(event["phase"])
        assert phases == ["initialize"] + ["update"] * iteration

    identifiers = {identifier for _, identifier in snapshots}
    assert len(identifiers) == N_PARTICLES
    assert set(snapshots) == {
        (iteration, identifier) for iteration in (0, 1) for identifier in identifiers
    }
    assert any(
        snapshots[0, identifier] != snapshots[1, identifier]
        for identifier in identifiers
    )
    assert optimizer.best_fitness == min(fitnesses)
    return optimizer


@pytest.mark.parametrize(
    "case", SPACE_CASES, ids=[case.identifier for case in SPACE_CASES]
)
def test_space_smoke(case: SpaceSmokeCase) -> None:
    space = case.build_space(case.objective)
    assert type(space) is case.space_class
    optimizer = _run_smoke(space)
    # The public API returns Any because each Space defines its representation.
    solution: dict[str, object] = optimizer.best_solution  # pyright: ignore[reportAny]
    assert isinstance(solution, dict)
    assert case.objective(**solution) == optimizer.best_fitness


def test_continuous_positional_smoke() -> None:
    def objective(x: object, y: object, /) -> float:
        # Positional-only parameters reject accidental keyword dispatch.
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert_variables_within_bounds({"x": x, "y": y}, bounds)
        return sphere(x, y)

    bounds = {"x": (1.0, 3.0), "y": (4.0, 6.0)}
    optimizer = _run_smoke(
        space_module.Continuous(
            boundaries=list(bounds.values()),
            cost_function=objective,
            use_cache=False,
        )
    )
    # The public API returns Any; this convention specifically promises a list.
    solution: list[object] = optimizer.best_solution  # pyright: ignore[reportAny]
    assert isinstance(solution, list)
    assert len(solution) == 2
    assert objective(*solution) == optimizer.best_fitness
