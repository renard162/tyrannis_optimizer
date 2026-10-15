"""Real processors must preserve the deterministic probe's serial semantics."""

import json
from collections.abc import Callable, Sequence
from copy import deepcopy
from functools import partial
from itertools import pairwise

import pytest
from _support.assertions import (
    assert_optimizer_result_consistent,
    assert_variables_within_bounds,
)

# The shared factory's AlgorithmBase annotation omits its type parameter.
from _support.factories import (
    make_optimizer,  # pyright: ignore[reportUnknownVariableType]
)
from _support.integration_probe import IntegrationProbeAlgorithm
from _support.numerics import BASE_SEED
from _support.objectives import sphere

import tyrannis.processor as processor_module
from tyrannis import Optimizer
from tyrannis.backend.local import Local
from tyrannis.core.algorithm import Serializable
from tyrannis.core.processor import ProcessorBase
from tyrannis.space.continuous import Continuous

ProcessorFactory = Callable[
    [],
    processor_module.Joblib
    | processor_module.ProcessPool
    | processor_module.ThreadsPool,
]
History = dict[tuple[int, str], dict[str, Serializable]]
N_PARTICLES = 4
# A positive interval lets updates improve on initialization's midpoint, so
# result propagation is observable too (a symmetric sphere starts at its best).
BOUNDS = {"x": (1.0, 3.0)}
PROCESSOR_FACTORIES: dict[str, ProcessorFactory] = {
    "Joblib": partial(
        processor_module.Joblib, n_jobs=2, joblib_backend="loky", batch_size=1
    ),
    "ProcessPool": partial(processor_module.ProcessPool, n_jobs=2, chunksize=1),
    "ThreadsPool": partial(processor_module.ThreadsPool, n_jobs=2, chunksize=1),
}


def _processor_cases() -> Sequence[object]:
    cases = [pytest.param(None, id="Serial")]
    for name in processor_module.__all__:
        processor_class = getattr(processor_module, name, None)
        if not isinstance(processor_class, type) or not issubclass(
            processor_class, ProcessorBase
        ):
            raise TypeError(
                f"Public processor {name!r} must be a ProcessorBase subclass."
            )
        if name not in PROCESSOR_FACTORIES:
            raise AssertionError(
                f"Public processor {name!r} needs a smoke configuration."
            )
        factory = PROCESSOR_FACTORIES[name]
        assert isinstance(factory(), processor_class)
        marks = (pytest.mark.multiprocess,) if name in {"Joblib", "ProcessPool"} else ()
        cases.append(pytest.param(factory, id=name, marks=marks))
    return cases


PROCESSORS = _processor_cases()


def _optimizer(
    processor_factory: ProcessorFactory | None,
    *,
    double_eval: bool = False,
    force_stop_at_iteration: int | None = None,
) -> Optimizer:
    return make_optimizer(
        space=Continuous(boundaries=BOUNDS.copy(), cost_function=sphere),
        algorithm=IntegrationProbeAlgorithm(
            double_eval=double_eval, force_stop_at_iteration=force_stop_at_iteration
        ),
        backend=Local(),
        processor=None if processor_factory is None else processor_factory(),
        history="iteration",
        n_particles=N_PARTICLES,
        n_iterations=2 if force_stop_at_iteration is None else 3,
        seed=BASE_SEED,
    )


def _history(optimizer: Optimizer) -> History:
    """Index controller snapshots, rejecting duplicates before normalization."""
    history: History = {}
    for values in optimizer.internal_history_generator_:
        row: dict[str, object] = dict(
            zip(optimizer.internal_history_columns_, values, strict=True)
        )
        iteration, identifier = row["iteration"], row["identifier"]
        assert isinstance(iteration, int)
        assert isinstance(identifier, str)
        key = (iteration, identifier)
        assert key not in history, f"Duplicate particle snapshot: {key}"
        internal_state = row["internal_state"]
        assert isinstance(internal_state, str)
        # json.loads has an untyped return; the probe's dump is JSON-compatible.
        state: dict[str, Serializable] = json.loads(internal_state)  # pyright: ignore[reportAny]
        variables: dict[str, Serializable] = {}
        for name in BOUNDS:
            value = row[name]
            assert isinstance(value, float)
            variables[name] = value
        fitness = row["fitness"]
        assert isinstance(fitness, float)
        history[key] = dict(
            state, identifier=identifier, variables=variables, fitness=fitness
        )
    return history


def _assert_lifecycle(history: History, *, last_iteration: int) -> None:
    identifiers = {identifier for iteration, identifier in history if iteration == 0}
    assert len(identifiers) == N_PARTICLES
    assert set(history) == {
        (iteration, identifier)
        for iteration in range(last_iteration + 1)
        for identifier in identifiers
    }
    for (iteration, identifier), state in history.items():
        trace = state["trace"]
        assert isinstance(trace, list)
        assert state["evaluation_count"] == len(trace)
        for count, event in enumerate(trace, start=1):
            assert isinstance(event, dict)
            assert event["particle_id"] == event["cache_particle_id"] == identifier
            assert event["iteration"] == event["cache_iteration"]
            assert event["phase"] == event["cache_phase"]
            assert event["evaluation_count"] == count
        # Complete trace equality below also checks materialized values and draws.
        first_phases = [
            (event["iteration"], event["phase"])
            for event in trace
            if isinstance(event, dict) and event["phase"] != "second_update"
        ]
        assert first_phases == [(0, "initialize")] + [
            (step, "update") for step in range(1, iteration + 1)
        ]
        if iteration:
            previous = history[iteration - 1, identifier]["trace"]
            assert isinstance(previous, list)
            assert trace[: len(previous)] == previous
            # A fresh stream must reach each new phase, including second_update.
            for before, after in pairwise(trace):
                assert isinstance(before, dict)
                assert isinstance(after, dict)
                assert before["draws"] != after["draws"]


def _assert_result(optimizer: Optimizer, reference: Optimizer) -> dict[str, object]:
    """Check the public result and retain a detached snapshot for repeated fit."""
    assert_optimizer_result_consistent(optimizer)
    # Optimizer.best_solution is Any because decoded Space representations vary.
    solution: dict[str, float] = optimizer.best_solution  # pyright: ignore[reportAny]
    assert isinstance(solution, dict)
    assert_variables_within_bounds(solution, BOUNDS)
    assert sphere(**solution) == optimizer.best_fitness
    assert optimizer.result_ == reference.result_
    assert_optimizer_result_consistent(reference)
    assert optimizer.best_fitness == reference.best_fitness
    history = _history(optimizer)
    fitnesses: list[float] = []
    for state in history.values():
        fitness = state["fitness"]
        assert isinstance(fitness, float)
        fitnesses.append(fitness)
    assert optimizer.best_fitness == min(fitnesses)
    assert optimizer.best_fitness < sphere(x=2.0)
    return deepcopy(
        {
            "result": optimizer.result_,
            "best_solution": solution,
            "best_fitness": optimizer.best_fitness,
        }
    )


@pytest.mark.parametrize("processor_factory", PROCESSORS)
def test_single_evaluation_preserves_serial_lifecycle(
    processor_factory: ProcessorFactory | None,
) -> None:
    reference = _optimizer(None).fit()
    optimizer = _optimizer(processor_factory).fit()
    history = _history(optimizer)
    _assert_lifecycle(history, last_iteration=2)
    assert history == _history(reference)
    for (iteration, _), state in history.items():
        assert state["evaluation_count"] == iteration + 1
    _ = _assert_result(optimizer, reference)


@pytest.mark.parametrize("processor_factory", PROCESSORS)
def test_second_evaluation_returns_only_selected_particles(
    processor_factory: ProcessorFactory | None,
) -> None:
    reference = _optimizer(None, double_eval=True).fit()
    optimizer = _optimizer(processor_factory, double_eval=True).fit()
    expected = _history(reference)
    history = _history(optimizer)
    _assert_lifecycle(history, last_iteration=2)
    assert history == expected
    # Derive selection from Serial's trace, not the probe's concrete ID rule.
    for iteration in (1, 2):
        selected: set[str] = set()
        for (step, identifier), state in expected.items():
            if step != iteration:
                continue
            trace = state["trace"]
            assert isinstance(trace, list)
            event = trace[-1]
            assert isinstance(event, dict)
            if event["phase"] == "second_update":
                selected.add(identifier)
        assert 0 < len(selected) < N_PARTICLES
        for (step, identifier), state in history.items():
            if step != iteration:
                continue
            previous_count = history[step - 1, identifier]["evaluation_count"]
            assert isinstance(previous_count, int)
            assert state["evaluation_count"] == previous_count + 1 + int(
                identifier in selected
            )
            # A returned second-update particle must replace the first version;
            # full state equality above checks its variables and fitness too.
    _ = _assert_result(optimizer, reference)


@pytest.mark.parametrize("processor_factory", PROCESSORS)
def test_force_stop_finishes_iteration_and_remains_latched_on_repeated_fit(
    processor_factory: ProcessorFactory | None,
) -> None:
    reference = _optimizer(None, force_stop_at_iteration=1)
    optimizer = _optimizer(processor_factory, force_stop_at_iteration=1)
    for run in (reference, optimizer):
        with pytest.warns(
            UserWarning,
            match="Integration probe requested early stop after iteration 1",
        ):
            _ = run.fit()
    history = _history(optimizer)
    _assert_lifecycle(history, last_iteration=1)
    assert history == _history(reference)
    for (iteration, _), state in history.items():
        assert state["evaluation_count"] == iteration + 1
    public_state = _assert_result(optimizer, reference)

    # History is freshly decoded and public_state is deep-copied: in-place
    # mutation cannot change the expected state alongside the observed state.
    _ = optimizer.fit()
    assert _history(optimizer) == history
    assert _assert_result(optimizer, reference) == public_state
