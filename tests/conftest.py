from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tyrannis.core.algorithm import (
    AlgorithmBase,
    CostFunctionWrapperBase,
    ParticleBase,
)
from tyrannis.core.backend import BackendBase
from tyrannis.core.processor import (
    LocalEvent,
    ProcessorBase,
)


# ---------------------------------------------------------------------------
# Generic test data
# ---------------------------------------------------------------------------


@pytest.fixture
def fitness_function() -> Callable[[dict[str, float]], float]:
    """Deterministic fitness function used by generic tests."""

    def sphere(variables: dict[str, float]) -> float:
        return sum(value**2 for value in variables.values())

    return sphere


@pytest.fixture
def boundaries() -> dict[str, tuple[float, float]]:
    """Default search-space boundaries used by generic tests."""

    return {
        "x": (-10.0, 10.0),
        "y": (-10.0, 10.0),
    }


@pytest.fixture
def variables() -> dict[str, float]:
    """Default particle variables."""

    return {
        "x": 1.0,
        "y": 2.0,
    }


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class DummyCostFunctionWrapper(CostFunctionWrapperBase):
    """Minimal cost-function wrapper for testing base classes."""


class DummyParticle(ParticleBase):
    """Minimal concrete implementation of ParticleBase."""


class DummyAlgorithm(AlgorithmBase):
    """Minimal concrete implementation of AlgorithmBase."""

    def __init__(self) -> None:
        pass

    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: float | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        if variables is None:
            variables = {name: 0.0 for name in self._boundaries}

        self._population[identifier] = DummyParticle(
            identifier=identifier,
            variables=variables,
            fitness=fitness,
        )

    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            raise ValueError("Particle identifier cannot be None.")

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        return

    def initialize_particle(self, identifier: str) -> ParticleBase:
        particle = self._population[identifier]

        if particle.fitness is None:
            particle.update(
                variables=particle.variables,
                fitness_function=self._fitness_function,
            )

        particle.consolidate(consolidate_new=True)

        return particle

    def update_particle(self, identifier: str) -> ParticleBase:
        particle = self._population[identifier]

        particle.update(
            variables=particle.variables,
            fitness_function=self._fitness_function,
        )

        return particle

    def post_iteration(self, actual_iter: int) -> None:
        self.update_solution_state()


class DummyProcessor(ProcessorBase[LocalEvent]):
    """Minimal concrete implementation of ProcessorBase."""

    def __init__(self) -> None:
        self._cost_function_wrapper = DummyCostFunctionWrapper

    def initialize_execution_context(self) -> None:
        self._stop_signal = LocalEvent()
        self._wait_signal = LocalEvent()

    def finalize_execution_context(self) -> None:
        self._stop_signal = LocalEvent()
        self._wait_signal = LocalEvent()

    def run(self) -> None:
        self.init_particles()

        for actual_iter in range(self._n_iter + 1):
            self.update_iter_counter(actual_iter)

            if self._stop_signal.is_set():
                break

            self.migration_control()
            self._algorithm.pre_iteration(actual_iter)

            for particle_id in self._algorithm.new_particles_id:
                self._algorithm.initialize_particle(particle_id)

            if actual_iter > 0:
                for particle_id in self._algorithm.population:
                    self._algorithm.update_particle(particle_id)

            self._algorithm.post_iteration(actual_iter)
            self.update_status()


class DummyBackend(BackendBase):
    """Minimal concrete implementation of BackendBase."""

    def __init__(self) -> None:
        self._identifier = "DummyBackend"
        self._cost_function_wrapper = DummyCostFunctionWrapper

    def execute(self) -> None:
        return


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def particle(
    variables: dict[str, float],
    fitness_function: Callable[[dict[str, float]], float],
) -> DummyParticle:
    """Create a basic particle with a known fitness."""

    return DummyParticle(
        identifier="particle:0",
        variables=variables.copy(),
        fitness=fitness_function(variables),
    )


@pytest.fixture
def algorithm(
    fitness_function: Callable[[dict[str, float]], float],
    boundaries: dict[str, tuple[float, float]],
) -> DummyAlgorithm:
    """Create an initialized DummyAlgorithm."""

    algorithm = DummyAlgorithm()

    algorithm.initialize_context(
        fitness_function=fitness_function,
        boundaries=boundaries,
    )

    algorithm.configure(
        identifier="DummyAlgorithm",
        cost_function_wrapper=DummyCostFunctionWrapper,
        seed=42,
    )

    return algorithm


@pytest.fixture
def processor() -> DummyProcessor:
    """Create a processor without an execution context."""

    return DummyProcessor()


@pytest.fixture
def initialized_processor(
    processor: DummyProcessor,
    algorithm: DummyAlgorithm,
) -> DummyProcessor:
    """Create a processor with its optimization context initialized."""

    processor.initialize_context(
        algorithm=algorithm,
        n_iter=2,
        n_particles=4,
        seed=42,
    )

    return processor


@pytest.fixture
def backend() -> DummyBackend:
    """Create a basic backend."""

    return DummyBackend()


@pytest.fixture
def initialized_backend(
    backend: DummyBackend,
    algorithm: DummyAlgorithm,
    processor: DummyProcessor,
) -> DummyBackend:
    """Create a backend with its optimization context initialized."""

    backend.initialize_context(
        algorithm=algorithm,
        n_iter=2,
        n_particles=4,
        processor=processor,
        seed=42,
    )

    return backend


# ---------------------------------------------------------------------------
# Spark
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def spark():
    """
    Create a local Spark session for integration tests.

    PySpark is an optional Tyrannis dependency, so tests using this fixture
    are skipped automatically when the Spark extra is not installed.
    """

    pyspark = pytest.importorskip("pyspark")
    from pyspark.sql import SparkSession

    spark = (
        SparkSession.builder.master("local[2]")
        .appName("tyrannis-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("ERROR")

    yield spark

    spark.stop()
