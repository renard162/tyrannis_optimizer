"""Deterministic lifecycle probe shared by integration tests.

Trace travels with the particle, including through dump/reconstruction.
evaluation_count counts requests through ParticleBase.update (including failed
requests), not cache misses in a Space's fitness cache.
"""

import warnings
from collections.abc import Callable
from copy import deepcopy
from typing import final, override

import numpy as np

from tyrannis.core.algorithm import (
    FITNESS_UNDEFINED,
    AlgorithmBase,
    ParticleBase,
    Serializable,
)


@final
class IntegrationProbeParticle(ParticleBase):
    """Particle carrying JSON-compatible snapshots of processing phases."""

    def __init__(
        self,
        identifier: str,
        variables: dict[str, float],
        fitness: np.float64 = FITNESS_UNDEFINED,
        trace: list[dict[str, Serializable]] | None = None,
        evaluation_count: int = 0,
    ) -> None:
        super().__init__(identifier, dict(variables), fitness)
        self.trace = deepcopy(trace) if trace is not None else []
        self.evaluation_count = evaluation_count

    @override
    def __call__(self) -> dict[str, Serializable]:
        state = super().__call__()
        state["trace"] = [deepcopy(event) for event in self.trace]
        state["evaluation_count"] = self.evaluation_count
        return state

    @override
    def update(
        self,
        variables: dict[str, float],
        fitness_function: Callable[[dict[str, float]], np.float64],
    ) -> None:
        self.evaluation_count += 1
        super().update(variables, fitness_function)


@final
class IntegrationProbeAlgorithm(AlgorithmBase[IntegrationProbeParticle]):
    """Observe dispatch without an optimization policy or Space knowledge.

    Initial positions are encoded midpoints; supplied positions/fitness survive
    initialization. Updates interpolate within encoded bounds using cached
    randomness and consolidate unconditionally.
    double_eval selects every other sorted ID for the second phase.
    force_stop_at_iteration requests stop after consolidation; zero denotes
    initialization. The framework owns stopping and context reset.
    """

    @override
    def __init__(
        self, *, double_eval: bool = False, force_stop_at_iteration: int | None = None
    ) -> None:
        if force_stop_at_iteration is not None and (
            isinstance(force_stop_at_iteration, bool) or force_stop_at_iteration < 0
        ):
            raise ValueError("force_stop_at_iteration must be a non-negative integer.")
        self.double_eval = double_eval
        self.force_stop_at_iteration = force_stop_at_iteration
        self._actual_iter = 0
        self._second_phase = False

    @override
    def create_particle(
        self,
        identifier: str,
        variables: dict[str, float] | None = None,
        fitness: np.float64 = FITNESS_UNDEFINED,
        *args: object,
        trace: list[dict[str, Serializable]] | None = None,
        evaluation_count: int = 0,
        **kwargs: object,
    ) -> None:
        del args, kwargs
        if variables is None:
            variables = {
                name: lower / 2 + upper / 2
                for name, (lower, upper) in self._boundaries.items()
            }
        self._population[identifier] = IntegrationProbeParticle(
            identifier, variables, fitness, trace, evaluation_count
        )

    @override
    def delete_particle(self, identifier: str | None) -> None:
        if identifier is None:
            identifier = next(iter(sorted(self._population)), None)
        if identifier is not None:
            del self._population[identifier]

    @override
    def pre_iteration(self, actual_iter: int) -> None:
        self._actual_iter = actual_iter
        self._second_phase = False
        self._double_particle_check = self.double_eval
        self.double_check_ids = []

    @override
    def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
        phase = (
            "initialize"
            if initialize
            else ("second_update" if self._second_phase else "update")
        )
        for identifier in sorted(particle_ids):
            scalar = float(self._rng.random())
            array = self._rng.random(len(self._boundaries))
            # Four uint32 words provide 128 bits of fresh entropy from this
            # algorithm's advancing stream (including its island seed lineage).
            entropy = self._rng.integers(0, 4_294_967_296, size=4, dtype=np.uint32)
            seed_sequence = np.random.SeedSequence(entropy)
            self._population[identifier].random_cache = {
                "particle_id": identifier,
                "iteration": self._actual_iter,
                "phase": phase,
                "scalar": scalar,
                "array": array,
                "generator": np.random.default_rng(seed_sequence),
            }

    def _process(self, identifier: str, phase: str) -> IntegrationProbeParticle:
        particle = self._population[identifier]
        cache: dict[str, object] = particle.random_cache
        cache_iteration = cache["iteration"]
        assert isinstance(cache_iteration, int)
        scalar = cache["scalar"]
        raw_values = cache["array"]
        generator = cache["generator"]
        assert isinstance(scalar, float)
        assert isinstance(raw_values, np.ndarray)
        assert isinstance(generator, np.random.Generator)
        values = list(map(float, raw_values))
        # Bounded, data-dependent draws deliberately exercise the cached-generator
        # contract without an optimization or rejection loop.
        draws = [float(generator.random()) for _ in range(1 + int(3 * scalar))]
        evaluate = phase != "initialize" or particle.fitness == FITNESS_UNDEFINED
        event: dict[str, Serializable] = {
            "particle_id": identifier,
            "iteration": self._actual_iter,
            "phase": phase,
            "cache_particle_id": str(cache["particle_id"]),
            "cache_iteration": cache_iteration,
            "cache_phase": str(cache["phase"]),
            "scalar": scalar,
            "array": [value for value in values],
            "draws": [value for value in draws],
            "evaluation_count": particle.evaluation_count + int(evaluate),
        }
        particle.trace.append(event)
        if evaluate:
            variables = dict(particle.variables)
            if phase != "initialize":
                for index, (name, (lower, upper)) in enumerate(
                    self._boundaries.items()
                ):
                    fraction = (scalar + values[index] + draws[0]) / 3
                    variables[name] = min(
                        upper, max(lower, (1 - fraction) * lower + fraction * upper)
                    )
            particle.update(variables, self._fitness_function)
        return particle

    @override
    def initialize_particle(self, identifier: str) -> IntegrationProbeParticle:
        return self._process(identifier, "initialize")

    @staticmethod
    @override
    def consolidate_new_particles(
        particle: IntegrationProbeParticle,
    ) -> IntegrationProbeParticle:
        particle.consolidate(True)
        return particle

    @override
    def update_particle(self, identifier: str) -> IntegrationProbeParticle:
        return self._process(identifier, "update")

    @override
    def inter_iteration(self, actual_iter: int) -> None:
        del actual_iter
        self.double_check_ids = (
            sorted(self._population)[::2] if self.double_eval else []
        )
        self._second_phase = True

    @override
    def second_update_particle(self, identifier: str) -> IntegrationProbeParticle:
        return self._process(identifier, "second_update")

    @override
    def post_iteration(self, actual_iter: int) -> None:
        for particle in self._population.values():
            if particle.candidate_variables is not None:
                particle.consolidate(True)
        self.update_solution_state()
        if (
            self.force_stop_at_iteration is not None
            and actual_iter >= self.force_stop_at_iteration
            and not self.force_stop
        ):
            self._force_iteration_stop = True
            warnings.warn(
                f"Integration probe requested early stop after iteration {actual_iter}.",
                UserWarning,
                stacklevel=2,
            )
