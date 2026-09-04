from typing import Any

from tyrannis.core.algorithm import (
    AlgorithmBase,
    CostFunctionWrapperBase,
    ParticleBase,
)


class DummyCostFunctionWrapper(CostFunctionWrapperBase):
    pass


class DummyParticle(ParticleBase):
    pass


class DummyAlgorithm(AlgorithmBase):
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
            identifier = next(iter(self._population))

        del self._population[identifier]

    def pre_iteration(self, actual_iter: int) -> None:
        pass

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
        pass
