from ..algorithm.base import ParticleBase


def get_fitness(particle: ParticleBase) -> float:
    if particle.fitness is None:
        raise RuntimeError(f"Particle '{particle.identifier}' does not have a fitness.")
    return particle.fitness
