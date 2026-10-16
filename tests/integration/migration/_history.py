"""Small views of the real, consolidated migration and population history."""

from typing import NotRequired, TypedDict

from tyrannis.core.algorithm import Serializable
from tyrannis.core.results import HistoryConfig


class ParticleSnapshot(TypedDict):
    identifier: str
    fitness: float
    trace: list[dict[str, Serializable]]
    evaluation_count: int


class HistoryRecord(TypedDict):
    event: str
    origin: str
    destination: NotRequired[str]
    iteration: int
    particle: ParticleSnapshot


def population_at(
    records: list[HistoryRecord], island: str, iteration: int
) -> dict[str, ParticleSnapshot]:
    """Read membership after an iteration, without inspecting worker state."""
    particles = [
        record["particle"]
        for record in records
        if record["event"] == HistoryConfig.get_event("iteration")
        and record["origin"] == island
        and record["iteration"] == iteration
    ]
    population = {particle["identifier"]: particle for particle in particles}
    assert len(population) == len(particles), "Duplicate particle in an island snapshot"
    return population
