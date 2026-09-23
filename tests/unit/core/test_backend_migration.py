"""Concrete contracts of the base migration driver."""

from __future__ import annotations

import json
from typing import final

import numpy as np
from numpy.random import SeedSequence

from tests._support.numerics import BASE_SEED
from tests._support.processors import (
    ProcessorMigrationDouble,
    ProcessorMigrationDriverDouble,
)
from tyrannis.backend.distributed.communication.no_communication import (
    NoCommunicationDriver,
    NoCommunicationProcessor,
)
from tyrannis.core.backend_communication import (
    CommunicationDriverBase,
    CommunicationProcessorBase,
)
from tyrannis.core.results import HistoryConfig


@final
class _RecordingCommunicationProcessor(NoCommunicationProcessor):
    def __init__(self, identifier: str, **kwargs: object) -> None:
        super().__init__(identifier, **kwargs)
        self.identifier = identifier
        self.kwargs = kwargs


class _RecordingMigrationProcessor(ProcessorMigrationDouble):
    def __init__(
        self,
        initial_iter: int,
        interval: int,
        communication_processor: CommunicationProcessorBase,
        seed: SeedSequence,
    ) -> None:
        super().__init__(initial_iter, communication_processor, seed)
        self.initial_iter: int = initial_iter
        self.interval: int = interval
        self.communication_processor: CommunicationProcessorBase = (
            communication_processor
        )
        self.seed: SeedSequence = seed


@final
class _ConfiguredDriver(ProcessorMigrationDriverDouble):
    def __init__(self) -> None:
        super().__init__()
        self._processor_class = _RecordingMigrationProcessor
        self._migration_processor_init_kargs = {"initial_iter": 3, "interval": 5}

    def context_state(
        self,
    ) -> tuple[
        CommunicationDriverBase,
        type[CommunicationProcessorBase],
        dict[str, object],
        HistoryConfig,
        int | None,
        SeedSequence,
        np.random.Generator,
        list[str],
    ]:
        return (
            self._communication_driver,
            self._communication_processor_class,
            self._migration_processor_init_kargs["communication"],
            self._history_config,
            self._seed,
            self._seed_sequence,
            self._rng,
            self._history_buffer,
        )


def _configured_driver(
    history_config: HistoryConfig | None = None,
) -> _ConfiguredDriver:
    driver = _ConfiguredDriver()
    driver.initialize_context(
        NoCommunicationDriver([]),
        _RecordingCommunicationProcessor,
        {"channel": "migration"},
        history_config if history_config is not None else HistoryConfig(),
        BASE_SEED,
    )
    return driver


def test_initialize_context_stores_copied_configuration_and_seeded_state() -> None:
    driver = _ConfiguredDriver()
    communication_driver = NoCommunicationDriver([])
    communication_kwargs = {"channel": "migration"}
    history_config = HistoryConfig(migration=True)

    driver.initialize_context(
        communication_driver,
        _RecordingCommunicationProcessor,
        communication_kwargs,
        history_config,
        BASE_SEED,
    )
    communication_kwargs["channel"] = "changed"

    (
        configured_communication_driver,
        communication_class,
        configured_kwargs,
        configured_history,
        configured_seed,
        seed_sequence,
        rng,
        history_buffer,
    ) = driver.context_state()

    assert configured_communication_driver is communication_driver
    assert communication_class is _RecordingCommunicationProcessor
    assert configured_kwargs == {"channel": "migration"}
    assert configured_history is history_config
    assert configured_seed == BASE_SEED
    assert (
        seed_sequence.generate_state(2).tolist()
        == SeedSequence(BASE_SEED).generate_state(2).tolist()
    )
    assert rng.random() == np.random.default_rng(BASE_SEED).random()
    assert history_buffer == []


def test_create_processor_module_separates_configs_and_spawns_independent_modules() -> (
    None
):
    driver = _configured_driver()

    first = driver.create_processor_module("island:a")
    second = driver.create_processor_module("island:b")

    assert isinstance(first, _RecordingMigrationProcessor)
    assert isinstance(second, _RecordingMigrationProcessor)
    assert first is not second
    assert isinstance(first.communication_processor, _RecordingCommunicationProcessor)
    assert isinstance(second.communication_processor, _RecordingCommunicationProcessor)
    assert first.communication_processor is not second.communication_processor
    assert first.communication_processor.identifier == "island:a"
    assert second.communication_processor.identifier == "island:b"
    assert first.communication_processor.kwargs == {"channel": "migration"}
    assert second.communication_processor.kwargs == {"channel": "migration"}
    assert (first.initial_iter, first.interval) == (3, 5)
    assert (second.initial_iter, second.interval) == (3, 5)

    first.communication_processor.kwargs["channel"] = "changed"
    assert second.communication_processor.kwargs == {"channel": "migration"}

    expected_first, expected_second = SeedSequence(BASE_SEED).spawn(2)
    assert first.seed is not second.seed
    assert (
        first.seed.generate_state(2).tolist()
        == expected_first.generate_state(2).tolist()
    )
    assert (
        second.seed.generate_state(2).tolist()
        == expected_second.generate_state(2).tolist()
    )
    assert (
        first.seed.generate_state(2).tolist() != second.seed.generate_state(2).tolist()
    )


def test_migration_log_skips_events_when_history_is_disabled() -> None:
    driver = _configured_driver(HistoryConfig(migration=False))

    driver.migration_log({"identifier": "p1"}, "island:a", "island:b", 4)

    assert driver.consume_migration_history() == []


def test_migration_log_records_events_and_consume_drains_the_batch() -> None:
    driver = _configured_driver(HistoryConfig(migration=True))
    particle = {"identifier": "p1", "fitness": 2.5}

    driver.migration_log(particle, "island:a", "island:b", 4)
    driver.migration_log(particle, "island:b", "island:c", 5)
    batch = driver.consume_migration_history()

    assert [json.loads(entry) for entry in batch] == [
        {
            "iteration": 4,
            "event": "0 - migration",
            "origin": "island:a",
            "destination": "island:b",
            "particle": particle,
        },
        {
            "iteration": 5,
            "event": "0 - migration",
            "origin": "island:b",
            "destination": "island:c",
            "particle": particle,
        },
    ]
    assert driver.consume_migration_history() == []
