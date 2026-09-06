from unittest.mock import Mock

import numpy as np
import pytest

from tyrannis.backend.processor.process import (
    ProcessPool,
    ProcessPoolCostFunctionWrapper,
    StopSignal,
)
from tyrannis.core.backend_migration import MigrationProcessorBase
from tyrannis.core.signals import LocalEvent


def sphere(variables: dict[str, float]) -> float:
    values = np.fromiter(variables.values(), dtype=float)

    return float(np.sum(values**2))


def create_migration_processor() -> Mock:
    return Mock(spec=MigrationProcessorBase)


def test_cost_function_wrapper() -> None:
    wrapper = ProcessPoolCostFunctionWrapper(sphere)

    result = wrapper({"x": 3.0, "y": 4.0})

    np.testing.assert_allclose(result, 25.0)


def test_cost_function_wrapper_serialization() -> None:
    wrapper = ProcessPoolCostFunctionWrapper(sphere)

    restored = ProcessPoolCostFunctionWrapper._restore(
        wrapper._serialized_function,
    )

    result = restored({"x": 3.0, "y": 4.0})

    np.testing.assert_allclose(result, 25.0)


def test_stop_signal() -> None:
    signal = StopSignal()

    assert not signal.is_set()

    signal.set()

    assert signal.is_set()

    signal.clear()

    assert not signal.is_set()


def test_stop_signal_manager_signal() -> None:
    signal = StopSignal()

    manager_signal = StopSignal()

    signal.set_manager_signal(manager_signal)  # type: ignore

    signal.set()

    assert signal.is_set()
    assert manager_signal.is_set()

    signal.clear()

    assert not signal.is_set()
    assert not manager_signal.is_set()


def test_stop_signal_manager_signal_reflects_current_state() -> None:
    signal = StopSignal()

    signal.set()

    manager_signal = StopSignal()

    signal.set_manager_signal(manager_signal)  # type: ignore

    assert manager_signal.is_set()

    signal.clear()

    assert not manager_signal.is_set()


def test_stop_signal_clear_manager_signal() -> None:
    signal = StopSignal()

    manager_signal = StopSignal()

    signal.set_manager_signal(manager_signal)  # type: ignore
    signal.clear_manager_signal()

    signal.set()

    assert signal.is_set()
    assert not manager_signal.is_set()


def test_stop_signal_manager_signal_requires_initialization() -> None:
    signal = StopSignal()

    with pytest.raises(
        RuntimeError,
        match="Manager signal has not been initialized",
    ):
        signal.manager_signal  # noqa: B018


def test_init() -> None:
    processor = ProcessPool()

    assert processor._n_process is None
    assert processor._multiprocessing_context == "spawn"
    assert processor._maxtasksperchild is None
    assert processor._chunksize is None
    assert processor._cost_function_wrapper is ProcessPoolCostFunctionWrapper


def test_init_with_parameters() -> None:
    processor = ProcessPool(
        n_process=4,
        multiprocessing_context="spawn",
        maxtasksperchild=10,
        chunksize=3,
    )

    assert processor._n_process == 4
    assert processor._multiprocessing_context == "spawn"
    assert processor._maxtasksperchild == 10
    assert processor._chunksize == 3


def test_init_rejects_invalid_multiprocessing_context() -> None:
    with pytest.raises(
        ValueError,
        match="Invalid multiprocessing context",
    ):
        ProcessPool(
            multiprocessing_context="invalid-context",
        )


def test_init_rejects_invalid_maxtasksperchild() -> None:
    with pytest.raises(
        ValueError,
        match="maxtasksperchild must be greater than zero",
    ):
        ProcessPool(maxtasksperchild=0)


def test_init_rejects_invalid_chunksize() -> None:
    with pytest.raises(
        ValueError,
        match="chunksize must be greater than zero",
    ):
        ProcessPool(chunksize=0)


def test_initialize_execution_context() -> None:
    processor = ProcessPool()
    migration_processor = create_migration_processor()

    processor._migration_processor = migration_processor

    processor.initialize_execution_context()

    assert isinstance(processor._stop_signal, StopSignal)
    assert not processor._stop_signal.is_set()

    migration_processor.start.assert_called_once_with(
        stop_signal=processor._stop_signal,
    )


def test_clear_execution_context() -> None:
    processor = ProcessPool()
    migration_processor = create_migration_processor()

    processor._migration_processor = migration_processor

    processor.initialize_execution_context()

    processor._stop_signal.set()

    processor.finalize_execution_context()

    migration_processor.stop.assert_called_once()

    assert isinstance(processor._stop_signal, LocalEvent)
    assert not processor._stop_signal.is_set()
