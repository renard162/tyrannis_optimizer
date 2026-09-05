import pytest

from tyrannis.backend.processor.threads import (
    ThreadsPool,
    ThreadsPoolCostFunctionWrapper,
)
from tyrannis.core.processor import LocalEvent


def test_init() -> None:
    processor = ThreadsPool()

    assert processor._n_process is None
    assert processor._chunksize == 1
    assert processor._cost_function_wrapper is ThreadsPoolCostFunctionWrapper


def test_init_with_parameters() -> None:
    processor = ThreadsPool(
        n_process=4,
        chunksize=3,
    )

    assert processor._n_process == 4
    assert processor._chunksize == 3


def test_init_rejects_invalid_chunksize() -> None:
    with pytest.raises(
        ValueError,
        match="chunksize must be greater than zero",
    ):
        ThreadsPool(chunksize=0)


def test_initialize_execution_context() -> None:
    processor = ThreadsPool()

    processor.initialize_execution_context()

    assert processor._stop_signal is not None
    assert processor._wait_signal is not None

    assert not processor._stop_signal.is_set()
    assert not processor._wait_signal.is_set()


def test_finalize_execution_context() -> None:
    processor = ThreadsPool()

    processor.initialize_execution_context()

    processor._stop_signal.set()
    processor._wait_signal.set()

    processor.finalize_execution_context()

    assert isinstance(processor._stop_signal, LocalEvent)
    assert isinstance(processor._wait_signal, LocalEvent)

    assert not processor._stop_signal.is_set()
    assert not processor._wait_signal.is_set()
