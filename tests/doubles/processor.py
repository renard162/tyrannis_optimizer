from doubles.algorithm import DummyAlgorithm, DummyCostFunctionWrapper

from tyrannis.core.processor import LocalEvent, ProcessorBase


class DummyProcessor(ProcessorBase[LocalEvent]):
    _cost_function_wrapper = DummyCostFunctionWrapper

    def __init__(self, algorithm: DummyAlgorithm | None = None) -> None:
        self._algorithm = algorithm

    def initialize_execution_context(self) -> None:
        self._stop_signal = LocalEvent()
        self._wait_signal = LocalEvent()

    def finalize_execution_context(self) -> None:
        self._stop_signal = LocalEvent()
        self._wait_signal = LocalEvent()

    def run(self) -> None:
        pass

    def initialize_loop_context(self) -> None:
        self._migration_signal = LocalEvent()

        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.initialize_loop_context(
            migration_signal=self._migration_signal,
        )

    def finalize_loop_context(self) -> None:
        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.finalize_loop_context()
        self._migration_signal = None  # type: ignore
