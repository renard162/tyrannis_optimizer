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
