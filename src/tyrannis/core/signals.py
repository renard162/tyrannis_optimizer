class LocalEvent:
    def __init__(self) -> None:
        self._state: bool = False

    def set(self) -> None:
        self._state = True

    def clear(self) -> None:
        self._state = False

    def is_set(self) -> bool:
        return self._state
