_SPACE_CLASSES = {}


def register_space(name: str, space_class: type) -> None:
    if name in _SPACE_CLASSES:
        return
    _SPACE_CLASSES[name] = space_class


def get_space_class(name: str) -> type | None:
    return _SPACE_CLASSES.get(name)


from .binary import Binary
from .categorical import Categorical
from .continuous import Continuous
from .integer import Integer
from .mixed import Mixed
from .permutation import Permutation

__all__ = [
    "Binary",
    "Categorical",
    "Continuous",
    "Integer",
    "Mixed",
    "Permutation",
]
