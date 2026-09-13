from .binary import Binary
from .continuous import Continuous
from .integer import Integer

__all__ = [
    "Binary",
    "Continuous",
    "Integer",
]

SPACE_CLASSES = {
    "binary": Binary,
    "continuous": Continuous,
    "integer": Integer,
}
