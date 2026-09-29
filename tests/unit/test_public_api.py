"""Public package export integrity."""

from types import ModuleType

import pytest

import tyrannis
from tyrannis import algorithm, backend, core, migration, processor, space


@pytest.mark.parametrize(
    "package",
    [
        pytest.param(tyrannis, id="tyrannis"),
        pytest.param(algorithm, id="algorithm"),
        pytest.param(backend, id="backend"),
        pytest.param(core, id="core"),
        pytest.param(migration, id="migration"),
        pytest.param(processor, id="processor"),
        pytest.param(space, id="space"),
    ],
)
def test_public_exports_are_available(package: ModuleType) -> None:
    assert len(package.__all__) == len(set(package.__all__))

    for name in package.__all__:
        assert hasattr(package, name)
