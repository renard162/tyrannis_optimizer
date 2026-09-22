"""The public Space registry keeps its first registration."""

import pytest

from tyrannis import space


def test_space_registration_preserves_first_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(space, "_SPACE_CLASSES", {})

    assert space.get_space_class("example") is None
    space.register_space("example", space.Continuous)
    assert space.get_space_class("example") is space.Continuous

    space.register_space("example", space.Binary)
    assert space.get_space_class("example") is space.Continuous
