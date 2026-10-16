"""Manager lifetime shared only by the concrete migration integrations."""

from collections.abc import Iterator
from multiprocessing import get_context
from multiprocessing.managers import SyncManager

import pytest


@pytest.fixture
def manager() -> Iterator[SyncManager]:
    with get_context("spawn").Manager() as runtime:
        yield runtime
