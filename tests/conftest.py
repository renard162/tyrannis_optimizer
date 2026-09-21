"""Global pytest configuration for the Tyrannis test suite."""

import pytest

from _support.objectives import CountingObjective


def pytest_configure(config: pytest.Config) -> None:
    """Register suite-local markers without changing project configuration."""
    config.addinivalue_line(
        "markers",
        "statistical: verifies a probabilistic property with a bounded sample.",
    )
    config.addinivalue_line(
        "markers",
        "optional: requires an optional runtime dependency and may be skipped.",
    )
    config.addinivalue_line(
        "markers",
        "multiprocess: starts a real process-based execution backend.",
    )


@pytest.fixture
def small_bounds() -> dict[str, tuple[float, float]]:
    """Return fresh, small continuous bounds suitable for shared test setup."""
    return {"x": (-5.0, 5.0), "y": (-3.0, 3.0)}


@pytest.fixture
def counting_objective() -> CountingObjective:
    """Return a fresh objective whose evaluation count is observable."""
    return CountingObjective()
