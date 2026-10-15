"""Public launcher contracts without starting MPI or replacing pytest."""

import os
import sys
from importlib.machinery import ModuleSpec
from unittest.mock import Mock

import pytest

from tyrannis.cli import mpi


@pytest.fixture
def launch_boundaries(monkeypatch: pytest.MonkeyPatch) -> tuple[Mock, Mock]:
    """Isolate discovery, process replacement, and the launch environment."""
    dependency = Mock(return_value=ModuleSpec("mpi4py", loader=None))
    execute = Mock()
    monkeypatch.setattr(mpi, "find_spec", dependency)
    monkeypatch.setattr(os, "execv", execute)
    # Register even an initially absent key so main()'s assignment is undone.
    monkeypatch.setenv("TYRANNIS_MPI_MODE", "")
    monkeypatch.delenv("TYRANNIS_MPI_MODE")
    return dependency, execute


@pytest.mark.parametrize("mode", ["parallel", "distributed"])
def test_launch_preserves_interpreter_script_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
    launch_boundaries: tuple[Mock, Mock],
    mode: str,
) -> None:
    dependency, execute = launch_boundaries
    monkeypatch.setattr(
        sys, "argv", ["tyrannis-mpi", mode, "optimizer.py", "--foo", "value"]
    )

    def observe_launch(_executable: str, _arguments: list[str]) -> None:
        assert os.environ["TYRANNIS_MPI_MODE"] == mode

    execute.side_effect = observe_launch

    mpi.main()

    dependency.assert_called_once_with("mpi4py")
    execute.assert_called_once_with(
        sys.executable,
        [sys.executable, "-m", "mpi4py.futures", "optimizer.py", "--foo", "value"],
    )


@pytest.mark.parametrize(
    "arguments", [["--help"], ["parallel", "--help"]], ids=["mode", "script"]
)
def test_help_exits_successfully_before_dependency_check_or_launch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launch_boundaries: tuple[Mock, Mock],
    arguments: list[str],
) -> None:
    dependency, execute = launch_boundaries
    monkeypatch.setattr(sys, "argv", ["tyrannis-mpi", *arguments])

    with pytest.raises(SystemExit) as exc_info:
        mpi.main()

    assert exc_info.value.code == 0
    output = capsys.readouterr()
    assert "Usage:" in output.out
    assert "parallel" in output.out and "distributed" in output.out
    assert output.err == ""
    dependency.assert_not_called()
    execute.assert_not_called()
    assert "TYRANNIS_MPI_MODE" not in os.environ


def test_invalid_mode_reports_usage_error_before_dependency_check_or_launch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launch_boundaries: tuple[Mock, Mock],
) -> None:
    dependency, execute = launch_boundaries
    monkeypatch.setattr(sys, "argv", ["tyrannis-mpi", "invalid", "optimizer.py"])

    with pytest.raises(SystemExit) as exc_info:
        mpi.main()

    assert exc_info.value.code == 1
    output = capsys.readouterr()
    assert "invalid" in output.err.lower() and "mode" in output.err.lower()
    assert "Usage:" in output.out
    dependency.assert_not_called()
    execute.assert_not_called()
    assert "TYRANNIS_MPI_MODE" not in os.environ


def test_missing_script_reports_usage_error_before_dependency_check_or_launch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    launch_boundaries: tuple[Mock, Mock],
) -> None:
    dependency, execute = launch_boundaries
    monkeypatch.setattr(sys, "argv", ["tyrannis-mpi", "parallel"])

    with pytest.raises(SystemExit) as exc_info:
        mpi.main()

    assert exc_info.value.code == 1
    output = capsys.readouterr()
    assert "script" in output.err.lower()
    assert "Usage:" in output.out
    dependency.assert_not_called()
    execute.assert_not_called()
    assert "TYRANNIS_MPI_MODE" not in os.environ


def test_missing_dependency_raises_without_preparing_or_launching_script(
    monkeypatch: pytest.MonkeyPatch,
    launch_boundaries: tuple[Mock, Mock],
) -> None:
    dependency, execute = launch_boundaries
    dependency.return_value = None
    monkeypatch.setattr(sys, "argv", ["tyrannis-mpi", "parallel", "optimizer.py"])

    with pytest.raises(Exception):  # noqa: B017
        mpi.main()

    dependency.assert_called_once_with("mpi4py")
    execute.assert_not_called()
    assert "TYRANNIS_MPI_MODE" not in os.environ
