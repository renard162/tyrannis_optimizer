"""MPI islands exchange a best particle before returning their updated results."""

import os
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import pytest

from tests._support.assertions import assert_optimizer_result_consistent
from tests._support.integration_probe import IntegrationProbeAlgorithm
from tests._support.mpi import _run_mpi
from tests._support.numerics import BASE_SEED
from tests._support.objectives import sphere
from tyrannis import Optimizer
from tyrannis.backend import MPIDistributed
from tyrannis.core.results import HistoryConfig
from tyrannis.migration import GlobalBest
from tyrannis.space import Continuous


def _run_optimizer() -> None:
    from mpi4py import MPI

    assert MPI.COMM_WORLD.Get_rank() == 0
    assert MPI.COMM_WORLD.Get_size() == 3

    def worker_fitness(x: float) -> float:
        rank = MPI.COMM_WORLD.Get_rank()
        assert rank in (1, 2)
        # At the initial midpoint x=0 the fitness identifies the worker rank.
        return rank - sphere(x)

    backend = MPIDistributed()
    optimizer = Optimizer(
        space=Continuous({"x": (-1.0, 1.0)}, cost_function=worker_fitness),
        algorithm=IntegrationProbeAlgorithm(),
        backend=backend,
        migration=GlobalBest(synchronous=True, initial_iter=1, check_interval=1),
        n_particles=1,
        n_iterations=1,
        seed=BASE_SEED,
        history=["iteration", "pre_iteration"],
    ).fit()

    rows: list[dict[str, object]] = [
        dict(zip(optimizer.history_columns, values, strict=True))
        for values in optimizer.history_generator
    ]
    ranks: list[float] = []
    candidates: list[tuple[float, str, float]] = []
    iteration = HistoryConfig.get_event("iteration")
    pre_iteration = HistoryConfig.get_event("pre_iteration")
    for island in ("island:0", "island:1"):
        history = {
            (row["iteration"], row["event"]): row
            for row in rows
            if row["origin"] == island
        }
        rank = history[0, iteration]["fitness"]
        assert isinstance(rank, float)
        ranks.append(rank)
        # Rank 2 now holds rank 1's fitness in its own slot, before reevaluation.
        slot = f"{island}|particle:0"
        after_checkpoint = history[1, pre_iteration]
        assert after_checkpoint["identifier"] == slot
        assert after_checkpoint["fitness"] == 1.0
        updated = history[1, iteration]
        identifier, x = updated["identifier"], updated["x"]
        assert isinstance(identifier, str) and isinstance(x, float)
        assert identifier == slot
        fitness = rank - sphere(x)
        assert updated["fitness"] == fitness < rank
        candidates.append((fitness, identifier, x))

    assert sorted(ranks) == [1.0, 2.0]
    fitness, identifier, x = min(candidates)
    assert optimizer.result_ == {
        "identifier": identifier,
        "variables": {"x": x},
        "fitness": fitness,
    }
    assert_optimizer_result_consistent(optimizer)
    first_result = optimizer.result_
    first_history = list(optimizer.internal_history_generator_)
    with patch.object(backend, "execute") as execute:
        with pytest.raises(RuntimeError):
            _ = optimizer.fit()
        execute.assert_not_called()
    assert optimizer.result_ == first_result
    assert list(optimizer.internal_history_generator_) == first_history
    print("MPI island exchange and global result verified", flush=True)


@pytest.mark.optional
@pytest.mark.multiprocess
def test_optimizer_consolidates_results_from_communicating_mpi_islands(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if find_spec("mpi4py") is None:
        pytest.skip("MPIDistributed requires mpi4py")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPIDistributed requires mpiexec or mpirun")

    smoke = (
        "from mpi4py import MPI\n"
        "world = MPI.COMM_WORLD\n"
        "print(f'rank {world.Get_rank()} of {world.Get_size()}', flush=True)\n"
        "if MPI.Query_thread() < MPI.THREAD_MULTIPLE:\n"
        "    raise SystemExit('MPI communication requires MPI_THREAD_MULTIPLE')"
    )
    try:
        startup = _run_mpi([launcher, "-n", "2", sys.executable, "-c", smoke], 30)
    except TimeoutError as exc:
        pytest.skip(str(exc))
    if startup.returncode:
        pytest.skip(f"MPI smoke exited {startup.returncode}: {startup.stdout}")
    assert sorted(
        line for line in startup.stdout.splitlines() if line.startswith("rank ")
    ) == ["rank 0 of 2", "rank 1 of 2"], startup.stdout

    script = Path(__file__).resolve()
    monkeypatch.setenv("PYTHONPATH", str(script.parents[3]), prepend=os.pathsep)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent), prepend=os.pathsep)
    monkeypatch.delenv("PYTHONOPTIMIZE", raising=False)
    result = _run_mpi(
        [launcher, "-n", "3", "tyrannis-mpi", "distributed", str(script)], 90
    )
    assert result.returncode == 0, result.stdout
    assert (
        "MPI island exchange and global result verified" in result.stdout.splitlines()
    ), result.stdout


if __name__ == "__main__":
    _run_optimizer()
