"""MPI workers return processed particles to the rank-zero Optimizer."""

import os
import shutil
import subprocess
import sys
from contextlib import suppress
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryFile

import pytest

from tests._support.assertions import assert_optimizer_result_consistent
from tests._support.integration_probe import IntegrationProbeAlgorithm
from tests._support.numerics import BASE_SEED
from tests._support.objectives import sphere
from tyrannis import Optimizer
from tyrannis.backend import MPIParallel
from tyrannis.space import Continuous


def _run_mpi(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    # A file keeps diagnostics available even if a rank outlives the launcher.
    with (
        TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as log,
        subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=os.name == "posix",
        ) as process,
    ):
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            _ = log.seek(0)
            raise TimeoutError(
                f"{command[0]} timed out after {timeout}s:\n{log.read()}"
            ) from exc
        finally:
            if process.poll() is None and os.name == "posix":
                # Give the launcher time to finalize the ranks itself.
                process.terminate()
                with suppress(subprocess.TimeoutExpired):
                    _ = process.wait(timeout=10)
            try:
                if os.name == "posix" and process.returncode != 0:
                    # Ranks can have separate process groups in this private session.
                    cleanup = subprocess.run(
                        ["pkill", "-KILL", "-s", str(process.pid)],
                        check=False,
                        timeout=5,
                    )
                    if (
                        cleanup.returncode != 1
                    ):  # 1 means the session was already empty.
                        cleanup.check_returncode()
                elif process.poll() is None:
                    _ = subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                        check=True,
                        timeout=10,
                    )
            finally:
                if process.poll() is None:
                    process.kill()
        _ = log.seek(0)
        return subprocess.CompletedProcess(command, status, stdout=log.read())


def _run_optimizer() -> None:
    from mpi4py import MPI

    assert MPI.COMM_WORLD.Get_rank() == 0
    assert MPI.COMM_WORLD.Get_size() == 3

    def worker_fitness(x: float) -> float:
        assert MPI.COMM_WORLD.Get_rank() in (1, 2)
        return -sphere(x)

    probe = IntegrationProbeAlgorithm()
    optimizer = Optimizer(
        space=Continuous(boundaries={"x": (-1.0, 1.0)}, cost_function=worker_fitness),
        algorithm=probe,
        backend=MPIParallel(),
        n_particles=2,
        n_iterations=1,
        seed=BASE_SEED,
    ).fit()

    for identifier, particle in probe.population.items():
        assert [(event["particle_id"], event["phase"]) for event in particle.trace] == [
            (identifier, "initialize"),
            (identifier, "update"),
        ]
        assert particle.fitness == -sphere(**particle.variables) < 0.0

    best = min(probe.population.values(), key=lambda particle: particle.fitness)
    assert optimizer.result_ == {
        "identifier": best.identifier,
        "variables": best.variables,
        "fitness": best.fitness,
    }
    assert_optimizer_result_consistent(optimizer)
    print(f"consolidated particles={len(probe.population)}", flush=True)


@pytest.mark.optional
@pytest.mark.multiprocess
def test_optimizer_consolidates_particles_from_mpi_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if find_spec("mpi4py") is None:
        pytest.skip("MPIParallel requires mpi4py")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("MPIParallel requires mpiexec or mpirun")

    smoke = (
        "from mpi4py import MPI\n"
        "world = MPI.COMM_WORLD\n"
        "print(f'rank {world.Get_rank()} of {world.Get_size()}', flush=True)"
    )
    try:
        startup = _run_mpi([launcher, "-n", "2", sys.executable, "-c", smoke], 30)
    except TimeoutError as exc:
        pytest.skip(str(exc))
    if startup.returncode:
        pytest.skip(f"MPI smoke exited {startup.returncode}: {startup.stdout}")
    assert sorted(
        line for line in startup.stdout.splitlines() if line.startswith("rank ")
    ) == [
        "rank 0 of 2",
        "rank 1 of 2",
    ], startup.stdout

    script = Path(__file__).resolve()
    # Ranks import the serialized probe from the checkout and use pytest's venv.
    monkeypatch.setenv("PYTHONPATH", str(script.parents[3]), prepend=os.pathsep)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent), prepend=os.pathsep)
    # These script assertions run outside pytest's assertion rewriting.
    monkeypatch.delenv("PYTHONOPTIMIZE", raising=False)
    result = _run_mpi(
        [launcher, "-n", "3", "tyrannis-mpi", "parallel", str(script)], 90
    )
    assert result.returncode == 0, result.stdout
    assert "consolidated particles=2" in result.stdout.splitlines(), result.stdout


if __name__ == "__main__":
    _run_optimizer()
