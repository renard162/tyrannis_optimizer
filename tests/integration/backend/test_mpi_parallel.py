"""MPI workers process particles for Optimizer and ScipyMinimizer on rank zero."""

import json
import os
import shutil
import sys
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest
from _support.assertions import assert_optimizer_result_consistent
from _support.integration_probe import (
    IntegrationProbeAlgorithm,
    assert_parallel_probe_continues,
)
from _support.mpi import _run_mpi
from _support.numerics import BASE_SEED
from _support.objectives import sphere
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult, minimize

from tyrannis import OptimizationHistory, Optimizer
from tyrannis.backend import MPIParallel
from tyrannis.compatibility.scipy import ScipyMinimizer
from tyrannis.space import Continuous


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
        history="iteration",
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
    assert_parallel_probe_continues(optimizer, probe)
    print(f"consolidated particles={len(probe.population)}", flush=True)


def _run_scipy_minimizer() -> None:
    from mpi4py import MPI

    assert MPI.COMM_WORLD.Get_rank() == 0

    def worker_fitness(x: NDArray[np.float64], offset: float) -> float:
        assert MPI.COMM_WORLD.Get_rank() in (1, 2)
        assert isinstance(x, np.ndarray) and x.shape == (2,)
        assert x[1] == 3.0 and offset == 7.0
        return offset - sphere(*x)

    def check_solution(result: OptimizeResult) -> None:
        assert result.x.shape == (2,)
        assert -1.0 <= result.x[0] <= 1.0 and result.x[1] == 3.0
        assert np.isfinite(result.fun) and result.fun == 7.0 - sphere(*result.x)

    def check_population(history: OptimizationHistory, last_iteration: int) -> None:
        # A fresh trace starts at initialization. Its prefix must travel with
        # the same particle through every update, including across executors.
        traces = {}
        for record in history.records:
            data = json.loads(record)
            particle = data["particle"]
            identifier = particle["identifier"]
            previous = traces.get(identifier, [])
            iteration = data["iteration"]
            trace = particle["trace"]
            assert iteration == len(previous)
            assert trace[:-1] == previous
            assert (trace[-1]["iteration"], trace[-1]["phase"]) == (
                iteration,
                "initialize" if iteration == 0 else "update",
            )
            assert np.isfinite(particle["fitness"])
            traces[identifier] = trace
        assert len(traces) == 2
        assert all(len(trace) == last_iteration + 1 for trace in traces.values())

    minimizer = ScipyMinimizer(
        IntegrationProbeAlgorithm(),
        n_iterations=2,
        n_particles=2,
        seed=BASE_SEED,
        backend=MPIParallel(),
        history="iteration",
    )
    bounds = [(-1.0, 1.0), (3.0, 3.0)]
    batches: list[NDArray[np.float64]] = []
    with pytest.warns(UserWarning, match="probe requested early stop"):
        stopped = minimize(
            worker_fitness,
            [0.0, 3.0],
            args=(7.0,),
            bounds=bounds,
            method=minimizer,
            callback=batches.append,
            options={
                "algorithm": IntegrationProbeAlgorithm(force_stop_at_iteration=1),
                "maxiter": 4,
            },
        )
    assert len(batches) == 1
    assert stopped.status == 2 and stopped.success is False and "nit" not in stopped
    check_solution(stopped)
    first_x = stopped.x.copy()
    first = minimizer.history
    assert stopped.history is first
    check_population(first, 1)
    records, columns, rows = list(first.records), first.columns, list(first.generator)

    batches.clear()
    complete = minimize(
        worker_fitness,
        [0.0, 3.0],
        args=(7.0,),
        bounds=bounds,
        method=minimizer,
        callback=batches.append,
        options={"maxiter": 4},
    )
    assert len(batches) == 2
    assert complete.status == 0 and complete.success is True and complete.nit == 4
    check_solution(complete)
    assert complete.history is minimizer.history and complete.history is not first
    check_population(minimizer.history, 4)
    assert stopped.history is first
    assert list(first.records) == records and first.columns == columns
    assert list(first.generator) == rows
    np.testing.assert_array_equal(stopped.x, first_x)
    print("ScipyMinimizer completed force stop and two MPI batches", flush=True)


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
    monkeypatch.setenv("PYTHONPATH", str(script.parents[2]), prepend=os.pathsep)
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent), prepend=os.pathsep)
    # These script assertions run outside pytest's assertion rewriting.
    monkeypatch.delenv("PYTHONOPTIMIZE", raising=False)
    result = _run_mpi(
        [launcher, "-n", "3", "tyrannis-mpi", "parallel", str(script)], 90
    )
    assert result.returncode == 0, result.stdout
    assert "consolidated particles=2" in result.stdout.splitlines(), result.stdout
    assert "ScipyMinimizer completed force stop and two MPI batches" in (
        result.stdout.splitlines()
    ), result.stdout


if __name__ == "__main__":
    _run_optimizer()
    _run_scipy_minimizer()
