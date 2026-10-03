# Tyrannis

**A flexible metaheuristic optimization framework built for diverse search spaces and scalable execution.**

Tyrannis is a Python framework for solving optimization problems with population-based metaheuristic algorithms while keeping the problem definition independent from the execution strategy. Define your search space, choose an optimization algorithm, and decide how the workload should run—from a simple local execution to parallel or distributed processing. Tyrannis also provides support for mixed-variable optimization, allowing continuous, integer, binary, categorical, and permutation variables to coexist in the same problem.

## Installation

### Basic installation

Install Tyrannis with its standard local execution capabilities:

```bash
pip install tyrannis
```

The base installation includes the core optimization framework, local execution, and the basic LRU and disk caching mechanisms.

### Optional features

Additional capabilities are available through optional dependency groups:

| Extra | Installation | Adds |
| --- | --- | --- |
| `cache` | `pip install "tyrannis[cache]"` | Additional dependencies for advanced caching resources. |
| `spark` | `pip install "tyrannis[spark]"` | Support for the `SparkParallel` and `SparkDistributed` backends. |
| `mpi` | `pip install "tyrannis[mpi]"` | Support for the `MPIParallel` and `MPIDistributed` backends and the `tyrannis-mpi` launcher. |
| `all` | `pip install "tyrannis[all]"` | Installs all optional Tyrannis dependencies. |

Extras can also be combined explicitly in a single installation. For example:

```bash
pip install "tyrannis[cache,spark,mpi]"
```

The `spark` and `mpi` extras install the Python dependencies required by their respective backends. A compatible Spark or MPI runtime and execution infrastructure must also be available in the environment where those backends are used.

## Public API

The core optimization requires three components: an **optimizer**, a **search space**, and an **optimization algorithm**. The processor, backend, and migration components are optional and can be added to control how the optimization is executed, parallelized, distributed, and coordinated.

### Algorithms

`tyrannis.algorithm`

Algorithms are population-based metaheuristics used to search for a minimum within the defined search space.

| Import | Description |
| --- | --- |
| `PSO` | Particle Swarm Optimization for population-based search. |
| `GSA` | Gravitational Search Algorithm using gravitational interactions between candidate solutions to guide population-based search. |
| `PSOGSA` | Particle Swarm Optimization and Gravitational Search Algorithm for hybrid population-based search. |
| `GeneticAlgorithm` | Real-coded Genetic Algorithm for continuous optimization. |
| `AntColony` | Ant Colony Optimization for Continuous Domains (ACOR), which uses an archive of continuous solutions to guide the search. |
| `BeeColony` | Artificial Bee Colony optimization inspired by the foraging behavior of honey bees. |
| `GreyWolf` | Grey Wolf Optimization inspired by the social hierarchy and hunting behavior of grey wolves, with optional exploration enhancement through EEGWO. |
| `WhaleAlgorithm` | Whale Optimization Algorithm inspired by the bubble-net hunting behavior of humpback whales, with nonlinear control of the exploration-exploitation transition. |
| `DifferentialEvolution` | Differential Evolution using differential mutation and crossover for continuous optimization. |
| `CMAES`     | Covariance Matrix Adaptation Evolution Strategy for continuous optimization. |

### Spaces

`tyrannis.space`

Spaces define the search domain, including the variables, their bounds or available values, and the cost function to be minimized.

| Import | Description |
| --- | --- |
| `Continuous` | Continuous variables bounded by numerical lower and upper limits. |
| `Integer` | Integer variables bounded by numerical lower and upper limits. |
| `Binary` | Binary variables restricted to `False` or `True`. |
| `Categorical` | Categorical variables selected from a finite set of discrete choices. |
| `Ordinal` | Ordinal variables selected from an ordered set of discrete choices, with optional positions defining their relative distances. |
| `Permutation` | Permutation variables for ordering and permutation-based optimization problems. |
| `Sequence` | Variable-length ordered sequences selected from a finite set of choices, with repetition allowed and categorical or ordinal decoding. |
| `Mixed` | Composition of multiple spaces, allowing heterogeneous variable types in the same optimization problem. |

### Processors

`tyrannis.processor`

Processors define how the optimization loop is executed within each machine or execution environment.

| Import | Description |
| --- | --- |
| `Joblib` | Executes particle processing in parallel using Joblib. |
| `ProcessPool` | Executes particle processing in parallel using multiple processes. |
| `ThreadsPool` | Executes particle processing in parallel using a pool of threads. |

When no processor is provided, `Optimizer` uses its internal serial processor. This keeps the default execution sequential while allowing parallel particle evaluation to be enabled explicitly.

### Backends

`tyrannis.backend`

Backends define how optimization execution is organized across machines and, for distributed execution, across optimization islands.

| Import | Description |
| --- | --- |
| `SparkParallel` | Distributes particle initialization and updates through Spark while keeping the optimization as a single population without islands; compatible with Databricks, including shared-access clusters. |
| `SparkDistributed` | Distributes independent optimization islands across Spark executors and supports inter-island migration; compatible with Databricks single-user clusters. |
| `MPIParallel` | Distributes particle initialization and updates through MPI while keeping the optimization as a single population without islands. |
| `MPIDistributed` | Distributes independent optimization islands across MPI worker nodes and supports inter-island migration. |

When no backend is provided, `Optimizer` uses the local backend and executes the optimization on a single machine.

### Migrations

`tyrannis.migration`

Migration strategies define how solutions are exchanged between optimization islands when using a distributed, island-based backend.

| Import | Description |
| --- | --- |
| `GlobalBest` | Shares the best solution between islands, with configurable synchronization behavior. |
| `IslandMigration` | Exchanges solutions between islands according to configurable selection, migration size, interval, movement, and trigger strategies. |

When no migration strategy is provided for an island-based backend, the islands remain isolated and do not exchange solutions.

## Usage

### Basic optimization

A simple continuous optimization problem can be configured by defining a search space, selecting an algorithm, and creating an optimizer:

```python
from tyrannis import Optimizer
from tyrannis.algorithm import PSO
from tyrannis.space import Continuous


def cost_function(x, y):
    return x**2 + y**2


space = Continuous(
    boundaries={"x": (-10, 10), "y": (-10, 10)},
    cost_function=cost_function
)

algorithm = PSO()

optimizer = Optimizer(
    space=space,
    algorithm=algorithm,
    n_iterations=100,
    n_particles=50
)

optimizer.fit()

print(optimizer.best_solution)
# {"x": 0.0, "y": 0.0}

print(optimizer.best_fitness)
# 0.0
```

### Mixed search space and parallel processing

Tyrannis can combine different variable types in the same optimization problem. In this example, a continuous variable and a categorical variable are optimized together using the Artificial Bee Colony algorithm and a Joblib (parallel) processor:

```python
from tyrannis import Optimizer
from tyrannis.algorithm import BeeColony
from tyrannis.processor import Joblib
from tyrannis.space import Categorical, Continuous, Mixed


def cost_function(x, category):
    category_target = {
        "low": 0.0,
        "medium": 1.0,
        "high": 2.0,
    }
    return (x - category_target[category]) ** 2


space = Mixed(
    spaces={
        "x": Continuous((-5.0, 5.0)),
        "category": Categorical(("low", "medium", "high"))
    },
    cost_function=cost_function
)

algorithm = BeeColony()
processor = Joblib(joblib_backend="loky")

optimizer = Optimizer(
    space=space,
    algorithm=algorithm,
    processor=processor,
    n_iterations=100,
    n_particles=50
)

optimizer.fit()

print(optimizer.best_solution)
# {"x": 0.0, "category": "low"}

print(optimizer.best_fitness)
# 0.0
```

In this example, no backend is specified, so the optimization runs locally. The `Joblib` processor parallelizes particle evaluation across the available CPUs.

### Spark backends

The Spark backends require an active `SparkSession`, which is passed directly to the backend. In Databricks, the preconfigured `spark` session can be used directly. `SparkParallel` keeps a single optimization population and is compatible with Databricks compute, including Serverless when `spark_code_archive` is not required. Because updated particle data is collected by the driver during every iteration, communication overhead can become significant; workloads with larger populations and fewer iterations are therefore generally more appropriate.

```python
from tyrannis.backend import SparkParallel

backend = SparkParallel(spark)

optimizer = Optimizer(
    space=space,
    algorithm=algorithm,
    n_iterations=50,
    n_particles=300,
    backend=backend
)
```

`SparkDistributed` creates independent optimization islands, with their number explicitly defined by `n_executors`. Since each island performs its optimization locally, it avoids the per-iteration particle-transfer overhead of `SparkParallel`. In Databricks, this backend **does not work with Serverless compute** and therefore requires a cluster. It has been tested with **Access Mode Dedicated** (formerly Single User) and does not work with **Access Mode Shared**. **Access Mode No Isolation** has not been tested. A typical Databricks use case is execution through Jobs using `job_clusters`, which can be configured with Dedicated access mode. When the available executor count is unknown, using the number of worker machines as the initial value for `n_executors` is a reasonable approximation.

```python
from tyrannis.backend import SparkDistributed

backend = SparkDistributed(spark, n_executors=4)

optimizer = Optimizer(
    space=space,
    algorithm=algorithm,
    n_iterations=300,
    n_particles=50,
    backend=backend
)
```

### MPI backends

MPI backends are configured in the optimizer normally, but the script must be executed through an MPI runtime using the corresponding `tyrannis-mpi` mode. `MPIParallel` keeps a single population on rank 0 and distributes particle processing among the remaining ranks. For this backend, the recommended topology is **one MPI rank per available CPU core**, allowing particle evaluations to be distributed across the available processing capacity.

```python
from tyrannis.backend import MPIParallel

backend = MPIParallel()
```

A typical execution with OpenMPI is:

```bash
mpiexec --map-by core -n <total-ranks> tyrannis-mpi parallel optimizer.py
```

`MPIDistributed` reserves rank 0 for the driver and assigns one complete optimization island to each remaining MPI rank. For this backend, the recommended topology is **one MPI rank per machine or node**, so each island can use the resources of its machine independently through its configured Tyrannis processor. The number of islands is therefore equal to the total number of MPI ranks minus one.

```python
from tyrannis.backend import MPIDistributed

backend = MPIDistributed()
```

A typical distributed execution is:

```bash
mpiexec --map-by ppr:1:node -n <total-ranks> tyrannis-mpi distributed optimizer.py
```

For additional execution options and usage details, run `tyrannis-mpi --help`.

## Status

Tyrannis is currently in the **alpha stage of development**. The core architecture, multiple optimization algorithms, heterogeneous search spaces, local and parallel processors, Spark backends, and migration strategies are already implemented, while the API and implementation continue to evolve.

The roadmap below lists capabilities that are **not yet part of the current public API** and are planned for future releases.

### Roadmap

#### Algorithms

* Age-Layered Population Structure
* Invasive Weed Optimization

#### Spaces

* Graph-Constrained Search Spaces

#### Backends

* Ray
* Dask

#### Compatibility

* Sysidentpy - MetaMSS

#### Testing

* Mathematical integrity tests for all algorithms completed
* Completion and maintenance of mathematical tests for spaces
* Completion and maintenance of unit tests
* Construction and maintenance of integration tests
* Increased coverage of algorithms, spaces, processors, backends, and migration strategies
* Validation of distributed execution behavior

## License

Tyrannis is distributed under the BSD 3-Clause License.
