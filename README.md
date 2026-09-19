# Tyrannis

**A flexible metaheuristic optimization framework built for diverse search spaces and scalable execution.**

Tyrannis is a Python framework for solving optimization problems with population-based metaheuristic algorithms while keeping the problem definition independent from the execution strategy. Define your search space, choose an optimization algorithm, and decide how the workload should run—from a simple local execution to parallel or distributed processing. Tyrannis also provides support for mixed-variable optimization, allowing continuous, integer, binary, categorical, and permutation variables to coexist in the same problem.

## Installation

### Basic installation

For a local installation with the standard execution capabilities:

```bash
pip install tyrannis
```

This installation provides the core framework, including local execution and the basic LRU and disk caching mechanisms.

### Advanced caching

To enable the additional caching capabilities:

```bash
pip install "tyrannis[cache]"
```

The `cache` extra installs the additional dependencies required by Tyrannis' advanced cache resources. Without this extra, only the basic LRU and disk cache mechanisms are available.

### Distributed processing with Spark

To enable distributed processing with Spark:

```bash
pip install "tyrannis[spark]"
```

This installs the dependencies required by the Spark backend. Distributed execution with Spark requires an appropriate Spark infrastructure to be usable.

### Combining extras

Extras can be installed together:

```bash
pip install "tyrannis[cache,spark]"
```

## Public API

The core optimization requires three components: an **optimizer**, a **search space**, and an **optimization algorithm**. The processor, backend, and migration components are optional and can be added to control how the optimization is executed, parallelized, distributed, and coordinated.

### Algorithms

`tyrannis.algorithm`

Algorithms are population-based metaheuristics used to search for a minimum within the defined search space.

| Import | Description |
| --- | --- |
| `PSO` | Particle Swarm Optimization for population-based search. |
| `PSOGSA` | Particle Swarm Optimization and Gravitational Search Algorithm for hybrid population-based search. |
| `GeneticAlgorithm` | Real-coded Genetic Algorithm for continuous optimization. |
| `AntColony` | Ant Colony Optimization for Continuous Domains (ACOR), which uses an archive of continuous solutions to guide the search. |
| `BeeColony` | Artificial Bee Colony optimization inspired by the foraging behavior of honey bees. |
| `GreyWolf` | Grey Wolf Optimization inspired by the social hierarchy and hunting behavior of grey wolves, with optional exploration enhancement through EEGWO. |
| `CMAES`     | Covariance Matrix Adaptation Evolution Strategy for continuous optimization. |

### Spaces

`tyrannis.space`

Spaces define the search domain, including the variables, their bounds or available values, and the cost function to be minimized.

| Import | Description |
| --- | --- |
| `Continuous` | Continuous variables bounded by numerical lower and upper limits. |
| `Integer` | Integer variables bounded by numerical lower and upper limits. |
| `Binary` | Binary variables restricted to `0` or `1`. |
| `Categorical` | Categorical variables selected from a finite set of discrete choices. |
| `Permutation` | Permutation variables for ordering and permutation-based optimization problems. |
| `Mixed` | Composition of multiple spaces, allowing heterogeneous variable types in the same optimization problem. |

### Processors

`tyrannis.processor`

Processors define how the optimization loop is executed within each machine or execution environment.

| Import | Description |
| --- | --- |
| `Joblib` | Executes particle processing through Joblib parallelism. |
| `ProcessPool` | Executes particle processing using multiple processes. |
| `ThreadsPool` | Executes particle processing using a pool of threads. |

When no processor is provided, `Optimizer` uses its internal serial processor. This keeps the default execution sequential while allowing parallel particle evaluation to be enabled explicitly.

### Backends

`tyrannis.backend`

Backends define how optimization execution is organized across machines and, for distributed execution, across optimization islands.

| Import | Description |
| --- | --- |
| `SparkParallel` | Distributes particle initialization and updates through Spark while keeping the optimization as a single population without islands. |
| `SparkDistributed` | Distributes independent optimization islands across Spark executors and supports inter-island migration. |

When no backend is provided, `Optimizer` uses the local backend and executes the optimization on a single machine.

### Migrations

`tyrannis.migration`

Migration strategies define how solutions are exchanged between optimization islands when using a distributed, island-based backend.

| Import | Description |
| --- | --- |
| `GlobalBest` | Shares the best solution between islands, with configurable synchronization behavior. |
| `IslandMigration` | Exchanges solutions between islands according to configurable selection, migration size, interval, movement, and trigger strategies. |

When no migration strategy is provided for an island-based backend, the islands remain isolated and do not exchange solutions.

## Examples

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

Tyrannis can combine different variable types in the same optimization problem. In this example, a continuous variable and a categorical variable are optimized together using the Artificial Bee Colony algorithm and a Joblib processor:

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

## Status

Tyrannis is currently in the **alpha stage of development**. The core architecture, multiple optimization algorithms, heterogeneous search spaces, local and parallel processors, Spark backends, and migration strategies are already implemented, while the API and implementation continue to evolve.

The roadmap below lists capabilities that are **not yet part of the current public API** and are planned for future releases.

### Roadmap

#### Algorithms

* GSA — Gravitational Search Algorithm
* Differential Evolution
* Whale Optimization Algorithm

#### Spaces

* Ordinal
* Set
* Graph

#### Backends

* MPI
* Ray
* Dask

#### Migration

* Diffusion

#### Testing

* Construction and maintenance of unit tests
* Construction and maintenance of integration tests
* Increased coverage of algorithms, spaces, processors, backends, and migration strategies
* Validation of distributed execution behavior

## License

Tyrannis is distributed under the BSD 3-Clause License.
