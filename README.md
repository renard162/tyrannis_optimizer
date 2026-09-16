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

This installs the dependencies required by the Spark backend. Distributed execution with Spark requires an appropriate Spark infrastructure to be available.

### Combining extras

Extras can be installed together:

```bash
pip install "tyrannis[cache,spark]"
```

## Public API

### Spaces

`tyrannis.space`

| Import        | Description                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------- |
| `Binary`      | Defines a binary search space whose variables can take the values `0` or `1`.                      |
| `Categorical` | Defines a categorical search space based on a finite set of discrete choices.                      |
| `Continuous`  | Defines a continuous search space bounded by numerical lower and upper limits.                     |
| `Integer`     | Defines an integer search space bounded by numerical lower and upper limits.                       |
| `Mixed`       | Combines multiple search spaces, allowing optimization problems with heterogeneous variable types. |
| `Permutation` | Defines a search space for permutation-based optimization problems.                                |

### Algorithms

`tyrannis.algorithm`

| Import                | Description                                                                               |
| --------------------- | ----------------------------------------------------------------------------------------- |
| `PSO`                 | Particle Swarm Optimization algorithm for population-based continuous and encoded search. |
| `ArtificialBeeColony` | Artificial Bee Colony algorithm inspired by the foraging behavior of honey bees.          |

### Processors

`tyrannis.processor`

| Import        | Description                                                  |
| ------------- | ------------------------------------------------------------ |
| `Joblib`      | Executes particle processing using Joblib-based parallelism. |
| `ProcessPool` | Executes particle processing using multiple processes.       |
| `ThreadsPool` | Executes particle processing using a pool of threads.        |

### Backends

`tyrannis.backend`

| Import             | Description                                                                                                                             |
| ------------------ | --------------------------------------------------------------------------------------------------------------------------------------- |
| `SparkParallel`    | Executes optimization using distributed processing with Spark on a single island.                                                       |
| `SparkDistributed` | Executes optimization using distributed processing with Spark across multiple islands, supporting distributed population and migration. |

### Migrations

`tyrannis.migration`

| Import            | Description                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------- |
| `GlobalBest`      | Shares the globally best solution between islands during distributed optimization.              |
| `IslandMigration` | Provides migration of solutions between islands according to the configured migration strategy. |

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
from tyrannis.algorithm import ArtificialBeeColony
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

algorithm = ArtificialBeeColony()
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

Tyrannis is currently in the **alpha stage of development**. The core architecture and initial optimization capabilities are already available, but the API and implementation are still evolving.

The following features are planned for future releases.

### Roadmap

#### Algorithms

* PSOGSA
* Genetic Algorithm
* GSA — Gravitational Search Algorithm
* Differential Evolution
* CMA-ES — Covariance Matrix Adaptation Evolution Strategy
* Ant Colony Optimization
* Grey Wolf Optimization
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
