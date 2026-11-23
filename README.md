# Tyrannis

**A flexible metaheuristic optimization framework built for diverse search spaces and scalable execution.**

Tyrannis is a Python framework for solving optimization problems with population-based metaheuristic algorithms while keeping the problem definition independent from the execution strategy. Define your search space, choose an optimization algorithm, and select whether to execute it locally, in parallel, or in a distributed environment. Tyrannis also provides support for mixed-variable optimization, allowing continuous, integer, binary, categorical, and permutation variables to coexist in the same problem.

<br>

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
| `cache` | `pip install "tyrannis[cache]"` | Additional LFU, FIFO, and Random Replacement caching strategies. |
| `spark` | `pip install "tyrannis[spark]"` | Support for the `SparkParallel` and `SparkDistributed` backends. |
| `mpi` | `pip install "tyrannis[mpi]"` | Support for the `MPIParallel` and `MPIDistributed` backends and the `tyrannis-mpi` launcher. |

Extras can also be combined explicitly in a single installation. For example:

```bash
pip install "tyrannis[cache,spark,mpi]"
```

The `spark` and `mpi` extras install the Python dependencies required by their respective backends. A compatible Spark or MPI runtime and execution infrastructure must also be available in the environment where those backends are used.

<br>

## Public API

The core optimization requires three components: an **optimizer**, a **search space**, and an **optimization algorithm**. The processor, backend, and migration components are optional and can be added to control how the optimization is executed, parallelized, distributed, and coordinated.

### Algorithms

`tyrannis.algorithm`

Algorithms are population-based metaheuristics used to search for a minimum within the defined search space.

| Import | Description | Family |
| --- | --- | --- |
| `PSO` | Particle Swarm Optimization for population-based search. [1, 2] | Agent Dynamics Algorithm |
| `GSA` | Gravitational Search Algorithm using gravitational interactions between candidate solutions to guide population-based search. [3] | Agent Dynamics Algorithm |
| `PSOGSA` | Particle Swarm Optimization and Gravitational Search Algorithm for hybrid population-based search. [4] | Agent Dynamics Algorithm |
| `AntColony` | Ant Colony Optimization for Continuous Domains (ACOR), which uses an archive of continuous solutions to guide the search. [5] | Colony-Based Swarm Intelligence |
| `BeeColony` | Artificial Bee Colony optimization inspired by the foraging behavior of honey bees. [6] | Colony-Based Swarm Intelligence |
| `GreyWolf` | Grey Wolf Optimization inspired by the social hierarchy and hunting behavior of grey wolves, with optional exploration enhancement through EEGWO. [7, 8] | Leader-Guided Collective Search |
| `WhaleAlgorithm` | Whale Optimization Algorithm inspired by the bubble-net hunting behavior of humpback whales, with nonlinear control of the exploration-exploitation transition. [9, 10] | Leader-Guided Collective Search |
| `GeneticAlgorithm` | Real-coded Genetic Algorithm for continuous optimization. [11, 12] | Evolutionary Algorithm |
| `DifferentialEvolution` | Differential Evolution using differential mutation and crossover for continuous optimization. [13] | Evolutionary Algorithm |
| `CMAES`     | Covariance Matrix Adaptation Evolution Strategy for continuous optimization. [14, 15] | Evolutionary Algorithm |
| `EGNA` | Estimation of Gaussian Networks Algorithm, a multivariate continuous Estimation of Distribution Algorithm that learns and samples Gaussian Bayesian networks. [16, 17] | Estimation of Distribution Algorithm |


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
| `SparkDistributed` | Distributes independent optimization islands as Spark partitions and supports inter-island migration; compatible with Databricks Dedicated access mode clusters. |
| `MPIParallel` | Distributes particle initialization and updates through MPI while keeping the optimization as a single population without islands. |
| `MPIDistributed` | Distributes independent optimization islands across MPI worker ranks and supports inter-island migration. |

When no backend is provided, `Optimizer` uses the local backend and executes the optimization on a single machine.


### Migrations

`tyrannis.migration`

Migration strategies define how solutions are exchanged between optimization islands when using a distributed, island-based backend.

| Import | Description |
| --- | --- |
| `GlobalBest` | Shares the best solution between islands, with configurable synchronization behavior. |
| `IslandMigration` | Exchanges solutions between islands according to configurable selection, migration size, interval, movement, and trigger strategies. |

When no migration strategy is provided for an island-based backend, the islands remain isolated and do not exchange solutions.

<br>

## Usage

The examples reuse imports and definitions from the basic example unless otherwise specified.

### Basic optimization

Define a cost function, choose a search space and an algorithm, then run the optimizer. By default, execution is local and sequential.

```python
from tyrannis import Optimizer
from tyrannis.algorithm import PSO
from tyrannis.space import Continuous


def cost_function(x, y):
    return x**2 + y**2


space = Continuous(
    boundaries={"x": (-10, 10), "y": (-10, 10)},
    cost_function=cost_function,
)

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
)

optimizer.fit()

print(optimizer.best_solution)
print(optimizer.best_fitness)
```

---

#### Non-continuous search space

Use `Permutation` to optimize the ordering of a set of choices. The cost function receives the decoded ordering as a list.

```python
from tyrannis.space import Permutation


def permutation_cost_function(order):
    return sum(abs(a - b) for a, b in zip(order, order[1:]))


space = Permutation(
    choices=(1, 2, 3, 4, 5),
    cost_function=permutation_cost_function,
)

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
)

optimizer.fit()
```

---

#### Binary search space

`Binary(bits=32)` represents 32 independent Boolean decisions. The decoded solution is a list of Boolean values, which Tyrannis unpacks into positional arguments when calling the cost function. The `*selected` parameter collects these arguments into a tuple named `selected`. This example selects eight options while maximizing their combined value.

```python
from tyrannis.space import Binary


def selection_cost(*selected):
    benefit = sum(index for index, chosen in enumerate(selected, start=1) if chosen)
    return 100 * abs(sum(selected) - 8) - benefit


space = Binary(bits=32, cost_function=selection_cost)

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
)

optimizer.fit()
```

---

#### Mixed search space

Use `Mixed` to combine continuous, categorical, and binary variables. Each variable has its own search-space type and independently defined bounds or available values. The cost function receives the named variables as keyword arguments.

```python
from tyrannis.space import Binary, Categorical, Continuous, Mixed


def mixed_cost_function(x, category, option_a, option_b):
    targets = {"low": 0.0, "medium": 1.0, "high": 2.0}
    target = targets[category] + option_a - option_b
    return (x - target) ** 2 + 0.1 * (option_a + option_b)


space = Mixed(
    spaces={
        "x": Continuous((-5.0, 5.0)),
        "category": Categorical(("low", "medium", "high")),
        "option_a": Binary(),
        "option_b": Binary(),
    },
    cost_function=mixed_cost_function,
)

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
)

optimizer.fit()
```

---

#### Caching

Enable caching on `Binary` to reuse cost-function results when the same decoded bit pattern is evaluated again. This example reuses `selection_cost` from the binary search space example.

```python
from tyrannis.space import Binary

space = Binary(
    bits=32,
    cost_function=selection_cost,
    use_cache=True,
    cache_type="lru",
    cache_size=10_000,
)

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
)

optimizer.fit()
```

---

### Parallel processing with processors

Select a processor to parallelize particle processing on the local machine. `Joblib` with the `loky` backend is the recommended general-purpose option.

```python
from tyrannis.processor import Joblib

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
    processor=Joblib(joblib_backend="loky"),
)

optimizer.fit()
```

---

### Spark backends

Spark requires an active `SparkSession`, available as `spark` in the following examples.

#### Distributed optimization

`SparkDistributed` runs independent optimization islands as Spark partitions. `n_executors` sets the number of islands, not the number of physical executors. On Databricks, use a Dedicated cluster rather than Serverless or Shared access mode.

```python
from tyrannis.backend import SparkDistributed

backend = SparkDistributed(
    spark,
    n_executors=4,
)

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
    backend=backend,
)

optimizer.fit()
```

<br>

#### Spark Parallel optimization

`SparkParallel` distributes particle processing through Spark while retaining a single optimization population. On Databricks, it can also use Serverless compute when `spark_code_archive` is not required.

```python
from tyrannis.backend import SparkParallel

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
    backend=SparkParallel(spark),
)

optimizer.fit()
```

---

### MPI backends

Save the imports, cost function, and search space from the basic example together with one of the following MPI configurations in `optimizer.py`. Execute the script using `tyrannis-mpi` through an MPI runtime.

#### MPI Distributed optimization

`MPIDistributed` assigns one optimization island to each worker rank, with rank 0 reserved for the driver.

```python
from tyrannis.backend import MPIDistributed

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
    backend=MPIDistributed(),
)

optimizer.fit()
```

```bash
mpiexec --map-by ppr:1:node -n <total-ranks> tyrannis-mpi distributed optimizer.py
```

<br>

#### MPI Parallel optimization

`MPIParallel` keeps a single population on rank 0 and distributes particle processing across the remaining ranks.

```python
from tyrannis.backend import MPIParallel

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
    backend=MPIParallel(),
)

optimizer.fit()
```

```bash
mpiexec --map-by core -n <total-ranks> tyrannis-mpi parallel optimizer.py
```

For additional options, run `tyrannis-mpi --help`.

---

### Migration

Use the `migration` parameter to exchange solutions between islands in a distributed backend. For example, `GlobalBest` shares improvements in the best solution found across islands.

```python
from tyrannis.backend import SparkDistributed
from tyrannis.migration import GlobalBest

optimizer = Optimizer(
    space=space,
    algorithm=PSO(),
    n_iterations=100,
    n_particles=50,
    backend=SparkDistributed(spark, n_executors=4),
    migration=GlobalBest(check_interval=10),
)

optimizer.fit()
```

<br>

## Roadmap

The following features are planned for upcoming releases:

#### Algorithms

* Age-Layered Population Structure
* Invasive Weed Optimization
* More Estimation of Distribution Algorithms (EDAs)

#### Spaces

* Graph-Constrained Search Spaces

#### Backends

* Ray
* Dask

#### Compatibility

* SysIdentPy - CustomMetaMSS integration

<br>

## Bibliography

[1] Kennedy, J. and Eberhart, R. (1995) 'Particle swarm optimization', *Proceedings of ICNN'95 - International Conference on Neural Networks*, pp. 1942–1948. https://doi.org/10.1109/ICNN.1995.488968

[2] Clerc, M. and Kennedy, J. (2002) 'The particle swarm - explosion, stability, and convergence in a multidimensional complex space', *IEEE Transactions on Evolutionary Computation*, 6(1), pp. 58–73. https://doi.org/10.1109/4235.985692

[3] Rashedi, E., Nezamabadi-pour, H. and Saryazdi, S. (2009) 'GSA: A gravitational search algorithm', *Information Sciences*, 179(13), pp. 2232–2248. https://doi.org/10.1016/j.ins.2009.03.004

[4] Mirjalili, S. and Hashim, S.Z.M. (2010) 'A new hybrid PSOGSA algorithm for function optimization', *Proceedings of the International Conference on Computer and Information Application (ICCIA 2010)*, pp. 374–377. https://doi.org/10.1109/ICCIA.2010.6141614

[5] Socha, K. and Dorigo, M. (2008) 'Ant colony optimization for continuous domains', *European Journal of Operational Research*, 185(3), pp. 1155–1173. https://doi.org/10.1016/j.ejor.2006.06.046

[6] Karaboga, D. and Basturk, B. (2007) 'A powerful and efficient algorithm for numerical function optimization: Artificial Bee Colony (ABC) algorithm', *Journal of Global Optimization*, 39(3), pp. 459–471. https://doi.org/10.1007/s10898-007-9149-x

[7] Mirjalili, S., Mirjalili, S.M. and Lewis, A. (2014) 'Grey Wolf Optimizer', *Advances in Engineering Software*, 69, pp. 46–61. https://doi.org/10.1016/j.advengsoft.2013.12.007

[8] Long, W., Jiao, J., Liang, X. and Tang, M. (2018) 'An exploration-enhanced grey wolf optimizer to solve high-dimensional numerical optimization', *Engineering Applications of Artificial Intelligence*, 68, pp. 63–80. https://doi.org/10.1016/j.engappai.2017.10.024

[9] Mirjalili, S. and Lewis, A. (2016) 'The Whale Optimization Algorithm', *Advances in Engineering Software*, 95, pp. 51–67. https://doi.org/10.1016/j.advengsoft.2016.01.008

[10] Zhong, M. and Long, W. (2017) 'Whale optimization algorithm with nonlinear control parameter', *MATEC Web of Conferences*, 139, 00157. https://doi.org/10.1051/matecconf/201713900157

[11] Eshelman, L.J. and Schaffer, J.D. (1993) 'Real-coded genetic algorithms and interval-schemata', *Foundations of Genetic Algorithms*, 2, pp. 187–202. https://doi.org/10.1016/B978-0-08-094832-4.50018-0

[12] Deb, K. and Deb, D. (2014) 'Analysing mutation schemes for real-parameter genetic algorithms', *International Journal of Artificial Intelligence and Soft Computing*, 4(1), pp. 1–28. https://doi.org/10.1504/IJAISC.2014.059280

[13] Storn, R. and Price, K. (1997) 'Differential Evolution - a simple and efficient heuristic for global optimization over continuous spaces', *Journal of Global Optimization*, 11, pp. 341–359. https://doi.org/10.1023/A:1008202821328

[14] Hansen, N. and Ostermeier, A. (2001) 'Completely derandomized self-adaptation in evolution strategies', *Evolutionary Computation*, 9(2), pp. 159–195. https://doi.org/10.1162/106365601750190398

[15] Hansen, N. (2006) 'The CMA Evolution Strategy: A comparing review', in *Towards a New Evolutionary Computation*, pp. 75–102. https://doi.org/10.1007/3-540-32494-1_4

[16] Bengoetxea, E., Miquélez, T., Larrañaga, P. and Lozano, J.A. (2002) 'Experimental results in function optimization with EDAs in continuous domain', in *Estimation of Distribution Algorithms: A New Tool for Evolutionary Computation*, pp. 181–194. https://doi.org/10.1007/978-1-4615-1539-5_8

[17] Geiger, D. and Heckerman, D. (1994) 'Learning Gaussian networks', *Uncertainty Proceedings 1994*, pp. 235–243. https://doi.org/10.1016/B978-1-55860-332-5.50035-3

<br>

## License

Tyrannis is distributed under the BSD 3-Clause License.
