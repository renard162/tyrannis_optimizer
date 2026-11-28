# Algorithm Development

This guide defines the implementation contract for optimization algorithms in Tyrannis Optimizer. An algorithm must preserve its mathematical formulation while remaining compatible with the framework's particle lifecycle, reproducibility, parallel execution, and migration.

Use **Part I** as a quick reference when implementing or reviewing an algorithm. Consult **Part II** for the detailed contracts, rationale, and examples.

## Contents

**Part I — Quick reference**

- [1. Implementation checklist](#1-implementation-checklist)
- [2. Method reference](#2-method-reference)
- [3. Execution lifecycle](#3-execution-lifecycle)
- [4. Essential invariants](#4-essential-invariants)

**Part II — Implementation contracts**

- [5. Architecture and state ownership](#5-architecture-and-state-ownership)
- [6. Initialization and iteration methods](#6-initialization-and-iteration-methods)
- [7. Fitness evaluation and consolidation](#7-fitness-evaluation-and-consolidation)
- [8. Randomness and random cache](#8-randomness-and-random-cache)
- [9. Optional second processing phase](#9-optional-second-processing-phase)
- [10. Optional forced termination](#10-optional-forced-termination)
- [11. State persistence and migration](#11-state-persistence-and-migration)
- [12. Mathematical and numerical requirements](#12-mathematical-and-numerical-requirements)
- [13. Public API, documentation, and validation](#13-public-api-documentation-and-validation)

---

## Part I — Quick reference

### 1. Implementation checklist

Follow this sequence when introducing a new algorithm.

- [ ] Define the reference mathematical formulation, including update equations, selection, schedules, stochastic operations, and numerical edge cases.
- [ ] Assess whether the algorithm fits the continuous encoded domain and population-wide processing model.
- [ ] Create a dedicated module under `src/tyrannis/algorithm/` containing a concrete particle class and a concrete algorithm class.
- [ ] Separate particle-specific persistent state from population-level state and transient evaluation state.
- [ ] Implement all required methods in [the method reference](#2-method-reference), including constructor validation and migration-compatible particle creation.
- [ ] Route every evaluation through `ParticleBase.update` and consolidate candidates according to the mathematical formulation.
- [ ] Prepare particle-specific randomness through `create_random_cache` before each processing phase.
- [ ] Implement the optional second processing phase or forced termination only when the algorithm requires it.
- [ ] Preserve serializable particle state, handle migration, and finalize all temporary particles before publishing iteration results.
- [ ] Verify numerical stability, coordinate boundaries, minimization semantics, and iteration numbering.
- [ ] Document the public constructor, export the algorithm, and update the relevant README entries.
- [ ] Validate the mathematical and execution contracts with focused tests; pass the project's lint, formatting, and type checks.

### 2. Method reference

The concrete algorithm implements the mathematical operations. The framework controls their execution order, dispatch, and population integration. The required methods are abstract in `AlgorithmBase`.

| Required method | Main responsibility | Must not do |
| --- | --- | --- |
| [`__init__`](#61-__init__) | Define and validate public parameters; initialize algorithm-specific state. | Accept framework-managed configuration. |
| [`create_particle`](#62-create_particle) | Create or reconstruct a particle and register it in `_population`. | Evaluate fitness or discard supplied persistent state. |
| [`delete_particle`](#63-delete_particle) | Remove a particle and maintain bookkeeping. | Choose a particle implicitly without a defined removal policy. |
| [`pre_iteration`](#64-pre_iteration) | Prepare shared state, schedules, population changes, and context-dependent constraints. | Evaluate new particles or prepare worker-specific random inputs. |
| [`create_random_cache`](#65-create_random_cache) | Assign fresh random state for the requested identifiers and processing phase. | Share mutable generator state between particles. |
| [`initialize_particle`](#66-initialize_particle) | Initialize the specified new particle and evaluate its initial solution if required. | Consolidate the particle, reevaluate a valid arrival solely because it is new, or use `_rng` directly. |
| [`consolidate_new_particles`](#67-consolidate_new_particles) | Statically consolidate the supplied new particle and establish its initial state. | Access the algorithm, other particles, fitness function, or external state. |
| [`update_particle`](#68-update_particle) | Compute, evaluate, and return the identified particle's update. | Modify other particles or rely on persistent changes to worker-local algorithm state. |
| [`post_iteration`](#69-post_iteration) | Finalize selection and population state, remove temporary particles, and update results. | Treat iteration `0` as a regular optimization iteration. |

Optional methods and extensions:

| Component | When to implement or use it |
| --- | --- |
| [`inter_iteration`](#9-optional-second-processing-phase) | A second processing phase depends on the results of the first population-wide phase. |
| [`second_update_particle`](#9-optional-second-processing-phase) | Selected particles need another update after intermediate population-level processing. |
| [`_force_iteration_stop`](#10-optional-forced-termination) | The algorithm detects an internal stopping condition and must prevent subsequent iterations. |
| [`ParticleBase.__call__`](#11-state-persistence-and-migration) | Additional particle-specific state must survive migration or reconstruction. |

Framework-provided operations used by the algorithm:

| Operation | Purpose |
| --- | --- |
| `initialize_context`, `configure` | Establish execution context, wrapped fitness function, and persistent RNG. |
| `update_population` | Merge returned particles by identifier and clear all population random caches (framework-managed). |
| `particle.update` | Evaluate a proposed solution and store its candidate state. |
| `particle.consolidate` | Accept or reject a candidate and clear its transient state. |
| `update_solution_state` | Explicitly refresh current best/worst particles and the retained local best after consolidation. |

### 3. Execution lifecycle

The framework initializes the common execution state through `initialize_context`, configures the wrapped fitness function and persistent random generator through `configure`, and creates the initial population through `create_particle`.

The following flow applies to each algorithm instance. The framework owns the loop and dispatches individual particle operations; the concrete algorithm implements the mathematical operations.

```text
Initialize execution context and configure algorithm
                         |
             Create initial population
                         |
                         v
         +---------------------------------+
    +--->| Another iteration scheduled and |-- No --> End
    |    | force_stop == False?            |
    |    +---------------+-----------------+
    |                    | Yes
    |                    v
    |        Migration control (if applicable)
    |                    |
    |        pre_iteration(actual_iter)
    |                    |
    |                    v
    |          New particles present? ---- No -----+
    |                    | Yes                     |
    |                    v                         |
    |     create_random_cache(new IDs, True)       |
    |     initialize_particle [per particle]       |
    |     consolidate_new_particles [per particle] |
    |     update_population                        |
    |                    |                         |
    |                    +<------------------------+
    |                    |
    |                    v
    |          actual_iter > 0? ---------- No ------------------+
    |                    | Yes                                  |
    |                    v                                      |
    |     create_random_cache(all IDs, False)                   |
    |     update_particle [per particle]                        |
    |     update_population                                     |
    |                    |                                      |
    |                    v                                      |
    |       double_particle_check? ------- No ------------------+
    |                    | Yes                                  |
    |                    v                                      |
    |         inter_iteration(actual_iter)                      |
    |                    |                                      |
    |        double_check_ids not empty? - No ------------------+
    |                    | Yes                                  |
    |                    v                                      |
    |     create_random_cache(selected IDs, False)              |
    |     second_update_particle [per selected particle]        |
    |     update_population                                     |
    |                    |                                      |
    |                    +<-------------------------------------+
    |                    |
    |         post_iteration(actual_iter)
    |                    |
    |         Publish results and iteration status
    |                    |
    +--------------------+
```

Each particle-processing phase completes before its results are merged and the next phase begins. The second update merges only the selected particles; it does not replace the population.

| Iteration | Executed phases |
| --- | --- |
| `0` | Preparation, initialization of new particles, initial consolidation, and finalization. |
| `1` onward | Preparation, initialization of new particles if present, regular update, optional second update, and finalization. |
| Continued execution | Resume from the existing iteration state; use the current `_max_iterations` horizon. |

Iteration `0` is setup, not the first optimization iteration. A force-stop request prevents a new iteration from starting; it does not interrupt the current one. See [iteration numbering](#610-iteration-numbering) and [forced termination](#10-optional-forced-termination) for details.

### 4. Essential invariants

| Topic | Rule | Details |
| --- | --- | --- |
| State ownership | A particle owns only its individual state; the algorithm owns shared optimization state. | [Architecture](#5-architecture-and-state-ownership) |
| Domain | Work exclusively with continuous encoded coordinates and boundaries; never decode user variables inside the algorithm. | [Architecture](#5-architecture-and-state-ownership) |
| Worker isolation | Modify only the identified particle during particle processing. | [Update contract](#68-update_particle) |
| Fitness | Minimize; `+np.inf` is undefined/worst, `-np.inf` is valid/best, and `NaN` is invalid. | [Numerical requirements](#12-mathematical-and-numerical-requirements) |
| Evaluation | Use `particle.update`, never call `_fitness_function` directly. | [Evaluation](#7-fitness-evaluation-and-consolidation) |
| Consolidation | Compare candidates before accepting or rejecting them; preserve the previous state when required. | [Consolidation](#72-candidate-consolidation) |
| Randomness | Use only the particle's cache in worker methods; never remove cache entries or use `pop` in initialization or update logic. | [Random cache](#8-randomness-and-random-cache) |
| Boundaries | Every evaluated coordinate must be within the encoded boundaries. | [Numerical requirements](#12-mathematical-and-numerical-requirements) |
| Migration | Serialize all non-reconstructible persistent particle state; exclude transient candidate and cache state. | [Persistence](#11-state-persistence-and-migration) |
| Results | Remove temporary particles and call `update_solution_state()` on the appropriate consolidated population. | [Finalization](#69-post_iteration) |

---

## Part II — Implementation contracts

### 5. Architecture and state ownership

#### 5.1. Module structure

Implement each algorithm in a dedicated module under `src/tyrannis/algorithm/`. Derive the two concrete classes from the abstractions in [`core/algorithm.py`](../src/tyrannis/core/algorithm.py):

```python
class ExampleParticle(ParticleBase):
    ...


class ExampleAlgorithm(AlgorithmBase[ExampleParticle]):
    ...
```

A dedicated particle class is required even when the algorithm does not need additional particle-specific state.

#### 5.2. Responsibilities

| Component | Owns |
| --- | --- |
| `ParticleBase` | An isolated solution: position, fitness, and individual state such as velocity and personal best. |
| `AlgorithmBase` | Interactions between particles, leaders, archives, distributions, models, adaptive parameters, and selection rules. |
| `Space` | Encoding, decoding, and the interpretation of user-defined variables. |
| Processors and backends | Execution scheduling, worker dispatch, integration of returned particles, and migration where applicable. |

A particle must not hold references to other particles, the population, or the algorithm. The algorithm must not detect, import, or configure execution infrastructure.

#### 5.3. Encoded optimization domain

The algorithm operates exclusively on the framework's continuous internal representation:

| Representation | Type |
| --- | --- |
| Encoded coordinates | `dict[str, float]` |
| Encoded boundaries | `dict[str, tuple[float, float]]` |

Treat these values as continuous coordinates regardless of the original search space.

An architectural compatibility assessment is required before implementing an algorithm that:

- Requires direct access to discrete or categorical semantics.
- Requires inherently sequential updates that observe changes made by previous workers.

Do not silently change the mathematical formulation to fit the framework.

### 6. Initialization and iteration methods

The methods in this section are required by `AlgorithmBase`. Optional processing methods are covered in [Section 9](#9-optional-second-processing-phase).

#### 6.1. `__init__`

**Purpose:** define the public configuration of the algorithm and initialize algorithm-specific state.

- Give every public configuration parameter a suitable default, so that `ExampleAlgorithm()` is valid.
- Validate types, ranges, finiteness, enumerated options, and parameter relationships as early as possible.
- Reject booleans for numeric parameters unless they have an explicit meaning.
- Normalize accepted Python and NumPy numeric types as appropriate.
- Initialize dimension-dependent state and resolve automatic parameters only when the required execution information is available.

Do not silently replace invalid user-supplied values. Do not accept framework-managed parameters such as boundaries, fitness function, population, iteration count, execution seed, or backend.

#### 6.2. `create_particle`

**Contract:** `create_particle(identifier, ...) -> None`

Create a particle and register it in `_population`.

| Argument | Requirement |
| --- | --- |
| `identifier` | The only required argument. |
| `variables` and additional particle state | Optional, with appropriate undefined defaults; names and types must match the particle constructor. |
| `fitness` | Use `fitness: np.float64 = FITNESS_UNDEFINED`. |

Generate missing positions and individual state according to the algorithm. Preserve supplied state when reconstructing a particle, particularly after migration.

Random draws needed for particle creation may use the persistent `_rng`.

**Never evaluate fitness here.** Creation and evaluation are separate operations.

#### 6.3. `delete_particle`

**Contract:** `delete_particle(identifier) -> None`

- Remove the specified particle.
- Maintain any algorithm-specific bookkeeping.
- If `identifier` is `None`, select a particle only when the algorithm defines a meaningful automatic removal policy. Otherwise, reject the request explicitly.

#### 6.4. `pre_iteration`

**Contract:** `pre_iteration(actual_iter) -> None`

Prepare the algorithm and population for the upcoming particle-processing phases.

| Responsibility | Typical operations |
| --- | --- |
| Context-dependent setup | Initialize dimension-dependent structures and validate execution-dependent constraints. |
| Shared optimization state | Update schedules, leaders, archives, distributions, and selection information. |
| Population management | Create or remove particles, temporary candidates, and replacements. |
| Worker preparation | Establish shared reference state that workers will read. |

Migration, when enabled, occurs before this method. Account for arrivals and departures, revalidate population-size constraints where necessary, and incorporate evaluated arrivals into leaders or models when required.

Population-level random operations may use `_rng` directly.

Do not evaluate newly created particles or prepare worker-specific randomness here. The framework detects new particles after this method and invokes the initialization phase.

#### 6.5. `create_random_cache`

**Contract:** `create_random_cache(particle_ids, initialize) -> None`

Prepare random state for exactly the requested identifiers and the immediately following particle-processing phase. Assign each particle its own `random_cache` dictionary.

| Argument | Meaning |
| --- | --- |
| `particle_ids` | Identifiers participating in the next phase. |
| `initialize=True` | Initialization of new particles. |
| `initialize=False` | Regular update or optional second update. |

Algorithms with two update phases must distinguish them using their own phase state. Leave the cache empty when no randomness is required.

See [Section 8](#8-randomness-and-random-cache) for the mandatory randomness rules and examples.

#### 6.6. `initialize_particle`

**Contract:** `initialize_particle(identifier) -> ParticleType`

Prepare the specified new particle for participation in the population and return it.

1. Retrieve the particle and initialize any required individual state.
2. Evaluate its initial solution if its fitness equals `FITNESS_UNDEFINED`.
3. Return the particle with the state needed for subsequent consolidation.

A minimal evaluation pattern is:

```python
particle = self._population[identifier]

if particle.fitness == FITNESS_UNDEFINED:
    particle.update(variables=particle.variables, fitness_function=self._fitness_function)

return particle
```

A newly received particle with an already evaluated fitness other than `FITNESS_UNDEFINED` must not be evaluated again solely because it is marked as new. `ParticleBase` initializes its candidate fields from the supplied evaluated state, allowing the subsequent initial consolidation to preserve that fitness.

Use the particle's random cache for any additional stochastic initialization. Do not consolidate the candidate here.

#### 6.7. `consolidate_new_particles`

**Contract:** `consolidate_new_particles(particle) -> ParticleType`

Implement this method as a `@staticmethod`. It must operate exclusively on the supplied particle.

```python
@staticmethod
def consolidate_new_particles(particle: ExampleParticle) -> ExampleParticle:
    particle.consolidate(consolidate_new=True)
    return particle
```

Establish any particle-specific state that depends on initial consolidation, such as a personal best. The returned particle must be ready to participate in the current iteration.

Do not access the algorithm, configuration, population, fitness function, RNG, or external state. Do not evaluate fitness here.

#### 6.8. `update_particle`

**Contract:** `update_particle(identifier) -> ParticleType`

Compute the transition from the current particle state to its candidate state, evaluate the candidate when required, and return the processed particle.

| Permitted | Prohibited or unsafe |
| --- | --- |
| Read shared algorithm state and other particles as reference information. | Modify other particles, creating race conditions and execution-order dependencies. |
| Modify and return the identified particle. | Rely on changes to other algorithm state persisting after worker execution. |
| Obtain random inputs from the particle's cache. | Access the persistent `_rng` or untracked random sources. |
| Preserve the previous population state until the appropriate consolidation phase. | Consolidate prematurely when other workers need the previous state. |

Keep persistent worker-produced information in the returned particle.

**Particles created during optimization:** a particle initialized during a positive-numbered iteration also participates in the regular update phase. Explicitly skip a redundant update if initialization already performed the evaluation required for that iteration. Do not skip a mathematically distinct evaluation merely because the particle is new.

#### 6.9. `post_iteration`

**Contract:** `post_iteration(actual_iter) -> None`

Finalize the iteration after all particle-processing phases.

- Apply the algorithm's selection and consolidation policy.
- Update persistent population-level state.
- Account for particles already consolidated during an intermediate phase or intentionally left unchanged.
- Remove all temporary particles before returning.
- Call `update_solution_state()` when the population represents the consolidated state that should be reported or retained.

Handle iteration `0` separately wherever a regular update candidate is required: no regular update has occurred at that point.

`update_solution_state()` is inherited but is not called automatically on behalf of the concrete algorithm. It computes the current best and worst particles and updates `local_best` with a deep copy when a strict improvement occurs or when no previous best exists. Keep the population nonempty when invoking it.

#### 6.10. Iteration numbering

Iteration `0` is exclusively the initial setup. It includes preparation, initialization, initial consolidation, and finalization, but neither regular particle updates nor second updates.

Actual optimization iterations run from `1` through the scheduled horizon. For a published zero-based schedule `t = 0, ..., T - 1`, the usual mapping is:

```text
t = actual_iter - 1
```

Derive schedule endpoints from the reference formulation rather than assuming that every schedule uses this convention.

Use `_max_iterations` for the current horizon. The framework may extend it when continuing execution; do not assume every call to `fit` starts at iteration `0`.

### 7. Fitness evaluation and consolidation

#### 7.1. Evaluation contract

Every fitness evaluation initiated by an algorithm must pass through:

```python
particle.update(variables=..., fitness_function=self._fitness_function)
```

This also applies to temporary particles and to multiple candidate evaluations within one processing phase. Never call `_fitness_function` directly.

`ParticleBase.update` stores proposed coordinates in `candidate_variables` and the evaluated result in `candidate_fitness`. It does not modify the current `variables` and `fitness`.

A subsequent update replaces the previous candidate. Preserve intermediate candidates explicitly when the mathematical formulation requires comparisons between them.

#### 7.2. Candidate consolidation

| Particle state | Operation | Result |
| --- | --- | --- |
| Established particle | `consolidate(consolidate_new=True)` | Accept the candidate as the current state. |
| Established particle | `consolidate(consolidate_new=False)` | Reject the candidate and preserve the current state. |
| New particle with undefined fitness | Initial consolidation | Promote the initial candidate to the current state and clear the new-particle flag. |
| New particle with supplied evaluated fitness | Initial consolidation | Preserve the supplied current state and clear the new-particle flag. |

A candidate must exist before consolidation. All of these paths clear the candidate fields after successful consolidation.

Choose the selection policy required by the mathematical formulation: unconditional replacement, strict greedy improvement, elitism, survivor selection, or another justified rule.

Compare fitness **before** consolidation. Do not treat ties as improvements without mathematical justification.

#### 7.3. Evaluation failures

The framework handles evaluation failures using the configured failure strategy. Do not bypass this mechanism or conceal implementation errors as successful evaluations.

### 8. Randomness and random cache

#### 8.1. Why the cache is required

Particle processing may run serially, in threads, in separate processes, or on distributed workers. Random state must be managed independently of the execution infrastructure.

| Direct worker-side randomness problem | Consequence |
| --- | --- |
| Copying the algorithm to workers | RNG state may be duplicated, producing identical or correlated draws. |
| Sharing a mutable RNG across threads | Assignment of random draws may depend on scheduling order. |
| Advancing an algorithm RNG inside a worker | Changes may fail to propagate to the algorithm's owner. |

`random_cache` separates random-state ownership from particle processing. The execution context that owns the algorithm prepares random state before dispatch, so the assignment of random inputs to a particle does not depend on which worker executes it or when it runs.

The persistent `self._rng` is initialized once by the framework during `configure` and advances throughout execution. It is not reinitialized at each iteration.

The goal is reproducibility for the same initial seed and execution conditions, not identical results when migration order, population order, or external fitness-function behavior differs.

```text
Execution seed
      |
      v
Persistent self._rng (advances continuously)
      |
      v
create_random_cache (before each processing phase)
      |
      +--> Particle A: named values or independent generator
      +--> Particle B: named values or independent generator
      +--> Particle C: named values or independent generator
                      |
                      v
              Particle processing
                      |
                      v
              update_population
                 (cache cleared)
```

#### 8.2. Cache representation and preparation

The cache is a temporary `dict[str, Any]` with descriptive, preferably flat keys defined by the concrete algorithm. It is not the cost-function cache and is not persistent particle state.

Choose the representation according to the required randomness:

| Requirement | Preferred representation |
| --- | --- |
| Known, reasonably sized number of draws | Pre-materialized scalars, arrays, or selections. |
| Selection of another particle | Store its identifier or index rather than its object. |
| Dynamic number of draws or excessive pre-materialization | An independent `np.random.Generator` for the particle and phase. |
| No random inputs | An empty dictionary. |

**Example: pre-materialized values**

```python
def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
    for identifier in particle_ids:
        particle = self._population[identifier]
        particle.random_cache = (
            {} if initialize else {"step": self._rng.standard_normal()}
        )
```

The corresponding update reads `particle.random_cache["step"]`.

**Example: independent generator**

Derive a new generator for each relevant particle and phase from fresh state consumed from `_rng`:

```python
entropy = self._rng.integers(0, 2**32, size=4, dtype=np.uint32)
particle.random_cache = {"random_generator": np.random.default_rng(entropy)}
```

The consuming method draws from `particle.random_cache["random_generator"]`. Advancing that generator's internal state is permitted; the generator remains an entry in the cache.

#### 8.3. Mandatory randomness rules

| Rule | Requirement |
| --- | --- |
| Worker methods | Use only cached randomness in `initialize_particle`, `update_particle`, and `second_update_particle`, including their helpers. Never access `_rng`, global NumPy random state, or untracked random sources there. |
| Generator ownership | Never store `_rng` itself in a cache, share mutable generator state between particles, or recreate generators from the original execution seed in each phase or iteration. |
| Access | Retrieve entries by key, not dictionary insertion order. Do not remove entries with `pop` or equivalent operations, or treat the cache as a consumable stack or queue. |
| Particle operations | Initialization and update logic must not use `pop` to remove elements from structures, directly or through helpers. |
| Phase lifetime | Prepare fresh state for each phase. Reuse across phases is valid only when explicitly required by the mathematical formulation. The framework owns cache cleanup. |
| External routines | Pass the appropriate controlled random source to stochastic numerical routines. Assess routines whose randomness cannot be controlled before using them. |

The framework clears the random caches through `update_population` after processing results have been merged.

### 9. Optional second processing phase

Use this mechanism when a second evaluation phase depends on the results of the first population-wide phase. Multiple evaluations of one particle that do not need an intervening population-wide step can remain within a single update method.

#### 9.1. Enabling the phase

In `pre_iteration`, set `self._double_particle_check = True` when `actual_iter == 0`.

Implement the following operations in addition to the regular lifecycle:

| Method | Responsibility |
| --- | --- |
| `inter_iteration(actual_iter) -> None` | Consolidate intermediate results as needed, update shared state, select participants, and assign `self._double_check_ids` (or `self.double_check_ids`). Do not create or remove particles. |
| `create_random_cache(particle_ids, initialize=False)` | Prepare only the selected particles for the second phase, using the state established by `inter_iteration`. |
| `second_update_particle(identifier) -> ParticleType` | Compute and evaluate the second candidate when needed, then return the particle under the same isolation, evaluation, and randomness rules as `update_particle`. |
| `post_iteration(actual_iter) -> None` | Finalize the remaining candidates and shared state, accounting for particles that did not participate in the second phase. |

#### 9.2. Execution rules

- `inter_iteration` runs after the first update results are merged.
- Set the selection list on every invocation, using an empty list when no second update is needed.
- When the list is empty, the framework skips the entire second particle-processing phase, including cache creation and population merging.
- There is no second phase at iteration `0`.

For a single-phase algorithm, leave `double_particle_check` disabled and retain the inherited implementations of `inter_iteration` and `second_update_particle`.

[`BeeColony`](../src/tyrannis/algorithm/bee_colony.py) demonstrates this contract through its employed-bee and onlooker-bee phases.

### 10. Optional forced termination

An algorithm may request termination when it detects an internal stopping condition, such as unrecoverable numerical degeneration.

Set `self._force_iteration_stop = True` and emit a warning informing the user of premature termination. Include the reason whenever it is available:

```python
self._force_iteration_stop = True
warnings.warn(
    "Optimization stopped prematurely: numerical degeneration.",
    RuntimeWarning,
    stacklevel=2,
)
```

| Aspect | Contract |
| --- | --- |
| Request location | An execution stage where algorithm state persists, not a worker-local update. |
| Timing | The framework reads `force_stop` before starting each iteration. It does not interrupt a running iteration or skip its remaining stages. |
| Safe finalization | If requested in `pre_iteration` or `inter_iteration`, ensure subsequent cache creation, updates, and finalization remain safe and preserve a valid population. |
| Lifetime | Latched for the current execution context. Subsequent calls to `fit`, `run`, or `execute` do not clear it. Only `initialize_context` resets it. |
| Distributed scope | Stops the affected island, not every island. |

Do not rely on an exception as a substitute for this mechanism.

[`CMAES`](../src/tyrannis/algorithm/cma_es.py) demonstrates forced termination and safe completion of the current iteration after a numerical failure.

### 11. State persistence and migration

#### 11.1. Serialization contract

Override the particle's `__call__` when additional persistent state is required. Return a dictionary whose keys correspond to the particle constructor and `create_particle` arguments.

| State category | Serialization policy |
| --- | --- |
| Core solution state | Include identifier, variables, and fitness. |
| Additional non-reconstructible state | Include all state necessary to reconstruct the solution without losing its mathematical trajectory, such as PSO velocity and personal best. |
| Reconstructible state | Omit only through a deliberate, documented migration policy. |
| Transient state | Exclude candidate variables, candidate fitness, evaluation errors, and random caches. |

Convert unsupported objects or NumPy arrays to serializable representations. Historical solutions and archives should hold independent snapshots rather than mutable references to particles that will change.

#### 11.2. Population changes

Migration and algorithm-specific creation or deletion may change both the composition and size of the population.

| Count | Meaning |
| --- | --- |
| `_n_particles` | Count refreshed after processor migration control. |
| `len(_population)` | Current dictionary size, potentially including temporary particles. |

Use the count appropriate to the mathematical operation. Revalidate constraints that depend on population size after relevant changes.

#### 11.3. Temporary particles

Temporary candidates must follow the same lifecycle as ordinary particles:

1. Assign collision-free identifiers, preferably incorporating the algorithm identifier and the candidate's purpose.
2. Create and initialize them through the normal lifecycle.
3. Evaluate and consolidate them under the usual particle contract.
4. Remove them before `post_iteration` returns.

Temporary particles must not enter persistent results or migrate to another island.

### 12. Mathematical and numerical requirements

#### 12.1. Fitness semantics

Tyrannis minimizes fitness: lower values are better.

| Fitness value | Interpretation |
| --- | --- |
| Finite value | Valid fitness, compared by minimization. |
| `FITNESS_UNDEFINED` (`+np.inf`) | Undefined or worst fitness. |
| `-np.inf` | Valid; best possible fitness. |
| `NaN` | Invalid. |

Use equality with `FITNESS_UNDEFINED`, not `np.isinf`, to detect undefined fitness.

#### 12.2. Numerical stability and boundaries

| Concern | Implementation requirement |
| --- | --- |
| Extreme and tied fitness | Handle infinities, ties, zero denominators, zero total weights, overflow, and degenerate distributions explicitly. Avoid indeterminate arithmetic such as `inf - inf` and unintended `NaN` propagation. |
| Boundaries | Ensure every evaluated coordinate is within `_boundaries`. `Space` does not repair out-of-bounds candidates. Use clipping, reflection, resampling, or another boundary-handling strategy consistent with the algorithm. |
| Vectorization | Use a stable variable order, for example `tuple(self._boundaries)`, when converting between dictionaries and arrays. Maintain coordinate alignment; prefer `zip(..., strict=True)` where applicable. |
| Selection | Avoid accidental selection pressure from dictionary insertion order. Where the formulation does not prescribe tie-breaking, an unbiased strategy may randomize candidates before a stable sort using the controlled RNG. |
| Numerical primitives | Prefer appropriate SciPy implementations, then NumPy, Python's `math`, and finally manual implementations. NumPy remains the natural choice for elementary and vectorized operations. |

Choose methods, tolerances, and regularizations based on mathematical and numerical justification. Avoid arbitrary epsilon adjustments and unnecessary dependencies.

#### 12.3. Mathematical fidelity

Before implementation, define:

- Equations and order of operations.
- Random distributions and iteration schedules.
- Selection and replacement policies.
- Handling of degeneracies and numerical edge cases.

Keep optimization logic native to Tyrannis rather than delegating the complete algorithm to an external optimizer.

### 13. Public API, documentation, and validation

#### 13.1. Implementation conventions

Use concrete generic types, clear private helper signatures, explicit representations for complex internal models, and the project's Ruff conventions. Avoid unnecessary trailing commas in new Python signatures or calls; follow the project's formatter where it requires them.

Expose relevant public configuration through read-only properties rather than exposing private execution state unnecessarily.

| Exception | Appropriate use |
| --- | --- |
| `TypeError` | Invalid types. |
| `ValueError` | Invalid values. |
| `RuntimeError` | Violated execution invariants. |

#### 13.2. Public constructor docstring

Follow the existing NumPy-style structure. The docstring must describe the optimization algorithm rather than internal framework mechanics.

| Section | Required content |
| --- | --- |
| Introduction | Full name, acronym, and search strategy. |
| Parameters | Every public argument, its type, default, meaning, and constraints. Explain automatic resolution rules. |
| Notes | Mathematical formulation, equations, schedules, selection, variants and their defaults, limitations, and relevant departures from cited formulations. |
| References | At least three academic references: foundational publication first, complementary formulation or analysis, and at least one relevant real-world application. Cite sources of selectable mathematical variants. |

Justify default values through the reference formulation or established literature. Explain differences from cited formulations without discussing internal execution machinery. Prefer peer-reviewed articles or established academic books with full DOI links; provide publication links when no DOI is available.

#### 13.3. Public API integration

- Export the algorithm in [`tyrannis.algorithm`](../src/tyrannis/algorithm/__init__.py) and update `__all__`.
- Update the [README](../README.md)'s Public API and applicable roadmap entries.

#### 13.4. Validation

Validate distinct mathematical and execution contracts with focused tests. Select cases that establish meaningful properties rather than adding redundant cases only for coverage.

| Contract | What to verify |
| --- | --- |
| Initialization and selection | Correct initial state and mathematically justified candidate acceptance or rejection. |
| Mathematics | Expected values derived independently from the implementation under test. |
| Reproducibility | Correct use of controlled random state under equivalent execution conditions. |
| Boundary and numerical cases | Valid evaluated coordinates, extreme values, and algorithm-specific degeneracies. |
| Execution lifecycle and migration | Correct state persistence, reconstruction, and compatibility with supported execution models. |
| Optional processing phases | Correct second-update and forced-termination behavior, when applicable. |

Run the relevant tests and verify Ruff linting, formatting, and basedpyright diagnostics for new or modified test code, following the [CI/CD guide](CI-CD.md).

An algorithm is complete only when its mathematical behavior, particle lifecycle, state persistence, numerical stability, public API, and compatibility with the supported execution models are coherent.
