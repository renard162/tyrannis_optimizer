# Algorithm Development

This document defines the implementation contract for optimization algorithms in Tyrannis Optimizer. It describes how to integrate an algorithm into the execution lifecycle while preserving its mathematical formulation, reproducibility, parallel execution, and migration compatibility.

## 1. Architecture and responsibilities

Implement each algorithm in a dedicated module under `src/tyrannis/algorithm/`, with two classes derived from the abstractions in [`core/algorithm.py`](../src/tyrannis/core/algorithm.py):

```python
class ExampleParticle(ParticleBase):
    ...


class ExampleAlgorithm(AlgorithmBase[ExampleParticle]):
    ...
```

A dedicated particle class is required even if no additional particle state is needed.

- **`ParticleBase`** represents an isolated solution and its individual state, such as position, fitness, velocity, and personal best. A particle must not hold references to other particles, the population, or the algorithm.
- **`AlgorithmBase`** coordinates interactions between particles and owns population-level state, including leaders, archives, distributions, models, adaptive parameters, and selection rules.
- **`Space`** owns encoding, decoding, and the interpretation of user-defined variables. Algorithms operate exclusively on continuous encoded coordinates (`dict[str, float]`) and encoded boundaries (`dict[str, tuple[float, float]]`).
- **Processors and backends** own execution scheduling, worker dispatch, population updates, and, where applicable, migration. The algorithm must not detect, import, or configure execution infrastructure.

An algorithm requiring direct access to discrete or categorical semantics, or inherently sequential updates that must observe previous workers' changes, requires an architectural compatibility assessment before implementation. Do not silently change its mathematical formulation to fit the framework.

## 2. Execution lifecycle

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

Each per-particle phase completes before its results are merged into the population and the next phase begins. `update_population` incorporates returned particles by identifier and clears the random caches. The second update merges only the selected particles, not a replacement population.

### 2.1. Iteration numbering

Iteration `0` is exclusively the initial setup. It includes preparation, initialization, initial consolidation, and finalization, but neither regular particle updates nor second updates.

Actual optimization iterations run from `1` through the scheduled iteration horizon. For a published zero-based schedule `t = 0, ..., T - 1`, the usual mapping is `t = actual_iter - 1`. Derive schedule endpoints from the reference formulation rather than assuming that all schedules use the same convention.

Use `_max_iterations` for the current horizon; the framework may extend it when continuing execution. Do not assume that every call to `fit` starts at iteration `0`.

## 3. Required implementation methods

All methods below are abstract requirements of `AlgorithmBase`. The optional second-phase methods are described separately in Section 6.

### 3.1. `__init__`

Define the public algorithm parameters and initialize algorithm-specific state. All public configuration parameters must have suitable defaults, allowing `ExampleAlgorithm()` without arguments.

Validate types, ranges, finiteness, enumerated options, and parameter relationships immediately when possible. Reject booleans for numeric parameters unless they have an explicit meaning. Normalize accepted Python and NumPy numeric types as appropriate. Do not silently replace invalid user-supplied values.

Do not accept framework-managed parameters such as boundaries, cost function, population, iteration count, execution seed, or backend. Initialize dimension-dependent state and resolve automatic parameters only when the necessary execution information is available.

### 3.2. `create_particle(identifier, ...) -> None`

Create a particle and register it in `_population`. Only `identifier` may be required; all remaining arguments must be optional, match the particle constructor's parameter names and types, and use appropriate undefined defaults.

Use `fitness: np.float64 = FITNESS_UNDEFINED`. Generate missing positions and particle-specific state according to the algorithm, but preserve supplied state when reconstructing a particle, particularly after migration. Random draws made during creation may use the persistent `_rng`.

Never evaluate fitness here. Creation and evaluation are separate operations.

### 3.3. `delete_particle(identifier) -> None`

Remove the specified particle and maintain any algorithm-specific bookkeeping. If `identifier` is `None`, select a particle only when the algorithm defines a meaningful automatic removal policy; otherwise reject the request explicitly.

### 3.4. `pre_iteration(actual_iter) -> None`

Prepare the algorithm and population for the upcoming particle-processing phases. Typical responsibilities include:

- Initialize dimension-dependent structures and validate execution-dependent constraints.
- Update schedules, leaders, archives, distributions, and selection information.
- Create or remove particles, including temporary candidates and replacements.
- Prepare shared reference state that workers will read during particle processing.

Migration, when enabled, occurs before this method. Account for arriving and departing particles and revalidate population-size constraints where necessary. Incorporate evaluated arrivals into leaders or models when required by the algorithm.

Do not evaluate newly created particles or prepare worker-specific randomness here. The framework detects new particles after this method and invokes the corresponding initialization phase. Population-level random operations performed here may use `_rng` directly.

### 3.5. `create_random_cache(particle_ids, initialize) -> None`

Prepare the random state for exactly the specified identifiers and the immediately following processing phase. Assign each particle its own `random_cache` dictionary.

`initialize=True` indicates initialization; `initialize=False` indicates an update, including the optional second update. Algorithms with two update phases must distinguish them using their own phase state. Leave the cache empty when no randomness is required. See Section 5 for the full randomness contract.

### 3.6. `initialize_particle(identifier) -> ParticleType`

Prepare the identified new particle for participation in the population and return it. Initialize individual state and evaluate its initial solution if its fitness equals `FITNESS_UNDEFINED`:

```python
particle = self._population[identifier]

if particle.fitness == FITNESS_UNDEFINED:
    particle.update(
        variables=particle.variables,
        fitness_function=self._fitness_function
    )

return particle
```

A newly received particle with an already evaluated fitness other than `FITNESS_UNDEFINED` must not be evaluated again solely because it is marked as new. Use the particle's random cache for any additional stochastic initialization. Do not consolidate the candidate here.

### 3.7. `consolidate_new_particles(particle) -> ParticleType`

Implement this as a `@staticmethod`. It must operate exclusively on the supplied particle without accessing the algorithm, configuration, population, fitness function, RNG, or external state.

```python
@staticmethod
def consolidate_new_particles(particle: ExampleParticle) -> ExampleParticle:
    particle.consolidate(consolidate_new=True)
    return particle
```

Also establish any particle-specific state that depends on initial consolidation, such as a personal best. The returned particle must be ready for the current iteration. Do not evaluate fitness here.

### 3.8. `update_particle(identifier) -> ParticleType`

Compute the transition from the current particle state to its candidate state, evaluate the candidate when required, and return the processed particle.

Read shared algorithm state and other particles only as reference information. Modify only the identified particle. Changes to other algorithm state made inside a worker are not guaranteed to persist; changes to other particles can introduce race conditions and execution-order dependencies.

Use the particle's random cache for all random operations. Do not prematurely consolidate candidates if other workers must observe the previous population state. Keep persistent worker-produced information in the returned particle.

A particle initialized during a positive-numbered iteration also participates in the regular update phase. If its initialization already performed the evaluation required for that iteration, explicitly skip the redundant update. Do not skip a mathematically distinct evaluation merely because the particle is new.

### 3.9. `post_iteration(actual_iter) -> None`

Finalize the iteration after all particle-processing phases. Implement the algorithm's selection and consolidation policy, update persistent population-level state, and remove temporary particles.

Handle iteration `0` separately wherever the regular update candidate is required. No regular update has occurred at that point. Also account for particles already consolidated during an intermediate phase or intentionally left unchanged.

Call `update_solution_state()` when the population represents the consolidated state that should be reported or retained. This inherited method computes the current best and worst particles and updates `local_best` using a deep copy when a strict improvement occurs (or no previous best exists). It is not called automatically on behalf of the concrete algorithm. Keep the population nonempty when invoking it.

## 4. Fitness evaluation and candidate consolidation

Every fitness evaluation initiated by an algorithm must pass through `particle.update(variables=..., fitness_function=self._fitness_function)`. Never call `_fitness_function` directly, even when evaluating temporary particles or several candidates in one phase.

`ParticleBase.update` stores the proposed coordinates in `candidate_variables` and the evaluated result in `candidate_fitness`. It does not modify the current `variables` and `fitness`. A subsequent update replaces the previous candidate; preserve intermediate candidates explicitly when the formulation requires comparisons between them.

For an established particle, `particle.consolidate(consolidate_new=True)` accepts the candidate and `particle.consolidate(consolidate_new=False)` rejects it. Both clear the candidate state. A candidate must exist before consolidation. On initial consolidation, the new-particle flag is cleared and any previously supplied fitness other than `FITNESS_UNDEFINED` is preserved; otherwise the initial candidate becomes the current state.

Select the policy required by the algorithm: unconditional replacement, strict greedy improvement, elitism, survivor selection, or another mathematically justified rule. Compare fitness before consolidation. Do not treat ties as improvements without justification.

The framework handles evaluation failures through its configured failure strategy. Do not bypass this mechanism or conceal implementation errors as successful evaluations.

## 5. Randomness and `random_cache`

### 5.1. Why the cache is required

Particle processing may execute serially, in threads, in separate processes, or on distributed workers. Copying an algorithm to workers may duplicate its RNG state, causing identical or correlated random draws. Sharing a mutable RNG across threads can instead make the assignment of random draws depend on scheduling order. Worker-side changes to an algorithm's RNG may also fail to propagate back to its owner.

`random_cache` separates random-state ownership from particle processing. The execution context that owns the algorithm prepares the random state before dispatch, so the random inputs assigned to a particle do not depend on which worker executes it or when that worker runs.

The persistent `self._rng` is initialized once by the framework during `configure`. It advances throughout the execution. The objective is reproducibility for the same initial seed and execution conditions, without repeating the same random stream at every iteration. This does not imply identical results when migration order, population order, or external cost-function behavior differs.

### 5.2. Preparing and consuming random state

The cache is a temporary `dict[str, Any]` with descriptive, preferably flat keys defined by the concrete algorithm. It is not the cost-function cache and is not persistent particle state.

Prefer pre-materialized scalars, arrays, or selections whenever the number of draws is known and reasonably sized. For example:

```python
def create_random_cache(self, particle_ids: list[str], initialize: bool) -> None:
    for identifier in particle_ids:
        particle = self._population[identifier]
        particle.random_cache = (
            {} if initialize else {"step": self._rng.standard_normal()}
        )
```

The corresponding update reads `particle.random_cache["step"]`. Prefer storing a selected particle's identifier or index rather than its object.

When draw counts are dynamic or pre-materialization would be excessive, a cache entry may contain an independent `np.random.Generator`. Derive a new generator for each relevant particle and phase from fresh state consumed from `_rng`, for example:

```python
entropy = self._rng.integers(0, 2**32, size=4, dtype=np.uint32)
particle.random_cache = {"random_generator": np.random.default_rng(entropy)}
```

The consuming method can then draw from `particle.random_cache["random_generator"]`. Advancing that generator's internal state is permitted. It remains an entry in the cache.

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

The following rules are mandatory:

- Use only cached randomness in `initialize_particle`, `update_particle`, and `second_update_particle`, including in their helpers. Never access `_rng`, global NumPy random state, or an untracked random source there.
- Never store `_rng` itself in a cache or share mutable generator state between particles. Never recreate generators from the original execution seed in each phase or iteration.
- Access cache entries by key, not insertion order. Do not remove entries with `pop` or equivalent operations; do not treat the cache as a consumable stack or queue. Particle initialization and update logic must not use `pop` to remove elements from structures, directly or through helpers.
- Prepare fresh state for each phase. Reuse across phases is valid only when explicitly required by the mathematical formulation. The framework owns cache cleanup.
- Pass the appropriate controlled random source to stochastic external numerical routines. Assess routines whose randomness cannot be controlled before using them.

## 6. Optional second particle-processing phase

Use this mechanism when a second evaluation phase depends on the results of the first population-wide phase. Multiple evaluations of one particle that do not require an intervening population-wide step can remain within a single update method.

In `pre_iteration`, set `self._double_particle_check = True` when `actual_iter == 0`. Then implement:

| Method | Responsibility |
| --- | --- |
| `inter_iteration(actual_iter) -> None` | Consolidate intermediate results as needed, update shared state, select particles for the next phase, and assign `self._double_check_ids` (or `self.double_check_ids`). Do not create or remove particles. |
| `create_random_cache(particle_ids, initialize=False)` | Prepare only the selected particles for the second phase, using the state established by `inter_iteration`. |
| `second_update_particle(identifier) -> ParticleType` | Compute and evaluate the second candidate when needed, then return the particle under the same isolation, evaluation, and randomness rules as `update_particle`. |
| `post_iteration(actual_iter) -> None` | Finalize the remaining candidates and shared state, accounting for particles that did not participate in the second phase. |

`inter_iteration` runs after the first update results are merged. Set the selection list on every invocation, using an empty list when no second update is needed. The framework skips the entire second particle-processing phase when the list is empty. There is no second phase at iteration `0`.

For a single-phase algorithm, leave `double_particle_check` disabled and retain the inherited implementations of `inter_iteration` and `second_update_particle`.

[`BeeColony`](../src/tyrannis/algorithm/bee_colony.py) demonstrates this contract through its employed-bee and onlooker-bee phases.

## 7. Optional forced termination

If an algorithm detects an internal stopping condition, such as unrecoverable numerical degeneration, it may request termination by setting `self._force_iteration_stop = True` and issuing a warning explaining the reason whenever available:

```python
self._force_iteration_stop = True
warnings.warn(
    "Optimization stopped prematurely: numerical degeneration.",
    RuntimeWarning,
    stacklevel=2
)
```

Make the request from an execution stage where algorithm state persists, not from a worker-local update. The framework reads the `force_stop` property before starting each iteration. It does not interrupt a running iteration or skip its remaining lifecycle stages. If the request is raised in `pre_iteration` or `inter_iteration`, ensure that subsequent cache creation, updates, and finalization remain safe and preserve a valid population. Do not rely on an exception as a substitute for this mechanism.

The request is latched for the lifetime of the current execution context. Subsequent calls to `fit`, `run`, or `execute` do not clear it. Only `initialize_context` resets it. In distributed execution it stops the affected island, not every island.

[`CMAES`](../src/tyrannis/algorithm/cma_es.py) demonstrates forced termination and safe completion of the current iteration after a numerical failure.

## 8. State persistence and migration

Override the particle's `__call__` when additional persistent state is required. Return a dictionary whose keys correspond to the particle constructor and `create_particle` arguments. Include all state necessary to reconstruct the solution without losing its mathematical trajectory, such as PSO velocity and personal best. Convert unsupported objects or NumPy arrays to serializable representations.

Exclude transient state such as candidate variables, candidate fitness, evaluation errors, and random caches. Omit additional reconstructible state only through a deliberate, documented migration policy. Historical solutions and archives should hold independent snapshots, not mutable references to particles that will change.

The population may change in composition and size through migration or algorithm-specific creation and deletion. `_n_particles` is refreshed after processor migration control; `len(_population)` reflects the current dictionary, potentially including temporary particles. Use the count appropriate to the mathematics.

Temporary particles must have collision-free identifiers, preferably incorporating the algorithm identifier and their purpose. They must follow the normal creation, initialization, evaluation, and removal contract. Remove all temporary particles before `post_iteration` returns, so they cannot enter persistent results or migrate to another island.

## 9. Mathematical and numerical requirements

- **Minimization:** Lower fitness is always better. `FITNESS_UNDEFINED` equals `+np.inf`, the worst fitness. `-np.inf` is valid and represents the best possible fitness. Use equality with `FITNESS_UNDEFINED`, not `np.isinf`, to detect undefined fitness. `NaN` is invalid.
- **Extreme values:** Handle infinite and tied fitness, zero denominators, zero total weights, overflow, and degenerate distributions explicitly. Avoid indeterminate arithmetic such as `inf - inf` and unintended `NaN` propagation.
- **Boundaries:** Ensure every evaluated coordinate is within `_boundaries`. `Space` does not repair out-of-bounds candidates. Clipping, reflection, resampling, or other boundary handling must be consistent with the algorithm.
- **Vectorization:** Use one stable variable order, for example `tuple(self._boundaries)`, when converting between dictionaries and arrays. Maintain coordinate alignment and prefer `zip(..., strict=True)` where applicable.
- **Selection:** Do not introduce accidental selection pressure from dictionary insertion order. Where the formulation does not prescribe tie-breaking, an unbiased strategy may randomize candidates before a stable sort using the controlled RNG.
- **Numerical primitives:** Prefer appropriate SciPy implementations, followed by NumPy, Python's `math`, and finally manual implementations. NumPy remains the natural choice for elementary and vectorized operations. Choose methods, tolerances, and regularizations based on mathematical and numerical justification. Avoid arbitrary epsilon adjustments and unnecessary dependencies.
- **Mathematical fidelity:** Define equations, operation order, random distributions, schedules, selection, replacement, and degeneracy handling before implementation. Keep the optimization logic native to Tyrannis rather than delegating the complete algorithm to an external optimizer.

## 10. Public API, documentation, and validation

Use concrete generic types, clear private helper signatures, explicit representations for complex internal models, and the project's Ruff conventions. Do not add trailing commas after the last argument in new Python signatures or calls. Expose read-only properties for relevant public configuration rather than exposing private execution state unnecessarily. Use `TypeError` for invalid types, `ValueError` for invalid values, and `RuntimeError` for violated execution invariants, where appropriate.

The public constructor docstring must describe the optimization algorithm rather than internal framework mechanics. Follow the existing NumPy-style structure:

- **Introduction:** Full name, acronym, and search strategy.
- **Parameters:** Every public argument, its type, default, meaning, and constraints. Defaults must be justified by the reference formulation or established literature; document automatic resolution rules.
- **Notes:** Mathematical formulation, equations, schedules, selection, variants and their defaults, limitations, and relevant departures from the cited formulations. Explain differences without discussing internal execution machinery.
- **References:** At least three academic references. Begin with the foundational publication, followed by a complementary formulation or analysis. Cite the sources of selectable mathematical variants and include at least one relevant real-world application. Prefer peer-reviewed articles or established academic books with full DOI links; provide publication links when a DOI is unavailable.

Export the new algorithm in `tyrannis.algorithm`, update `__all__` and the README's Public API and applicable roadmap entries.

Validate distinct mathematical and execution contracts with focused tests, including initialization, candidate selection, reproducibility, boundary cases, migration, and optional phases when applicable. Mathematical expected results must be derived independently from the implementation. Avoid redundant cases added only for coverage. Run the relevant tests and verify Ruff linting, formatting, and basedpyright diagnostics for new or modified test code.

An algorithm is complete only when its mathematical behavior, particle lifecycle, state persistence, numerical stability, public API, and compatibility with the supported execution models are coherent.
