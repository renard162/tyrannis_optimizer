from functools import partial

from ..core.algorithm import CostFunctionWrapperBase
from ..core.processor import (
    ProcessorBase,
    evaluate_particle,
)
from ..core.signals import LocalEvent


class SerialCostFunctionWrapper(CostFunctionWrapperBase):
    """Serial processor cost-function wrapper."""


class Serial(ProcessorBase):
    def __init__(self) -> None:
        self._cost_function_wrapper = SerialCostFunctionWrapper

    def initialize_execution_context(self) -> None:
        self.start_migration()

    def finalize_execution_context(self) -> None:
        self.stop_migration()

    def initialize_loop_context(self) -> None:
        self._migration_signal = LocalEvent()
        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.initialize_loop_context(
            migration_signal=self._migration_signal,
        )

    def finalize_loop_context(self) -> None:
        if self._migration_processor is None:
            raise RuntimeError("Migration processor cannot be None.")

        self._migration_processor.finalize_loop_context()
        self._migration_signal = None

    def run(self) -> None:
        self.init_particles()
        self.initialize_loop_context()

        try:
            for actual_iter in range(self._n_iter + 1):
                self.migration_control(actual_iter)

                self._algorithm.pre_iteration(actual_iter)
                self.pre_iteration_log(actual_iter)

                new_particles_ids = self._algorithm.new_particles_id
                if new_particles_ids:
                    self._algorithm.create_random_cache(
                        particle_ids=new_particles_ids,
                        initialize=True,
                    )
                    worker = partial(
                        evaluate_particle,
                        algorithm=self._algorithm,
                        fitness_failure_strategy=self._fitness_failure_strategy,
                        initialize_particle=True,
                        second_update=False,
                    )

                    new_particles = [
                        worker(particle_id) for particle_id in new_particles_ids
                    ]
                    self.error_log(
                        actual_iter=actual_iter,
                        updated_particles=new_particles,
                    )
                    new_particles = [
                        self._algorithm.consolidate_new_particles(particle)
                        for particle in new_particles
                    ]
                    self.new_particle_log(
                        actual_iter=actual_iter,
                        new_particles=new_particles,
                    )
                    self._algorithm.update_population(new_particles)

                if actual_iter > 0:
                    self._algorithm.create_random_cache(
                        particle_ids=[idx for idx in self._algorithm.population],
                        initialize=False,
                    )
                    worker = partial(
                        evaluate_particle,
                        algorithm=self._algorithm,
                        fitness_failure_strategy=self._fitness_failure_strategy,
                        initialize_particle=False,
                        second_update=False,
                    )

                    processed_particles = [
                        worker(particle_id)
                        for particle_id in self._algorithm.population
                    ]
                    self.error_log(
                        actual_iter=actual_iter,
                        updated_particles=processed_particles,
                    )
                    self._algorithm.update_population(processed_particles)

                    if self._algorithm.double_particle_check:
                        self._algorithm.inter_iteration(actual_iter)
                        double_check_ids = list(self._algorithm.double_check_ids)

                        if double_check_ids:
                            self._algorithm.create_random_cache(
                                particle_ids=double_check_ids,
                                initialize=False,
                            )
                            worker = partial(
                                evaluate_particle,
                                algorithm=self._algorithm,
                                fitness_failure_strategy=self._fitness_failure_strategy,
                                initialize_particle=False,
                                second_update=True,
                            )

                            processed_particles = [
                                worker(particle_id) for particle_id in double_check_ids
                            ]
                            self.error_log(
                                actual_iter=actual_iter,
                                updated_particles=processed_particles,
                            )
                            self._algorithm.update_population(processed_particles)

                self._algorithm.post_iteration(actual_iter)
                self.iteration_log(actual_iter)

                self.update_status()
                self.best_log(actual_iter)
        finally:
            self.finalize_loop_context()
