from ..core.orchestrator import OrchestratorBase


class ScipyMinimizer(OrchestratorBase):
    """Independent Tyrannis orchestrator intended for use as a custom method
    in ``scipy.optimize.minimize()``.

    The planned integration will use the ``Continuous`` search space exclusively.
    """
