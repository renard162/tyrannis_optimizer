# Changelog

## Version 1.1.0

- [Added] : SciPy integration through `ScipyMinimizer`.
- [Added] : `OptimizationHistory`.
- [Added] : Pickle serialization and restoration of optimization history.
- [Removed] : Legacy history methods and properties from `Optimizer`.
- [Changed] : Reduced memory consumption of optimization history.
- [Fixed] : History decoding affecting the optimizer's random state.
- [Fixed] : Handling of signed infinities in CSV history exports.

## Version 1.0.1

- Initial public release.
