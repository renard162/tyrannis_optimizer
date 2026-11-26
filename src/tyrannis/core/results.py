import json
from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from csv import writer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, TypedDict, cast, override

import numpy as np

from .space import SpaceBase


@dataclass(frozen=True, slots=True)
class HistoryConfig:
    migration: bool = False
    pre_iteration: bool = False
    new_particle: bool = False
    error: bool = False
    iteration: bool = False
    status: bool = False
    best: bool = False

    EVENT: ClassVar[dict[str, str]] = {
        "migration": "0 - migration",
        "pre_iteration": "1 - pre_iteration",
        "new_particle": "2 - new_particle",
        "error": "3 - error",
        "iteration": "4 - iteration",
        "iter_best": "5 - iter_best",
        "iter_worst": "5 - iter_worst",
        "local_best": "6 - local_best",
    }

    @classmethod
    def get_event(cls, event_type: str) -> str:
        return cls.EVENT[event_type]

    @property
    def history_enabled(self) -> bool:
        return (
            self.migration
            or self.pre_iteration
            or self.new_particle
            or self.error
            or self.iteration
            or self.status
            or self.best
        )

    @property
    def registered_events(self) -> list[str]:
        return list(self.EVENT.values())


@dataclass
class ProcessorResult:
    result: dict[str, Any] | None = None
    history: list[str] = field(default_factory=list)

    @contextmanager
    def record_iteration(self) -> Generator[None, None, None]:
        """Publish the best and history together when an iteration completes."""
        best = self.result
        history_size = len(self.history)
        try:
            yield
        except BaseException:
            self.result = best
            del self.history[history_size:]
            raise


class _HistoryParticle(TypedDict, total=False):
    """Standard particle fields; JSON may also contain algorithm-specific state."""

    identifier: str
    variables: dict[str, float]
    fitness: float


class _HistoryRecord(TypedDict, total=False):
    """Fields read from the existing serialized event envelope."""

    event: str
    iteration: int
    origin: str
    destination: str
    particle: _HistoryParticle


class OptimizationHistory:
    """
    Store optimization records and interpret them independently of execution.

    This object owns the interpretation of serialized history, including event
    filtering, variable metadata, decoding, and CSV export. It retains no orchestrator,
    algorithm, processor, backend, migration driver, particle population, or
    complete execution-result container. It remains usable after those objects
    and the original search space have been released.

    Parameters
    ----------
    records:
        List of JSON strings in the format produced by ``ProcessorResult.history``.
        The list is retained by reference, without copying or parsing its entries
        during construction. Pass an empty list for an empty history. The caller
        may append or extend this same list, including between executions that
        use the same variable schema and decoding configuration. Clearing or
        replacing its entries also affects this history. Concurrent mutation
        during iteration is not supported.
    config:
        Immutable history configuration used when recording the events. The
        configuration and a snapshot of its registered event labels are retained.
        Flags control recording upstream; reading filters by registered labels,
        rather than applying those flags again.
    space:
        Initialized search space corresponding to the recorded internal variables.
        Its ``copy_for_decoding`` contract supplies a separate decoding snapshot;
        the original space is not retained or modified. Unsupported custom spaces
        must implement that contract or explicitly declare their dependencies.

    Record organization
    -------------------
    Each JSON object contains an ``event``, ``iteration``, ``origin``, and a
    ``particle`` dictionary; migration events also contain ``destination``.
    The particle contains ``identifier``, ``fitness``, and ``variables`` in the
    continuous solver representation. Other particle fields are algorithm-specific
    internal state. Records retain their original strings and list order. Unknown
    events are skipped. JSON parsing and decoding errors propagate when the
    corresponding record is consumed; construction does not validate all records.

    Representations
    ---------------
    Both representations start with ``event``, ``iteration``, ``identifier``,
    ``origin``, ``destination``, and ``fitness``, using ``None`` for absent metadata.
    ``columns`` and ``generator`` expose decoded variables in the search space's
    order. Dictionary decoding contributes its values in insertion order;
    positional decoding contributes its elements. A sequence or permutation
    remains a single cell for its corresponding decoded variable.

    ``internal_columns_`` and ``internal_generator_`` expose continuous variables
    in encoded-boundary order, using ``None`` for missing variables, followed by
    ``internal_state``. That last cell is a JSON string containing all particle
    fields except ``identifier``, ``variables``, and ``fitness``, or ``None`` when
    no such fields exist. Internal rows do not invoke the decoder.

    Each generator property creates a fresh lazy iterator; no active generator or
    tabular representation is stored. Empty histories yield no rows but still
    expose their columns. Memory consists of the shared record list, small metadata
    snapshots, independent decoding data, and the current record and row for each
    active iterator. Column properties return new lists.

    CSV export
    ----------
    Use ``history.save_csv("history.csv")`` to write decoded rows, or
    ``history.save_csv("internal_history.csv", internal_state=True)`` to write
    encoded variables and internal particle state. Export writes the selected
    header and consumes the existing generator one row at a time, formatting
    special fitness values without materializing the complete history.

    Decoding isolation
    ------------------
    The snapshot retains only explicitly selected decoding dependencies, including
    effective bounds, choices, decoder parameters, and independent random state.
    It excludes the objective function, fitness caches, and resources used solely
    for evaluation. ``Mixed`` snapshots its initialized component spaces recursively;
    ``Sequence`` snapshots its initialized categorical or ordinal base space and
    preserves its length limits and STOP semantics. No space is reinitialized and
    existing decoding methods perform every conversion.

    With stochastic decoders, reconstructed values can differ from those originally
    passed to the objective function. Decoded reads advance the history's own random
    state, so repeated or interleaved reads can also differ. Neither construction
    nor reading advances the original optimization's random generators. The history
    does not reconstruct the actual sampled inputs from the original evaluations.

    Serialization
    -------------
    The stored records, configuration, metadata, decoding snapshot, and its current
    random state can be preserved by standard ``pickle``. No objective callable,
    bound execution method, closure, or generator is stored as a dependency.
    Concrete space classes must be importable when unpickling, and user-defined
    choices and decoding parameters must support independent deep copying and
    pickle serialization with unchanged decoding semantics. Local classes, closures,
    or intrinsically non-pickleable user objects are not made serializable here.
    """

    __slots__: tuple[str, ...] = (
        "_config",
        "_internal_variable_names",
        "_records",
        "_registered_events",
        "_space",
        "_variable_names",
    )

    _records: list[str]
    _config: HistoryConfig
    _registered_events: tuple[str, ...]
    _space: SpaceBase
    _variable_names: tuple[str, ...]
    _internal_variable_names: tuple[str, ...]

    def __init__(
        self, records: list[str], config: HistoryConfig, space: SpaceBase
    ) -> None:
        """Retain the records and capture initialized decoding state and metadata."""
        self._space = space.copy_for_decoding()
        self._variable_names = tuple(self._space.variable_names)
        self._internal_variable_names = tuple(self._space.encoded_variable_names)
        self._config = config
        self._registered_events = tuple(config.registered_events)
        self._records = records

    @override
    def __repr__(self) -> str:
        """Summarize record and variable counts without parsing stored records."""
        return (
            f"OptimizationHistory(records={len(self._records)}, "
            f"variables={len(self._variable_names)}, "
            f"internal_variables={len(self._internal_variable_names)})"
        )

    @property
    def records(self) -> list[str]:
        """Return the shared JSON record list for accumulation without copying.

        Appended records must use this history's variable schema and decoder
        configuration. Mutations are visible to subsequent history reads.
        """
        return self._records

    @property
    def config(self) -> HistoryConfig:
        """Return the immutable event configuration supplied at construction."""
        return self._config

    @property
    def columns(self) -> list[str]:
        """Return metadata columns followed by decoded variable names."""
        return [
            "event",
            "iteration",
            "identifier",
            "origin",
            "destination",
            "fitness",
            *self._variable_names,
        ]

    @property
    def generator(self) -> Iterator[list[object]]:
        """Yield registered records as decoded rows, decoding only on consumption.

        Dictionary values and positional elements retain the decoder's order.
        Stochastic reads advance only the independent decoding snapshot.
        """
        for data in self._iter_records():
            particle = data.get("particle", {})
            variables = particle.get("variables", {})
            # SpaceBase.decode is typed as Any for heterogeneous space outputs;
            # the supported spaces return positional lists or named dictionaries.
            decoded = cast(
                list[object] | dict[str, object], self._space.decode(variables)
            )
            values: Iterable[object] = (
                decoded.values() if isinstance(decoded, dict) else decoded
            )
            yield [
                data.get("event"),
                data.get("iteration"),
                particle.get("identifier"),
                data.get("origin"),
                data.get("destination"),
                particle.get("fitness"),
                *values,
            ]

    @property
    def internal_columns_(self) -> list[str]:
        """Return metadata, encoded variable names, and the internal-state column."""
        return [
            "event",
            "iteration",
            "identifier",
            "origin",
            "destination",
            "fitness",
            *self._internal_variable_names,
            "internal_state",
        ]

    @property
    def internal_generator_(self) -> Iterator[list[object]]:
        """Yield registered records as internal rows without invoking decoding.

        Missing variables become ``None``. Additional particle fields form a
        JSON ``internal_state`` cell, or ``None`` if no such fields exist.
        """
        for data in self._iter_records():
            particle = data.get("particle") or {}
            variables = particle.get("variables") or {}
            internal_state = {
                key: value
                for key, value in particle.items()
                if key not in {"identifier", "variables", "fitness"}
            }
            yield [
                data.get("event"),
                data.get("iteration"),
                particle.get("identifier"),
                data.get("origin"),
                data.get("destination"),
                particle.get("fitness"),
                *(variables.get(name) for name in self._internal_variable_names),
                json.dumps(internal_state) if internal_state else None,
            ]

    def save_csv(
        self,
        file_location: str | Path,
        encoding: str = "utf-8",
        sep: str = ",",
        internal_state: bool = False
    ) -> None:
        """Save registered history rows to CSV, generating each row on demand.

        Parameters
        ----------
        file_location:
            Destination converted to ``Path``. Its extension must be ``.csv``
            (case-insensitive), its parent directory must exist, and an existing
            destination must be a file. Existing files are overwritten.
        encoding:
            Text encoding used to open the file. Defaults to ``"utf-8"``.
        sep:
            Single-character CSV delimiter. Defaults to ``","``.
        internal_state:
            If ``False`` (default), use ``columns`` and ``generator`` to export
            decoded variables. If ``True``, use ``internal_columns_`` and
            ``internal_generator_`` to export encoded variables and internal
            particle state without decoding.

        Raises
        ------
        ValueError
            The extension is not ``.csv``, an existing destination is not a file,
            or the separator does not contain exactly one character.
        FileNotFoundError
            The destination's parent directory does not exist.

        Notes
        -----
        The header is written even when there are no rows. Records are consumed
        lazily through the selected generator, without materializing the history.
        Infinite fitness is written as ``"Raise"`` for error events and
        ``"Infinity"`` otherwise; NaN is written as ``"NaN"``, and other fitness
        values use their string representation. File, CSV, parsing, and decoding
        errors propagate; a failure can leave a partially written file.
        Decoded export uses this history's independent decoding state, including
        its stochastic behavior, and requires no producing orchestrator.
        """

        file_location = Path(file_location)

        if file_location.suffix.lower() != ".csv":
            raise ValueError(
                f"History file must have a '.csv' extension: {file_location}"
            )

        if file_location.exists() and not file_location.is_file():
            raise ValueError(f"History file location is not a file: {file_location}")

        if not file_location.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {file_location.parent}"
            )

        if len(sep) != 1:
            raise ValueError(f"CSV separator must be a single character: {sep!r}")
        columns = self.internal_columns_ if internal_state else self.columns
        row_generator = self.internal_generator_ if internal_state else self.generator
        fitness_index = columns.index("fitness")
        error_event = self._config.get_event("error")

        with file_location.open("w", newline="", encoding=encoding) as file:
            csv_writer = writer(file, delimiter=sep)
            csv_writer.writerow(columns)

            for row in row_generator:
                # The heterogeneous row loses the per-column type; recorded
                # particle fitness follows the _HistoryParticle float contract.
                fitness = cast(float, row[fitness_index])

                if np.isinf(fitness) and (row[0] == error_event):
                    row[fitness_index] = "Raise"
                elif np.isinf(fitness):
                    row[fitness_index] = "Infinity"
                elif np.isnan(fitness):
                    row[fitness_index] = "NaN"
                else:
                    row[fitness_index] = str(fitness)

                csv_writer.writerow(row)

    def _iter_records(self) -> Iterator[_HistoryRecord]:
        """Parse one record at a time and retain only registered event labels."""
        for entry in self._records:
            # json.loads cannot express the schema of component-produced records.
            # This annotation describes that schema without adding eager validation.
            data = cast(_HistoryRecord, json.loads(entry))
            if data.get("event") in self._registered_events:
                yield data
