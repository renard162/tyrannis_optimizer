import json
import pickle
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import contextmanager
from csv import writer
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, TypedDict, cast, overload, override

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


class _PublishedRecords(Sequence[str]):
    """Read-only view of the published prefix of a Tyrannis-owned record list."""

    __slots__ = ("_count", "_source")

    def __init__(self, records: list[str]) -> None:
        self._publish(records)

    def _publish(self, records: list[str]) -> None:
        self._source = records
        self._count = len(records)

    @override
    def __len__(self) -> int:
        return self._count

    @overload
    def __getitem__(self, index: int) -> str: ...

    @overload
    def __getitem__(self, index: slice) -> list[str]: ...

    @override
    def __getitem__(self, index: int | slice) -> str | list[str]:
        published = range(self._count)
        if isinstance(index, slice):
            return [self._source[i] for i in published[index]]
        return self._source[published[index]]

    @override
    def __iter__(self) -> Iterator[str]:
        index = 0
        while index < self._count:
            yield self._source[index]
            index += 1

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (list, _PublishedRecords)):
            return NotImplemented
        return len(self) == len(other) and all(a == b for a, b in zip(self, other))

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[list[str]]]:
        records = self._source
        if self._count != len(records):
            records = records[: self._count]
        return type(self), (records,)


class OptimizationHistory:
    """
    Store optimization records and interpret them independently of execution.

    This object owns the interpretation of serialized history, including event
    filtering, variable metadata, decoding, and export. It retains no orchestrator,
    algorithm, processor, backend, migration driver, particle population, or
    complete execution-result container. It remains usable after those objects
    and the original search space have been released.

    Parameters
    ----------
    records:
        List of JSON strings in the format produced by ``ProcessorResult.history``.
        Construction takes a shallow copy of the caller's list and publishes
        all its entries without parsing them. Later changes to the input list
        do not affect the history. Internally, the orchestrator shares the
        backend's accumulated list and publishes its current length. The backend
        may append records while the published prefix remains intact. Concurrent
        updates during iteration are not supported.
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
    Use ``history.export("history.csv")`` to write decoded rows, or
    ``history.export("internal_history.csv", internal_state=True)`` to write
    encoded variables and internal particle state. Export writes the selected
    header and consumes the existing generator one row at a time, formatting
    special fitness values without materializing the complete history.
    CSV is the default format and requires a case-insensitive ``.csv`` extension.
    The ``encoding``, ``sep``, and ``internal_state`` arguments apply only to CSV.
    This tabular output cannot be loaded to reconstruct the history object.

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
    The published records, configuration, metadata, decoding snapshot, and its current
    random state can be preserved by standard ``pickle``. No objective callable,
    bound execution method, closure, or generator is stored as a dependency.
    Use ``history.export("history.pkl", format="pickle")`` to serialize the
    complete object and ``OptimizationHistory.load("history.pkl")`` to recover it.
    Pickle accepts case-insensitive ``.pkl`` and ``.pickle`` extensions; the format
    is selected explicitly, never inferred from the extension. Export writes
    directly to a binary file using the highest available Pickle protocol, without
    consuming records, decoding variables, or advancing random state. Loading
    restores the current decoding state without constructing a new snapshot or
    reinitializing spaces, and requires no producing orchestrator.
    Concrete space classes must be importable when unpickling, and user-defined
    choices and decoding parameters must support independent deep copying and
    pickle serialization with unchanged decoding semantics. Local classes, closures,
    or intrinsically non-pickleable user objects are not made serializable here.
    Compatibility across Tyrannis versions is not guaranteed. Load only Pickle
    files from trusted sources: deserialization can execute arbitrary code.
    """

    __slots__: tuple[str, ...] = (
        "_config",
        "_internal_variable_names",
        "_records",
        "_registered_events",
        "_space",
        "_variable_names",
    )

    _records: _PublishedRecords
    _config: HistoryConfig
    _registered_events: tuple[str, ...]
    _space: SpaceBase
    _variable_names: tuple[str, ...]
    _internal_variable_names: tuple[str, ...]

    def __init__(
        self, records: list[str], config: HistoryConfig, space: SpaceBase
    ) -> None:
        """Snapshot the supplied records and capture decoding state and metadata."""
        self._space = space.copy_for_decoding()
        self._variable_names = tuple(self._space.variable_names)
        self._internal_variable_names = tuple(self._space.encoded_variable_names)
        self._config = config
        self._registered_events = tuple(config.registered_events)
        self._records = _PublishedRecords(list(records))

    def _publish_records(self, records: list[str]) -> None:
        """Share the execution-owned list and publish its current prefix."""
        self._records._publish(records)

    def __setstate__(
        self, state: tuple[dict[str, object] | None, dict[str, object]]
    ) -> None:
        attributes, slots = state
        if attributes:
            self.__dict__.update(attributes)
        for name, value in slots.items():
            # Older Pickles stored all published records directly in this slot.
            if name == "_records" and isinstance(value, list):
                value = _PublishedRecords(value)
            setattr(self, name, value)

    @override
    def __repr__(self) -> str:
        """Summarize record and variable counts without parsing stored records."""
        return (
            f"OptimizationHistory(records={len(self._records)}, "
            f"variables={len(self._variable_names)}, "
            f"internal_variables={len(self._internal_variable_names)})"
        )

    @property
    def records(self) -> Sequence[str]:
        """Return a live, read-only view of the published JSON records.

        Holding the view across publications also follows replacement backend
        lists. Getting the view takes constant time and space. Slicing copies
        only the selected entries.
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

    def export(
        self,
        file_location: str | Path,
        format: Literal["csv", "pickle"] = "csv",
        *,
        encoding: str = "utf-8",
        sep: str = ",",
        internal_state: bool = False,
    ) -> None:
        """Export tabular records or serialize the complete history object.

        Parameters
        ----------
        file_location:
            Destination converted to ``Path``. The parent directory must exist
            and an existing destination must be a file. Files are overwritten.
        format:
            Explicit format: ``"csv"`` (default) requires ``.csv``; ``"pickle"``
            requires ``.pkl`` or ``.pickle``. Extensions are case-insensitive
            and never determine the format automatically.
        encoding:
            CSV text encoding, defaulting to ``"utf-8"``. Ignored for Pickle.
        sep:
            Single-character CSV delimiter, defaulting to ``","``. Ignored
            for Pickle.
        internal_state:
            For CSV, export decoded variables by default, or encoded variables
            and internal particle state when ``True``. Ignored for Pickle.

        Raises
        ------
        ValueError
            Unsupported format, invalid extension, an existing destination that
            is not a file, or a CSV separator that is not one character long.
        FileNotFoundError
            The destination's parent directory does not exist.

        Notes
        -----
        CSV writes the header and generates rows on demand. Infinite fitness is
        written as ``"Raise"`` for error events and ``"Infinity"`` otherwise;
        NaN becomes ``"NaN"`` and other fitness values use their string form.
        Decoded CSV export advances only this history's stochastic decoding state.

        Pickle preserves published records, configuration, metadata, and current
        independent decoding state without consuming generators or advancing
        random state.
        It uses ``pickle.HIGHEST_PROTOCOL`` directly on the binary file. Recover
        the complete object with ``load``; CSV cannot reconstruct the history.
        User-defined objects must support standard Pickle, and their classes must
        be importable when loading. Local classes and closures are not supported.
        Compatibility across Tyrannis versions is not guaranteed. Load only
        trusted Pickle files, since deserialization can execute arbitrary code.
        File, CSV, parsing, decoding, and serialization errors propagate; failures
        can leave a partially written file.
        """
        if format not in {"csv", "pickle"}:
            raise ValueError(f"Unsupported history export format: {format!r}")
        if format == "csv":
            self._export_csv(file_location, encoding, sep, internal_state)
        elif format == "pickle":
            self._export_pickle(file_location)

    @classmethod
    def load(cls, file_location: str | Path) -> Self:
        """Recover a complete history from a trusted Pickle file.

        Parameters
        ----------
        file_location:
            Source converted to ``Path``, with a case-insensitive ``.pkl`` or
            ``.pickle`` extension. CSV import is not supported.

        Returns
        -------
        Self
            The deserialized object, with its records, configuration, metadata,
            and current independent decoding state, ready for public access and
            export without the producing orchestrator. Neither the history
            constructor nor space copying or initialization is performed.

        Raises
        ------
        ValueError
            The extension is not ``.pkl`` or ``.pickle``.
        TypeError
            The recovered object is not an instance of ``cls``.

        Notes
        -----
        Only load files from trusted sources. ``pickle.load`` can execute
        arbitrary code before the type check; that check is not a security
        mechanism. File and deserialization errors propagate. User-defined
        classes must be importable and their objects must support standard
        Pickle. Compatibility across Tyrannis versions is not guaranteed.
        Random states are restored as saved, without resetting them or decoding
        records; original stochastic objective inputs are not reconstructed.
        """
        file_location = Path(file_location)
        if file_location.suffix.lower() not in {".pkl", ".pickle"}:
            raise ValueError(
                f"History file must have a '.pkl' or '.pickle' extension: {file_location}"
            )

        with file_location.open("rb") as file:
            # Pickle has no static schema; validate the recovered object below.
            history = cast(object, pickle.load(file))
        if not isinstance(history, cls):
            raise TypeError(f"History file must contain an instance of {cls.__name__}.")
        return history

    def _export_pickle(self, file_location: str | Path) -> None:
        """Serialize this instance directly, preserving its current decoding state.

        Require a case-insensitive ``.pkl`` or ``.pickle`` extension and the same
        destination policy as CSV. Standard Pickle uses the components' existing
        serialization mechanisms without materializing rows. A fully published
        record list is serialized directly.
        """
        file_location = Path(file_location)
        if file_location.suffix.lower() not in {".pkl", ".pickle"}:
            raise ValueError(
                f"History file must have a '.pkl' or '.pickle' extension: {file_location}"
            )

        if file_location.exists() and not file_location.is_file():
            raise ValueError(f"History file location is not a file: {file_location}")

        if not file_location.parent.exists():
            raise FileNotFoundError(
                f"Parent directory does not exist: {file_location.parent}"
            )

        with file_location.open("wb") as file:
            pickle.dump(self, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _export_csv(
        self,
        file_location: str | Path,
        encoding: str = "utf-8",
        sep: str = ",",
        internal_state: bool = False,
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
