"""Append-only JSONL results and atomic JSON writes.

Raw results are the source of truth for every table and figure, so the writer never truncates or
rewrites an existing results file. A process killed mid-write can leave a partial last line. The next
appender terminates it and writes a marker record right after it, so the damaged line stays visible for
audit, while the reader skips exactly those marked lines. Any other malformed line is a hard error.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from pathlib import Path
from typing import Any

MARKER_KEY = "__raq_kv_marker__"
TRUNCATED_MARKER = {MARKER_KEY: "previous_line_truncated"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if hasattr(obj, "item") and callable(obj.item):  # numpy / torch scalars
        return obj.item()
    if hasattr(obj, "tolist") and callable(obj.tolist):  # numpy arrays / torch tensors
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dumps(obj: Any) -> str:
    """Serialize with UTF-8 characters preserved. NaN/Inf are allowed (e.g. perplexity under faults)."""
    return json.dumps(obj, ensure_ascii=False, default=_json_default, allow_nan=True)


class JsonlAppender:
    """Append records to a JSONL file; each record is flushed (and optionally fsynced) immediately."""

    def __init__(self, path: Path, fsync: bool = True) -> None:
        self.path = Path(path)
        self.fsync = fsync
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8", newline="\n")  # noqa: SIM115
        self._terminate_partial_line()

    def _terminate_partial_line(self) -> None:
        if self.path.stat().st_size == 0:
            return
        with open(self.path, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            last = fh.read(1)
        if last != b"\n":
            self._fh.write("\n" + dumps(TRUNCATED_MARKER) + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())

    def append(self, record: dict[str, Any]) -> None:
        self._fh.write(dumps(record) + "\n")
        self._fh.flush()
        if self.fsync:
            os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> JsonlAppender:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all complete records (marker records are not returned).

    A malformed line is tolerated only if it is the unterminated last line (crash during the current write)
    or is immediately followed by the truncation marker (crash during an earlier run, repaired on resume).
    """
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    records: list[dict[str, Any]] = []
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            is_unterminated_tail = i == len(lines) - 1 and not raw.endswith("\n")
            next_is_marker = i + 1 < len(lines) and lines[i + 1].strip() == dumps(TRUNCATED_MARKER)
            if is_unterminated_tail or next_is_marker:
                warnings.warn(f"Skipping truncated line {i + 1} in {path}", stacklevel=2)
                continue
            raise ValueError(f"Corrupted JSONL record at {path}:{i + 1}") from None
        if isinstance(record, dict) and MARKER_KEY in record:
            continue
        records.append(record)
    return records


def write_json_atomic(path: Path, obj: Any) -> None:
    """Write JSON via a temporary file and ``os.replace`` so readers never observe a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=_json_default, allow_nan=True)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def finite_or_none(x: float) -> float | None:
    """Helper for summary tables that must stay strict JSON."""
    return x if math.isfinite(x) else None
