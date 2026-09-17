from __future__ import annotations

from pathlib import Path

import pytest

from raq_kv.utils.io import JsonlAppender, read_json, read_jsonl, write_json_atomic


def test_append_and_read(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    with JsonlAppender(path) as w:
        w.append({"a": 1, "text": "çğış"})
        w.append({"a": float("nan")})
    records = read_jsonl(path)
    assert records[0] == {"a": 1, "text": "çğış"} and len(records) == 2


def test_truncated_last_line_is_skipped_and_repaired(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\n{"a": 2, "b"', encoding="utf-8")
    with pytest.warns(UserWarning):
        assert read_jsonl(path) == [{"a": 1}]
    with JsonlAppender(path) as w:
        w.append({"a": 3})
    with pytest.warns(UserWarning):
        assert read_jsonl(path) == [{"a": 1}, {"a": 3}]
    assert '{"a": 2, "b"' in path.read_text(encoding="utf-8")  # damaged bytes are kept for audit
    with JsonlAppender(path) as w:  # a second resume must not add another marker
        w.append({"a": 4})
    with pytest.warns(UserWarning):
        assert read_jsonl(path) == [{"a": 1}, {"a": 3}, {"a": 4}]
    assert path.read_text(encoding="utf-8").count("previous_line_truncated") == 1


def test_corrupted_middle_line_raises(tmp_path: Path):
    path = tmp_path / "r.jsonl"
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_atomic_json(tmp_path: Path):
    path = tmp_path / "sub" / "m.json"
    write_json_atomic(path, {"x": [1, 2]})
    write_json_atomic(path, {"x": [3]})
    assert read_json(path) == {"x": [3]}
    assert not path.with_suffix(".json.tmp").exists()
