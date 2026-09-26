"""YAML config loading.

Resolution order for one experiment file:
  1. ``include:`` files (paths relative to ``configs/``), merged in order, recursively;
  2. the file's own keys (override includes);
  3. CLI overrides ``a.b.c=value`` (values parsed as YAML);
  4. ``sweep:`` expansion — ``{dotted.key: [v1, v2, ...]}`` produces the Cartesian product;
  5. validation into :class:`ExperimentConfig`.
"""

from __future__ import annotations

import copy
import itertools
from pathlib import Path
from typing import Any

import yaml

from raq_kv.config.schema import ExperimentConfig
from raq_kv.utils.paths import configs_dir


def load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_include_path(entry: str, including_file: Path, root: Path) -> Path:
    candidate = Path(entry)
    if candidate.is_absolute():
        return candidate
    for base in (root, including_file.parent):
        if (base / candidate).is_file():
            return base / candidate
    raise FileNotFoundError(f"include '{entry}' not found (searched {root} and {including_file.parent})")


def load_with_includes(path: Path, root: Path | None = None, _stack: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"include cycle: {' -> '.join(map(str, (*_stack, path)))}")
    root = root or configs_dir()
    data = load_yaml(path)
    includes = data.pop("include", []) or []
    if isinstance(includes, str):
        includes = [includes]
    merged: dict[str, Any] = {}
    for entry in includes:
        merged = deep_merge(merged, load_with_includes(_resolve_include_path(entry, path, root), root, (*_stack, path)))
    return deep_merge(merged, data)


def set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    node = data
    for part in parts[:-1]:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ValueError(f"cannot set '{dotted_key}': '{part}' is not a mapping")
        node = child
    node[parts[-1]] = value


def apply_overrides(data: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = copy.deepcopy(data)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override '{item}' must look like key.path=value")
        key, raw = item.split("=", 1)
        set_dotted(out, key.strip(), yaml.safe_load(raw))
    return out


def expand_sweep(data: dict[str, Any]) -> list[dict[str, Any]]:
    data = copy.deepcopy(data)
    sweep = data.pop("sweep", None) or {}
    if not sweep:
        return [data]
    keys = list(sweep)
    for key in keys:
        if not isinstance(sweep[key], list) or not sweep[key]:
            raise ValueError(f"sweep '{key}' must be a non-empty list")
    variants = []
    for combo in itertools.product(*(sweep[k] for k in keys)):
        variant = copy.deepcopy(data)
        for key, value in zip(keys, combo, strict=True):
            set_dotted(variant, key, value)
        variants.append(variant)
    return variants


def load_experiment_configs(
    path: Path, overrides: list[str] | None = None, root: Path | None = None
) -> list[ExperimentConfig]:
    """Includes, then overrides, then sweep expansion -- and an override the sweep would erase is refused.

    The order is what makes the refusal necessary. `expand_sweep` writes each swept key with `set_dotted`, so an
    override touching a path **under** a swept key is applied and then silently overwritten. Concretely,
    ``--set task.n_samples=2`` on a config whose sweep lists whole ``task`` dicts does nothing at all, with no
    warning: it looks like a 2-sample smoke run and is the full campaign. That cost 15 minutes of GPU on D67's
    calibration, and on a longer config it would have cost a great deal more -- the failure is invisible in the
    output, because the run that happens is a perfectly valid one that nobody asked for.
    """
    raw = apply_overrides(load_with_includes(path, root), overrides or [])
    swept = set(raw.get("sweep") or {})
    for item in overrides or []:
        head = item.split("=", 1)[0].split(".")[0]
        if head in swept:
            raise ValueError(
                f"override {item!r} targets {head!r}, which this config sweeps: sweep expansion rewrites that key "
                f"afterwards, so the override would be silently discarded. Edit the sweep entries, or use a config "
                f"without a sweep on {head!r}.")
    return [ExperimentConfig.model_validate(variant) for variant in expand_sweep(raw)]
