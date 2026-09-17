"""Repository-relative path discovery. No absolute paths are hard-coded anywhere in the project."""

from __future__ import annotations

from pathlib import Path


def repo_root(start: Path | None = None) -> Path:
    """Return the nearest ancestor directory that contains ``pyproject.toml``."""
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise FileNotFoundError(f"pyproject.toml not found above {here}")


def configs_dir(root: Path | None = None) -> Path:
    return (root or repo_root()) / "configs"


def resolve_output_root(output_root: str, root: Path | None = None) -> Path:
    """Resolve the configured output root; relative paths are anchored at the repository root."""
    path = Path(output_root)
    return path if path.is_absolute() else (root or repo_root()) / path
