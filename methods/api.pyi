"""Canonical steering inference contract; see ../README.md for semantics."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

def load_artifact(artifact_dir: Path, *, model: Any) -> Any:
    """Load a method-native intervention and validate model compatibility."""
    ...

def apply(
    model: Any, artifact: Any, *, strength: float
) -> AbstractContextManager[None]:
    """Apply steering within the context and restore model state on every exit."""
    ...
