"""Canonical steering inference contract. See README.md for semantics."""

from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

def load_intervention(intervention_dir: Path, *, model: Any) -> Any:
    """Load a method-native intervention and validate model compatibility."""
    ...

def apply(
    model: Any, intervention: Any, *, strength: float
) -> AbstractContextManager[None]:
    """Apply steering within the context and restore model state on every exit."""
    ...
