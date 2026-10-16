"""Shared evaluation records and contracts; implementations are not yet present."""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict

MethodName = Literal["baseline", "caa", "icv", "sae", "reft_r1"]
BenchmarkName = Literal["bbq", "holistic_bias", "mmlu_pro", "alpaca_eval"]

class Example(TypedDict):
    example_id: str
    data: dict[str, Any]  # Native fields, including references and scoring metadata.

class Prediction(TypedDict):
    example_id: str
    output: str

class Metric(TypedDict):
    subgroup: str
    metric: str
    value: float | None
    n: int

def load_examples(config: Mapping[str, Any]) -> list[Example]:
    """Load the configured split with stable, unique example IDs."""
    ...

def format_prompt(example: Example, config: Mapping[str, Any]) -> str:
    """Format model input; exclude reference answers and scoring-only metadata."""
    ...

def score(
    examples: Sequence[Example],
    predictions: Sequence[Prediction],
    config: Mapping[str, Any],
) -> list[Metric]:
    """Score after checking exact, unique example-ID coverage; never generate."""
    ...

def generate(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    *,
    decoding: Mapping[str, Any],
    seed: int,
) -> list[str]:
    """Return raw continuations, excluding input prompts, in input order."""
    ...

def run(
    config_path: Path,
    *,
    method: MethodName,
    benchmark: BenchmarkName,
    run_id: str,
) -> Path:
    """Save config, predictions, and metrics in a new run directory; return it."""
    ...
