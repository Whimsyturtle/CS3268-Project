"""Validated readers for completed evaluation and tuning runs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

METHODS = {"baseline", "caa", "icv", "sae", "reft_r1"}
BENCHMARKS = {"bbq", "holistic_bias", "mmlu_pro", "alpaca_eval"}


class ExplorerDataError(ValueError):
    """Raised when saved experiment data violates its declared contract."""


@dataclass(frozen=True)
class RunIdentity:
    benchmark: str
    method: str
    run_id: str
    model_name: str
    model_revision: str
    dataset: str
    dataset_revision: str
    split: str
    source_commit: str
    decoding: dict[str, Any]
    seed: int | None
    strength: float | None


@dataclass(frozen=True)
class LoadedRun:
    path: Path
    identity: RunIdentity
    config: dict[str, Any]
    predictions: tuple[dict[str, Any], ...]
    metrics: tuple[dict[str, Any], ...]


def _within(root: Path, candidate: Path) -> Path:
    resolved_root = root.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ExplorerDataError(f"Path is outside configured root: {candidate}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExplorerDataError(f"Cannot read JSON from {path}: {error}") from error
    if not isinstance(value, dict):
        raise ExplorerDataError(f"Expected a JSON object in {path}")
    return value


def _required_string(mapping: dict[str, Any], key: str, source: Path) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ExplorerDataError(f"Expected non-empty string {key!r} in {source}")
    return value


def _identity(config: dict[str, Any], path: Path) -> RunIdentity:
    benchmark = _required_string(config, "benchmark", path)
    method = _required_string(config, "method", path)
    run_id = _required_string(config, "run_id", path)
    if benchmark not in BENCHMARKS or method not in METHODS:
        raise ExplorerDataError(f"Unsupported benchmark or method in {path}")
    model = config.get("model")
    benchmark_config = config.get("benchmark_config")
    method_config = config.get("method_config")
    source = config.get("source")
    decoding = config.get("decoding")
    if not all(isinstance(item, dict) for item in (model, benchmark_config, method_config, source, decoding)):
        raise ExplorerDataError(f"Incomplete resolved configuration in {path}")
    strength = None if method == "baseline" else method_config.get("strength")
    if strength is not None and (not isinstance(strength, (int, float)) or isinstance(strength, bool)):
        raise ExplorerDataError(f"Invalid strength in {path}")
    seed = benchmark_config.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ExplorerDataError(f"Invalid seed in {path}")
    return RunIdentity(
        benchmark=benchmark,
        method=method,
        run_id=run_id,
        model_name=_required_string(model, "name", path),
        model_revision=_required_string(model, "revision", path),
        dataset=_required_string(benchmark_config, "dataset", path),
        dataset_revision=_required_string(benchmark_config, "revision", path),
        split=_required_string(benchmark_config, "split", path),
        source_commit=_required_string(source, "commit", path),
        decoding=dict(decoding),
        seed=seed,
        strength=None if strength is None else float(strength),
    )


def _load_predictions(path: Path) -> tuple[dict[str, Any], ...]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ExplorerDataError(f"Cannot read predictions from {path}: {error}") from error
    for line_number, line in enumerate(lines, 1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ExplorerDataError(f"Invalid JSON on {path}:{line_number}") from error
        if not isinstance(record, dict):
            raise ExplorerDataError(f"Expected an object on {path}:{line_number}")
        example_id = record.get("example_id")
        output = record.get("output")
        if not isinstance(example_id, str) or not isinstance(output, str):
            raise ExplorerDataError(f"Invalid prediction on {path}:{line_number}")
        if example_id in seen:
            raise ExplorerDataError(f"Duplicate example_id {example_id!r} in {path}")
        seen.add(example_id)
        records.append({"example_id": example_id, "output": output})
    return tuple(records)


def _load_metrics(path: Path) -> tuple[dict[str, Any], ...]:
    metrics: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames != ["subgroup", "metric", "value", "n"]:
                raise ExplorerDataError(f"Invalid metric columns in {path}")
            for row_number, row in enumerate(reader, 2):
                try:
                    value = None if row["value"] == "" else float(row["value"])
                    count = int(row["n"])
                except (TypeError, ValueError) as error:
                    raise ExplorerDataError(f"Invalid metric on {path}:{row_number}") from error
                if count < 0 or not row["subgroup"] or not row["metric"]:
                    raise ExplorerDataError(f"Invalid metric on {path}:{row_number}")
                metrics.append({"subgroup": row["subgroup"], "metric": row["metric"], "value": value, "n": count})
    except OSError as error:
        raise ExplorerDataError(f"Cannot read metrics from {path}: {error}") from error
    return tuple(metrics)


def load_run(path: Path, *, results_root: Path | None = None) -> LoadedRun:
    """Load one complete run without modifying any source file."""
    path = _within(results_root, path) if results_root is not None else path.resolve()
    required = [path / "config.json", path / "predictions.jsonl", path / "metrics.csv"]
    missing = [item.name for item in required if not item.is_file()]
    if missing:
        raise ExplorerDataError(f"Incomplete run {path}: missing {', '.join(missing)}")
    if results_root is not None:
        required = [_within(results_root, item) for item in required]
    config = _read_json(required[0])
    identity = _identity(config, required[0])
    expected_suffix = (identity.benchmark, identity.method, identity.run_id)
    if tuple(path.parts[-3:]) != expected_suffix:
        raise ExplorerDataError(f"Run path disagrees with config: {path}")
    return LoadedRun(path, identity, config, _load_predictions(required[1]), _load_metrics(required[2]))


def discover_runs(results_root: Path) -> list[LoadedRun]:
    """Discover complete, valid runs in deterministic order."""
    root = results_root.resolve()
    runs: list[LoadedRun] = []
    if not root.is_dir():
        return runs
    for config_path in sorted(root.glob("*/*/*/config.json")):
        if config_path.parts[-4] == "tuning":
            continue
        run_dir = config_path.parent
        if (run_dir / "predictions.jsonl").is_file() and (run_dir / "metrics.csv").is_file():
            runs.append(load_run(run_dir, results_root=root))
    return runs


def validate_generation_comparable(runs: Iterable[LoadedRun]) -> list[str]:
    """Reject unsafe joins and return non-fatal seed warnings."""
    selected = list(runs)
    if len(selected) < 2:
        raise ExplorerDataError("At least two runs are required for a generation comparison")
    first = selected[0]
    keys = ("model_name", "model_revision", "benchmark", "dataset", "dataset_revision", "split", "source_commit", "decoding")
    for run in selected[1:]:
        mismatches = [key for key in keys if getattr(first.identity, key) != getattr(run.identity, key)]
        if mismatches:
            raise ExplorerDataError(f"Runs are not generation-comparable: {', '.join(mismatches)}")
    expected_ids = {row["example_id"] for row in first.predictions}
    for run in selected[1:]:
        if {row["example_id"] for row in run.predictions} != expected_ids:
            raise ExplorerDataError("Runs have different example-ID sets")
    warnings = []
    if len({run.identity.seed for run in selected}) > 1:
        warnings.append("Seeds differ, so this is not a paired stochastic-generation comparison.")
    return warnings


def join_generations(runs: Iterable[LoadedRun]) -> dict[str, Any]:
    selected = list(runs)
    warnings = validate_generation_comparable(selected)
    outputs = {run.identity.method: {row["example_id"]: row["output"] for row in run.predictions} for run in selected}
    if len(outputs) != len(selected):
        raise ExplorerDataError("Select at most one run per method")
    ids = [row["example_id"] for row in selected[0].predictions]
    return {
        "warnings": warnings,
        "prompt_status": "unavailable",
        "examples": [{"example_id": example_id, "outputs": {method: values[example_id] for method, values in outputs.items()}} for example_id in ids],
    }


def load_tuning_summaries(results_root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((results_root / "tuning").glob("*/summary.json")):
        path = _within(results_root, path)
        summary = _read_json(path)
        summaries.append({"run_id": path.parent.name, "summary": summary})
    return summaries

