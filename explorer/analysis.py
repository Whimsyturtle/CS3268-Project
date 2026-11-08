"""CPU-only summaries for controls and captured representations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA

from explorer.results import ExplorerDataError

CONTROL_FAMILIES = {
    "caa": "norm-matched random directions",
    "icv": "norm-matched random directions",
    "sae": "matched random features",
    "reft_r1": "rank-matched random or untrained interventions",
}


def _contained_file(root: Path, path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ExplorerDataError(f"Capture file is outside configured root: {path}")
    return resolved


def control_summary(observed: float, controls: Sequence[float], *, family: str, minimum: int = 5) -> dict[str, Any]:
    if not family:
        raise ExplorerDataError("Control family is required")
    values = [float(value) for value in controls]
    if not math.isfinite(observed) or not all(math.isfinite(value) for value in values):
        raise ExplorerDataError("Control effects must be finite")
    if len(values) < minimum:
        return {"status": "insufficient", "family": family, "observed": observed, "controls": values, "count": len(values)}
    lower = sum(value < observed for value in values)
    equal = sum(value == observed for value in values)
    percentile = 100 * (lower + 0.5 * equal) / len(values)
    return {"status": "available", "family": family, "observed": observed, "controls": values, "count": len(values), "percentile": percentile}


def load_control_file(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExplorerDataError(f"Cannot read controls from {path}: {error}") from error
    required = {"family", "matching", "metric", "observed", "controls"}
    if not isinstance(record, dict) or not required.issubset(record):
        raise ExplorerDataError(f"Invalid control record in {path}")
    matching = record["matching"]
    metric = record["metric"]
    observed = record["observed"]
    controls = record["controls"]
    if not isinstance(matching, dict) or not matching:
        raise ExplorerDataError("Control matching criteria must be a non-empty object")
    if not isinstance(metric, dict) or not all(isinstance(metric.get(key), str) and metric[key] for key in ("benchmark", "subgroup", "name")):
        raise ExplorerDataError("Control metric identity is incomplete")
    if not isinstance(observed, dict) or not all(key in observed for key in ("run_id", "method", "effect")):
        raise ExplorerDataError("Observed control provenance is incomplete")
    method = observed["method"]
    if method not in CONTROL_FAMILIES or record["family"] != CONTROL_FAMILIES[method]:
        raise ExplorerDataError("Control family is incompatible with the observed method")
    if not isinstance(controls, list) or any(not isinstance(item, dict) or not all(key in item for key in ("run_id", "effect")) for item in controls):
        raise ExplorerDataError("Control run provenance is incomplete")
    run_ids = [item["run_id"] for item in controls]
    if any(not isinstance(run_id, str) or not run_id for run_id in run_ids) or len(set(run_ids)) != len(run_ids):
        raise ExplorerDataError("Control run IDs must be non-empty and unique")
    if not isinstance(observed["run_id"], str) or not observed["run_id"] or observed["run_id"] in run_ids:
        raise ExplorerDataError("Observed run ID must be distinct from control run IDs")
    summary = control_summary(observed["effect"], [item["effect"] for item in controls], family=record["family"])
    return {**record, "summary": summary}


def load_capture_bundle(path: Path, *, capture_root: Path) -> dict[str, Any]:
    root = capture_root.resolve()
    bundle = path.resolve()
    if not bundle.is_relative_to(root):
        raise ExplorerDataError(f"Capture path is outside configured root: {path}")
    config_path = _contained_file(root, bundle / "capture_config.json")
    index_path = _contained_file(root, bundle / "capture_index.jsonl")
    activations_path = _contained_file(root, bundle / "activations.npy")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        lines = index_path.read_text(encoding="utf-8").splitlines()
        index = [json.loads(line) for line in lines]
        activations = np.load(activations_path, mmap_mode="r")
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise ExplorerDataError(f"Cannot read capture bundle {bundle}: {error}") from error
    if not isinstance(config, dict) or not all(isinstance(row, dict) for row in index):
        raise ExplorerDataError(f"Invalid capture metadata in {bundle}")
    model = config.get("model")
    if not isinstance(model, dict) or not all(isinstance(model.get(key), str) and model[key] for key in ("name", "revision")):
        raise ExplorerDataError("Capture model identity is incomplete")
    if config.get("split_role") != "development" or not isinstance(config.get("source_split"), str) or not config["source_split"]:
        raise ExplorerDataError("Shared PCA requires an identified development capture")
    baseline_run_id = config.get("baseline_run_id")
    if not isinstance(baseline_run_id, str) or not baseline_run_id:
        raise ExplorerDataError("Capture baseline_run_id is required")
    if activations.ndim != 2 or len(index) != activations.shape[0]:
        raise ExplorerDataError("Capture index and activation shape disagree")
    if activations.shape[0] < 2 or activations.shape[1] < 2 or not np.isfinite(activations).all():
        raise ExplorerDataError("Captures must be finite with at least two rows and dimensions")
    required = {"run_id", "example_id", "method", "strength", "state", "category", "layer", "hook", "token_selection", "source_split"}
    if any(not required.issubset(row) for row in index):
        raise ExplorerDataError("Capture index is missing required fields")
    compatibility = ("layer", "hook", "token_selection")
    for key in compatibility:
        if len({json.dumps(row[key], sort_keys=True) for row in index}) != 1:
            raise ExplorerDataError(f"Incompatible capture {key} values")
    if any(row["source_split"] != config["source_split"] for row in index):
        raise ExplorerDataError("Capture rows disagree with the declared source split")
    baseline_rows = [position for position, row in enumerate(index) if row["state"] == "baseline"]
    if len(baseline_rows) < 2:
        raise ExplorerDataError("At least two baseline rows are required for shared PCA")
    if any(index[position]["run_id"] != baseline_run_id for position in baseline_rows):
        raise ExplorerDataError("Baseline rows must use the declared development baseline run")
    projector = PCA(n_components=2).fit(np.asarray(activations[baseline_rows]))
    coordinates = projector.transform(activations)
    records = [{**row, "x": float(coordinates[position, 0]), "y": float(coordinates[position, 1])} for position, row in enumerate(index)]
    return {
        "config": config,
        "records": records,
        "explained_variance_ratio": projector.explained_variance_ratio_.tolist(),
    }

