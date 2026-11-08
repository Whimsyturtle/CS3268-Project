"""Local read-only web application for completed intervention experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from explorer.analysis import load_capture_bundles, load_control_file
from explorer.results import ExplorerDataError, discover_runs, join_generations, load_tuning_summaries

ROOT = Path(__file__).resolve().parents[1]
STATIC_ROOT = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class ExplorerSettings:
    results_root: Path
    capture_root: Path | None = None
    capture_bundles: tuple[Path, ...] = ()
    control_files: tuple[Path, ...] = ()
    experiment_groups: tuple[dict[str, Any], ...] = ()
    host: str = "127.0.0.1"
    port: int = 8000


class GenerationRequest(BaseModel):
    run_keys: list[str]


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _path_list(config: dict[str, Any], key: str, base: Path) -> tuple[Path, ...]:
    values = config.get(key, [])
    if not isinstance(values, list) or not all(isinstance(value, str) and value for value in values):
        raise ExplorerDataError(f"{key} must be a list of non-empty paths")
    return tuple(_resolve(base, value) for value in values)


def load_settings(path: Path | None = None) -> ExplorerSettings:
    if path is None:
        return ExplorerSettings(results_root=ROOT / "results")
    config_path = path.resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExplorerDataError(f"Cannot read explorer configuration {path}: {error}") from error
    if not isinstance(config, dict):
        raise ExplorerDataError("Explorer configuration must be a JSON object")
    base = config_path.parent
    results_value = config.get("results_root", "results")
    if not isinstance(results_value, str) or not results_value:
        raise ExplorerDataError("results_root must be a non-empty path")
    results_root = _resolve(base, results_value)
    capture_value = config.get("capture_root")
    if capture_value is not None and (not isinstance(capture_value, str) or not capture_value):
        raise ExplorerDataError("capture_root must be a non-empty path")
    capture_root = _resolve(base, capture_value) if capture_value else None
    bundle_values = config.get("capture_bundles", [])
    if bundle_values and capture_root is None:
        raise ExplorerDataError("capture_root is required when capture_bundles are configured")
    capture_bundles = _path_list(config, "capture_bundles", capture_root or base)
    control_files = _path_list(config, "control_files", base)
    experiment_groups = config.get("experiment_groups", [])
    if not isinstance(experiment_groups, list) or not all(isinstance(group, dict) for group in experiment_groups):
        raise ExplorerDataError("experiment_groups must be a list of objects")
    names = [group.get("name") for group in experiment_groups]
    if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
        raise ExplorerDataError("Experiment-group names must be non-empty and unique")
    host = config.get("host", "127.0.0.1")
    port = config.get("port", 8000)
    if not isinstance(host, str) or host not in {"127.0.0.1", "localhost"}:
        raise ExplorerDataError("The explorer may bind only to localhost")
    if not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535:
        raise ExplorerDataError("Explorer port must be an integer from 1 to 65535")
    return ExplorerSettings(results_root, capture_root, capture_bundles, control_files, tuple(experiment_groups), host, port)


def _run_key(run: Any) -> str:
    identity = run.identity
    return f"{identity.benchmark}/{identity.method}/{identity.run_id}"


def _run_record(run: Any) -> dict[str, Any]:
    identity = asdict(run.identity)
    return {"key": _run_key(run), **identity, "metric_count": len(run.metrics), "prediction_count": len(run.predictions)}


def formatter_source_commit(root: Path, benchmark: str) -> str | None:
    relevant = ["evaluation/common.py", f"evaluation/{benchmark}"]
    status = subprocess.run(["git", "status", "--porcelain", "--", *relevant], cwd=root, capture_output=True, text=True, check=True)
    if status.stdout.strip():
        return None
    process = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True)
    return process.stdout.strip()


def _selected_metric(run: Any, selector: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(selector, dict) or not all(isinstance(selector.get(key), str) and selector[key] for key in ("metric", "subgroup")):
        raise ExplorerDataError("Experiment-group metric selector is incomplete")
    matches = [row for row in run.metrics if row["metric"] == selector["metric"] and row["subgroup"] == selector["subgroup"]]
    if len(matches) != 1 or matches[0]["value"] is None:
        raise ExplorerDataError(f"Metric {selector['subgroup']}/{selector['metric']} is not uniquely available in {_run_key(run)}")
    return matches[0]


def _experiment_group(group: dict[str, Any], by_key: dict[str, Any]) -> dict[str, Any]:
    split_role = group.get("split_role")
    points = group.get("points")
    if split_role not in {"development", "test"} or not isinstance(points, list) or not points:
        raise ExplorerDataError(f"Experiment group {group['name']} needs a split role and points")
    records = []
    group_revision = None
    group_source = None
    group_policy = None
    group_bias_decoding = None
    group_capability_decoding = None
    bias_axis = None
    capability_axis = None
    for point in points:
        if not isinstance(point, dict) or not all(isinstance(point.get(key), dict) for key in ("bias", "capability")):
            raise ExplorerDataError(f"Invalid point in experiment group {group['name']}")
        try:
            bias_run = by_key[point["bias"]["run"]]
            capability_run = by_key[point["capability"]["run"]]
        except (KeyError, TypeError) as error:
            raise ExplorerDataError(f"Experiment group {group['name']} references an unknown run") from error
        bias_identity = bias_run.identity
        capability_identity = capability_run.identity
        if (bias_identity.model_name, bias_identity.model_revision, bias_identity.method, bias_identity.strength) != (capability_identity.model_name, capability_identity.model_revision, capability_identity.method, capability_identity.strength):
            raise ExplorerDataError(f"Experiment group {group['name']} pairs incompatible runs")
        if bias_identity.source_commit != capability_identity.source_commit:
            raise ExplorerDataError(f"Experiment group {group['name']} pairs different source revisions")
        bias_decoding = {key: value for key, value in bias_identity.decoding.items() if key != "max_new_tokens"}
        capability_decoding = {key: value for key, value in capability_identity.decoding.items() if key != "max_new_tokens"}
        if bias_decoding != capability_decoding:
            raise ExplorerDataError(f"Experiment group {group['name']} has incompatible decoding policies")
        current_revision = (bias_identity.model_name, bias_identity.model_revision)
        current_axis_bias = (bias_identity.benchmark, bias_identity.dataset, bias_identity.dataset_revision, bias_identity.split, point["bias"].get("metric"), point["bias"].get("subgroup"))
        current_axis_capability = (capability_identity.benchmark, capability_identity.dataset, capability_identity.dataset_revision, capability_identity.split, point["capability"].get("metric"), point["capability"].get("subgroup"))
        if group_revision is None:
            group_revision = current_revision
            group_source = bias_identity.source_commit
            group_policy = bias_decoding
            group_bias_decoding = bias_identity.decoding
            group_capability_decoding = capability_identity.decoding
            bias_axis = current_axis_bias
            capability_axis = current_axis_capability
        elif current_revision != group_revision or bias_identity.source_commit != group_source:
            raise ExplorerDataError(f"Experiment group {group['name']} mixes model or source revisions")
        elif bias_decoding != group_policy:
            raise ExplorerDataError(f"Experiment group {group['name']} mixes decoding policies")
        elif bias_identity.decoding != group_bias_decoding or capability_identity.decoding != group_capability_decoding:
            raise ExplorerDataError(f"Experiment group {group['name']} mixes per-axis generation budgets")
        elif current_axis_bias != bias_axis or current_axis_capability != capability_axis:
            raise ExplorerDataError(f"Experiment group {group['name']} mixes metric axes or dataset identities")
        if point.get("method") != bias_identity.method or point.get("strength") != bias_identity.strength:
            raise ExplorerDataError(f"Experiment group {group['name']} point identity disagrees with its runs")
        bias_metric = _selected_metric(bias_run, point["bias"])
        capability_metric = _selected_metric(capability_run, point["capability"])
        records.append({
            "method": bias_identity.method,
            "strength": bias_identity.strength,
            "seed": bias_identity.seed,
            "bias": {**bias_metric, "benchmark": bias_identity.benchmark, "run_key": _run_key(bias_run)},
            "capability": {**capability_metric, "benchmark": capability_identity.benchmark, "run_key": _run_key(capability_run)},
        })
    return {"name": group["name"], "status": "available", "split_role": split_role, "model_name": group_revision[0], "model_revision": group_revision[1], "points": records}


def _experiment_groups(settings: ExplorerSettings, loaded_runs: list[Any]) -> list[dict[str, Any]]:
    by_key = {_run_key(run): run for run in loaded_runs}
    output = []
    for group in settings.experiment_groups:
        try:
            output.append(_experiment_group(group, by_key))
        except ExplorerDataError as error:
            output.append({"name": group["name"], "status": "unavailable", "split_role": group.get("split_role"), "reason": str(error), "points": []})
    return output


def create_app(settings: ExplorerSettings | None = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="Intervention Explorer", docs_url=None, redoc_url=None)

    @app.exception_handler(ExplorerDataError)
    async def data_error_handler(_request: Request, error: ExplorerDataError) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=422)

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = request.headers.get("host", "").split(":", 1)[0]
        if host not in {"127.0.0.1", "localhost", "testserver"}:
            return JSONResponse({"detail": "Unexpected Host header"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin not in {f"http://127.0.0.1:{settings.port}", f"http://localhost:{settings.port}"}:
            return JSONResponse({"detail": "Cross-origin requests are not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def runs():
        return discover_runs(settings.results_root)

    @app.get("/")
    def index():
        return FileResponse(STATIC_ROOT / "index.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name not in {"app.js", "style.css"}:
            raise HTTPException(status_code=404)
        media_type = "text/javascript" if name.endswith(".js") else "text/css"
        return FileResponse(STATIC_ROOT / name, media_type=media_type)

    @app.get("/api/status")
    def status():
        loaded = runs()
        return {
            "mode": "offline",
            "gpu_required": False,
            "run_count": len(loaded),
            "controls_configured": len(settings.control_files),
            "captures_configured": len(settings.capture_bundles),
        }

    @app.get("/api/runs")
    def list_runs():
        return [_run_record(run) for run in runs()]

    @app.get("/api/metrics")
    def list_metrics():
        rows = []
        for run in runs():
            for metric in run.metrics:
                rows.append({"run_key": _run_key(run), "benchmark": run.identity.benchmark, "method": run.identity.method, "strength": run.identity.strength, **metric})
        return rows

    @app.get("/api/tuning")
    def tuning():
        return load_tuning_summaries(settings.results_root)

    @app.get("/api/groups")
    def groups():
        return _experiment_groups(settings, runs())

    @app.post("/api/generations")
    def generations(request: GenerationRequest):
        by_key = {_run_key(run): run for run in runs()}
        if len(request.run_keys) < 2 or len(set(request.run_keys)) != len(request.run_keys):
            raise HTTPException(status_code=422, detail="Select at least two distinct runs")
        unknown = [key for key in request.run_keys if key not in by_key]
        if unknown:
            raise HTTPException(status_code=404, detail=f"Unknown run: {unknown[0]}")
        selected = [by_key[key] for key in request.run_keys]
        return join_generations(selected, formatter_source_commit=formatter_source_commit(ROOT, selected[0].identity.benchmark))

    @app.get("/api/controls")
    def controls():
        if not settings.control_files:
            return {"status": "unavailable", "reason": "No validated control records are configured.", "records": []}
        records = [load_control_file(path) for path in settings.control_files]
        return {"status": "available", "records": records}

    @app.get("/api/representation")
    def representation():
        if settings.capture_root is None or not settings.capture_bundles:
            return {"status": "unavailable", "reason": "No compatible activation captures are configured.", "bundles": []}
        bundles = load_capture_bundles(settings.capture_bundles, capture_root=settings.capture_root)
        return {"status": "available", "bundles": bundles}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    settings = load_settings(args.config)
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()

