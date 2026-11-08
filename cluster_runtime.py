"""Check the shared SoC cluster runtime before loading a model; see docs/Cluster_Runtime.md."""

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
GPU_NAME = "NVIDIA H100 NVL MIG 3g.47gb"
GPU_SMS = 60
CUDA_VERSION = "12.6"


def on_cluster():
    """Also catch direct launches on SoC nodes outside a Slurm allocation."""
    host = socket.gethostname().split(".")[0]
    return bool(os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
                or re.fullmatch(r"xlogin\d*|xgp[a-z]\d+", host))


def check_cluster_runtime(*, force=False):
    """Return verified runtime evidence on the cluster; local development is unchanged."""
    if not force and not on_cluster():
        return None
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOBID")
    if not job:
        raise RuntimeError("Run inside a Slurm h100-47 compute allocation; see docs/Cluster_Runtime.md.")
    python_pin = (ROOT / ".python-version").read_text(encoding="utf-8").strip()
    if platform.python_implementation() != "CPython" or platform.python_version() != python_pin:
        raise RuntimeError(f"Cluster runs require CPython {python_pin}; run uv sync --locked first.")
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Cluster runs require the uv project environment; use uv run --locked python.")

    # Let uv validate the complete platform-specific dependency graph and stale locks.
    # Check this interpreter's environment, even when VIRTUAL_ENV names another one.
    try:
        checked = subprocess.run(
            ["uv", "sync", "--locked", "--check", "--active", "--offline", "--python", sys.executable],
            cwd=ROOT, env={**os.environ, "VIRTUAL_ENV": sys.prefix},
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError as error:
        raise RuntimeError("uv must be on PATH to verify uv.lock before cluster runs.") from error
    if checked.returncode:
        raise RuntimeError("Cluster packages do not match uv.lock. Run uv sync --locked on a compute node.\n"
                           + checked.stdout + checked.stderr)
    if torch.version.cuda != CUDA_VERSION:
        raise RuntimeError(f"Cluster runs require the PyTorch CUDA {CUDA_VERSION} build, got {torch.version.cuda}.")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Cluster runs require exactly one visible h100-47 MIG device.")
    device = torch.cuda.get_device_properties(0)
    if device.name != GPU_NAME or device.multi_processor_count != GPU_SMS:
        raise RuntimeError(
            f"Expected {GPU_NAME}, {GPU_SMS} SMs; got {device.name}, {device.multi_processor_count} SMs. "
            "Release this allocation and request h100-47 again; see docs/Cluster_Runtime.md."
        )
    driver = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    evidence = {
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": device.name, "gpu_multiprocessors": device.multi_processor_count,
            "gpu_total_memory": device.total_memory,
            "uv_lock_sha256": hashlib.sha256((ROOT / "uv.lock").read_bytes()).hexdigest(),
            "packages": dict(sorted((dist.metadata["Name"], dist.version)
                                    for dist in importlib.metadata.distributions())),
        },
        "allocation": {"host": socket.gethostname(), "slurm_job_id": job, "driver": driver},
    }
    print("Verified cluster runtime: " + json.dumps(evidence), flush=True)
    return evidence


def save_cluster_runtime(model, directory, *, resuming=False):
    """Retain allocation evidence and reject mixing runtimes in a partial generation run."""
    evidence = getattr(model, "_cs3268_cluster_runtime", None)
    path = Path(directory) / "cluster_runtime.json"
    if evidence is None:
        if path.exists():
            raise ValueError("Resume this cluster run in a verified cluster runtime.")
        return
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["runtime"] != evidence["runtime"]:
            raise ValueError("Cluster runtime changed; use a new run ID for fresh generations.")
    else:
        if resuming:
            raise ValueError("Partial generations have no cluster runtime evidence; use a new run ID.")
        saved = {"runtime": evidence["runtime"], "allocations": []}
    if evidence["allocation"] not in saved["allocations"]:
        saved["allocations"].append(evidence["allocation"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(saved, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


if __name__ == "__main__":
    check_cluster_runtime(force=True)
