"""Repeat one saved tuning baseline batch; do not judge or select a strength."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
from pathlib import Path

import torch

from evaluation.runner import generate, load_model, source_commit
from tuning import read_development, write_json_atomic


def environment(model):
    device = model.device
    properties = torch.cuda.get_device_properties(device)
    return {
        "hostname": socket.gethostname(),
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name)
                     for name in ("torch", "transformers", "tokenizers", "huggingface-hub")},
        "cuda_build": torch.version.cuda,
        "device": str(device),
        "gpu": properties.name,
        "gpu_total_memory": properties.total_memory,
        "gpu_multiprocessors": properties.multi_processor_count,
        "nvidia_smi": subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True,
        ).stdout,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "environment_variables": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
            "OMP_NUM_THREADS", "SLURM_JOB_ID",
        )},
    }


def check(run_dir: Path, reference_dirs: list[Path], output: Path):
    if output.exists():
        raise FileExistsError(f"Use a new report path: {output}")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    development = read_development(Path(config["examples"]))
    if [row["id"] for row in development] != config["development"]["example_ids"]:
        raise ValueError("Development order differs from the saved run")
    rows = development[:config["generation"]["batch_size"]]
    prompts = [row["prompt"] for row in rows]
    references = {}
    for directory in [run_dir, *reference_dirs]:
        saved = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if any(saved[key] != config[key] for key in ("model", "generation", "seed", "development")):
            raise ValueError(f"Reference generation settings differ: {directory}")
        records = [json.loads(line) for line in
                   (directory / "generations.jsonl").read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        baseline = [row for row in records if row["method"] == "baseline" and row["strength"] == 0]
        by_id = {row["example_id"]: row for row in baseline}
        if len(by_id) != len(baseline) or any(
            row["id"] not in by_id or by_id[row["id"]]["prompt"] != row["prompt"] for row in rows
        ):
            raise ValueError(f"Reference baseline prompts differ or are missing: {directory}")
        references[str(directory)] = [by_id[row["id"]]["output"] for row in rows]
    if not torch.cuda.is_available():
        raise RuntimeError("Run this diagnostic in the allocated NVIDIA GPU shell")

    model, tokenizer = load_model(config["model"])
    tokenizer.padding_side = "left"
    inputs = tokenizer.apply_chat_template(
        [[{"role": "user", "content": prompt}] for prompt in prompts],
        add_generation_prompt=True, padding=True, return_tensors="pt", return_dict=True,
    )
    report = {
        "purpose": __doc__, "source": source_commit(), "run": str(run_dir),
        "model": config["model"], "generation": config["generation"], "seed": config["seed"],
        "environment": environment(model),
        "model_class": type(model).__name__, "model_dtype": str(model.dtype),
        "attention_implementation": model.config._attn_implementation,
        "model_config": model.config.to_dict(),
        "model_generation_config": model.generation_config.to_dict(),
        "tokenizer_class": type(tokenizer).__name__,
        "chat_template": tokenizer.get_chat_template(),
        "examples": [{"id": row["id"], "prompt": row["prompt"]} for row in rows],
        "inputs": {key: value.tolist() for key, value in inputs.items()},
        "reference_outputs": references, "outputs": [],
    }
    for repeat in range(2):
        print(f"Baseline repeat {repeat + 1}/2: {len(rows)} prompts", flush=True)
        report["outputs"].append(generate(
            model, tokenizer, prompts, decoding=config["generation"], seed=config["seed"],
        ))
    report["checks"] = {
        "same_process_equal": report["outputs"][0] == report["outputs"][1],
        "matches_saved": {name: [sum(a == b for a, b in zip(outputs, expected, strict=True))
                                 for outputs in report["outputs"]]
                          for name, expected in references.items()},
        "output_sha256": [hashlib.sha256(json.dumps(outputs).encode()).hexdigest()
                          for outputs in report["outputs"]],
    }
    write_json_atomic(output, report)
    print(json.dumps(report["checks"], indent=2))
    print(f"Saved {output}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--reference-run-dir", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    check(args.run_dir, args.reference_run_dir, args.output)
