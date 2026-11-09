"""Run Meta's released ParlAI current-utterance style classifier over JSONL inputs."""

import argparse
import hashlib
import json
import platform
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path):
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--datapath", type=Path, required=True)
    parser.add_argument("--expected-parlai-version", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--input-sha256", required=True)
    args = parser.parse_args()

    import parlai
    import torch
    from parlai.core.agents import create_agent
    from parlai.core.message import Message
    from parlai.core.opt import Opt

    parlai_version = getattr(parlai, "__version__", "unknown")
    if parlai_version != args.expected_parlai_version:
        raise RuntimeError(f"Expected ParlAI {args.expected_parlai_version}, found {parlai_version}.")
    actual_input_sha256 = sha256(args.input)
    if actual_input_sha256 != args.input_sha256:
        raise RuntimeError("Classifier input SHA-256 does not match the caller's value.")

    args.datapath.mkdir(parents=True, exist_ok=True)
    # The released model.opt names Meta's unreleased parlai_internal agent; ParlAI's own
    # style_gen tests load this checkpoint with the public ClassifierAgent instead.
    overrides = {
        "model": "projects.style_gen.classifier:ClassifierAgent",
        "classes_from_file": "image_chat_personalities_file",
        "print_scores": True,
    }
    agent = create_agent(
        Opt({"model_file": args.model_file, "datapath": str(args.datapath), "override": overrides}),
        requireModelExists=True,
    )
    styles = list(agent.class_list)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input, encoding="utf-8") as source, open(
        args.output, "w", encoding="utf-8"
    ) as destination:
        while batch := [json.loads(line) for _, line in zip(range(args.batch_size), source)]:
            observations = []
            clones = []
            for record in batch:
                clone = agent.clone()
                clones.append(clone)
                observations.append(clone.observe(Message({"text": record["text"], "episode_done": True})))
            actions = agent.batch_act(observations)
            for clone, record, action in zip(clones, batch, actions):
                if hasattr(clone, "self_observe"):
                    clone.self_observe(action)
                by_class = action["probs_by_class"]
                probabilities = [float(by_class[style]) for style in styles]
                destination.write(
                    json.dumps({"example_id": record["example_id"], "probabilities": probabilities}) + "\n"
                )

    model_path = Path(agent.opt["model_file"])
    class_path = Path(agent.opt["classes_from_file"])
    manifest = {
        "classifier": "ParlAI current-utterance-only style classifier",
        "requested_model_file": args.model_file,
        "parlai_version": parlai_version,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "input_sha256": actual_input_sha256,
        "styles": styles,
        "artifacts": [
            artifact(model_path),
            artifact(Path(str(model_path) + ".opt")),
            artifact(Path(agent.opt["dict_file"])),
            artifact(class_path),
        ],
    }
    args.manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
