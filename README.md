# CS3268 Project — Group 6

Comparing CAA, ICV, SAE, and ReFT-r1 activation-steering methods on Gemma-3-1b to reduce social bias while preserving general model capability (Track 2: Evaluation).

- [Proposal](CS3268_Project_Proposal_Group_6.md): steering methods, benchmarks, experiment design.
- [Course requirements](docs/CS3268_Project_Document.md): project tracks, requirements, deliverables.
- [Cluster guide](docs/SoC_Cluster_Guide.md): setup and use of the compute cluster; examples use a separate guide repository.
- [Glossary](GLOSSARY.md): preferred terms and simple definitions.
- [AGENTS.md](AGENTS.md): instructions for agents.

## Directory structure

```text
CS3268-Group-6/
├── shared_examples/
│   ├── construction.jsonl
│   └── development.jsonl
├── steering/       # Steering implementations
│   ├── caa/
│   ├── icv/
│   ├── sae/
│   └── reft_r1/
├── steering_vectors/ # Saved interventions
│   ├── caa/
│   ├── icv/
│   ├── sae/
│   └── reft_r1/
├── evaluation/     # Benchmark implementations
│   ├── bbq/
│   ├── holistic_bias/
│   ├── mmlu_pro/
│   └── alpaca_eval/
├── configs/        # Experiment configurations
├── results/        # <benchmark>/<method>/<run_id>/
├── tests/          # Focused implementation tests
└── docs/           # Course and cluster documents
```

Shared interfaces are defined once in [steering/api.pyi](steering/api.pyi) and
[evaluation/api.pyi](evaluation/api.pyi).

See [shared examples](shared_examples/README.md), [steering](steering/README.md),
[saved interventions](steering_vectors/README.md), [evaluation](evaluation/README.md),
[configurations](configs/README.md), and [results](results/README.md) for details.

## Setup

Install [uv](https://docs.astral.sh/uv/), then run `uv sync --locked`.
[pyproject.toml](pyproject.toml) declares dependency constraints and selects the
PyTorch [CUDA](GLOSSARY.md#cuda) 12.6 (`cu126`) builds on Linux and Windows;
macOS uses PyPI builds. [uv.lock](uv.lock) pins the resolved package versions
and download hashes. Keep it committed and use `--locked` to reject stale locks.
Python is constrained to 3.12 or newer, without an exact interpreter pin.

The CUDA build selection does not install or pin the host NVIDIA driver or system
CUDA toolkit. On a GPU compute node, check the installed build and GPU access:

```bash
uv run --locked python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.version.cuda == '12.6'; assert torch.cuda.is_available()"
```

Gemma 3 is gated on Hugging Face: accept its license, then run
`uv run --locked hf auth login`. Run tests with `uv run --locked pytest`.

## Contributing

- Track major tasks, experiments, and deliverables in [GitHub issues](https://github.com/Whimsyturtle/CS3268-Project/issues). Small fixes can stay within an existing task or PR.
- Before creating an issue, check open and closed issues and related PRs. Give each issue one outcome, dependencies, and completion checks; link the relevant source documents rather than copying their requirements. Issues track progress, while repository documents and saved artifacts remain the sources for design and results.
- Record the agreed owner with an assignee, and keep the issue current when work starts, a blocker appears, or verification changes its status. Use milestones to group delivery stages; keep detailed work in issues.
- Use a separate branch for each task and submit changes to `main` through a pull request (PR). Do not push directly to `main`.
- Keep each PR focused on one goal. Use the [PR template](.github/pull_request_template.md) to describe the goal, changes, and checks performed, including any checks not run.
- Link related issues in PRs. Use `Closes #N` only when merging the PR will satisfy every completion check; use `Refs #N` for partial progress. A merged implementation does not complete an issue that also requires experiments.
- Close completed issues with links to the merged changes or verified deliverables. For experiments, include run IDs, artifact locations, and checks; leave incomplete or invalidated results open. Historical issues may be closed when existing evidence satisfies their stated scope.
- Approval is optional; authors may merge their own PRs. Teammate review is encouraged for changes to data preparation, steering construction or training, experiment settings, or evaluation.
