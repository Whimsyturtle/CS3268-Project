# CS3268 Project — Group 6

Comparing CAA, ICV, SAE, and ReFT-r1 on Gemma-3-1b to reduce social bias while preserving general capability (Track 2: Evaluation).

- [Proposal](CS3268_Project_Proposal_Group_6.md): methods, benchmarks, and experiment design.
- [Course requirements](docs/CS3268_Project_Document.md) and [cluster guide](docs/SoC_Cluster_Guide.md).
- [Glossary](GLOSSARY.md) and [agent instructions](AGENTS.md).

## Directory structure

```text
ICV/
├── steering/       # Shared API and caa/, icv/, sae/, reft_r1/
├── evaluation/     # Shared API and four benchmark folders
├── configs/        # Experiment configurations
├── data/           # Construction examples and custom split definitions
├── checkpoints/    # Saved interventions
├── results/        # <benchmark>/<method>/<run_id>/
├── tests/          # Focused implementation tests
└── docs/           # Course and cluster documents
```

The `api.pyi` files define interfaces; runtime implementations are not present yet.

See [steering](steering/README.md), [evaluation](evaluation/README.md),
[configurations](configs/README.md), [data](data/README.md),
[checkpoints](checkpoints/README.md), and [results](results/README.md) for details.
Downloaded data, checkpoints, and generated results are ignored by Git.

## Contributing

- Use a separate branch for each task and submit changes to `main` through a pull request (PR). Do not push directly to `main`.
- Keep each PR focused on one goal. Use the [PR template](.github/pull_request_template.md) to describe the goal, changes, and checks performed, including any checks not run.
- Approval is optional; authors may merge their own PRs. Teammate review is encouraged for changes to data preparation, steering construction or training, experiment settings, or evaluation.
