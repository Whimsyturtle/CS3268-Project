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
├── steering_vectors/ # Saved interventions; each folder has config.json
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
[evaluation/api.pyi](evaluation/api.pyi); runtime implementations are not present yet.

See [steering](steering/README.md), [evaluation](evaluation/README.md),
[configurations](configs/README.md), and [results](results/README.md) for details.

`shared_examples/` holds paired records with `id`, `prompt`, `biased_response`,
and `unbiased_response`. Construction examples build interventions; separate
development examples tune settings. Keep related variants in the same split
and final benchmark data separate. Both files are initially empty.

Each `steering_vectors/<method>/config.json` records how its saved intervention
was produced: model revision, layers, example IDs, seed, and method settings
(including SAE checkpoint/features or ReFT-r1 training settings where applicable).
These files are currently empty JSON objects, not runnable configurations.
Saved interventions include any additional parameters required for inference.
Generated outputs are ignored by Git; these configs are tracked. Pretrained SAEs
use the external model cache, not a repository folder.

## Contributing

- Use a separate branch for each task and submit changes to `main` through a pull request (PR). Do not push directly to `main`.
- Keep each PR focused on one goal. Use the [PR template](.github/pull_request_template.md) to describe the goal, changes, and checks performed, including any checks not run.
- Approval is optional; authors may merge their own PRs. Teammate review is encouraged for changes to data preparation, steering construction or training, experiment settings, or evaluation.
