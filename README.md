# CS3268 Project — Group 6

Comparing CAA, ICV, SAE, and ReFT-r1 activation-steering methods on Gemma-3-1b to reduce social bias while preserving general model capability (Track 2: Evaluation).

- [Proposal](CS3268_Project_Proposal_Group_6.md): steering methods, benchmarks, experiment design.
- [Course requirements](docs/CS3268_Project_Document.md): project tracks, requirements, deliverables.
- [Cluster guide](docs/SoC_Cluster_Guide.md): setup and use of the compute cluster; examples use a separate guide repository.
- [Glossary](GLOSSARY.md): preferred terms and simple definitions.
- [AGENTS.md](AGENTS.md): instructions for agents.

## Directory structure

This repository currently contains documentation and Python interface stubs
(`api.pyi`). The stubs specify the agreed interfaces; they do not execute model
loading, steering, generation, or scoring. Implementations will live in matching
`api.py` files. No model checkpoint, dataset, or result is bundled here.

```text
ICV/
├── configs/                 # Experiment configurations
├── data/                    # Construction examples and custom split definitions
├── methods/
│   ├── api.pyi              # Shared steering interface
│   ├── caa/api.pyi
│   ├── icv/api.pyi
│   ├── sae/api.pyi
│   └── reft_r1/api.pyi
├── evaluation/
│   ├── api.pyi              # Shared records, benchmark interface, and runner
│   ├── bbq/api.pyi
│   ├── holistic_bias/api.pyi
│   ├── mmlu_pro/api.pyi
│   └── alpaca_eval/api.pyi
├── checkpoints/             # Saved steering vectors, features, or parameters
├── results/
│   └── <benchmark>/
│       └── <method>/        # baseline, caa, icv, sae, or reft_r1
│           └── <run_id>/    # Created when an experiment runs
│               ├── config.json
│               ├── predictions.jsonl
│               └── metrics.csv
├── tests/                   # Focused tests added with implementations
└── docs/                    # Existing course and cluster documents
```

The four benchmark directories are `bbq`, `holistic_bias`, `mmlu_pro`, and
`alpaca_eval`. Each has all five method directories under `results/`.
`.gitkeep` files retain empty result directories in Git; they are not results.

## Steering interface

[methods/api.pyi](methods/api.pyi) is the canonical inference contract. Each
method's stub re-exports that contract to avoid copying its signatures. Its
future implementation must supply its own functions with those signatures.

1. `load_artifact(artifact_dir, model=model)` loads the method's saved intervention
   and checks that its base model and intervention locations match the model.
2. `apply(model, artifact, strength=alpha)` returns a context manager that applies
   the intervention and restores the model on exit, including after an exception.

An artifact is the method's saved intervention; see the [glossary](GLOSSARY.md).
Its in-memory representation is method-specific, so the contract uses `Any`
for it. Model and tokenizer objects also use `Any` until the shared loader is
implemented. These annotations do not establish model compatibility.

Construction remains method-specific: averaging for CAA, PCA for ICV, feature
selection for SAE, and training for ReFT-r1. It is outside this evaluation API.
Each method stores its outputs under `checkpoints/<method>/`. Store construction
settings, source example IDs, exact model revision, and intervention locations
alongside the saved intervention. SAE must also identify its compatible SAE
checkpoint and selected features.

The unsteered baseline uses the same evaluation code with
`contextlib.nullcontext()`. Strength is method-specific; equal numbers do not
imply equal intervention effects.

## Evaluation interface

[evaluation/api.pyi](evaluation/api.pyi) defines shared records and signatures.
Each benchmark's stub re-exports `load_examples`, `format_prompt`, and `score`.

| Function | Responsibility |
| --- | --- |
| `load_examples(config)` | Return unique example IDs and native benchmark fields, including references and scoring metadata. |
| `format_prompt(example, config)` | Produce the model input without leaking references or scoring metadata. |
| `generate(model, tokenizer, prompts, decoding=..., seed=...)` | Return one raw response per prompt in the same order. |
| `score(examples, predictions, config)` | Join by example ID and return metric rows for this benchmark. |
| `run(config_path, method=..., benchmark=..., run_id=...)` | Orchestrate inference and scoring; return the saved run directory. |

The runner owns model loading, tokenization, prompt formatting settings,
decoding, and seeding. Method implementations only apply their interventions.
Preserve all benchmark metadata needed for scoring. Scorers must reject missing,
duplicate, or unexpected prediction IDs rather than silently scoring a subset.
Use the [proposal](CS3268_Project_Proposal_Group_6.md) as the source for required
metrics. Metric formulas, the MCQ answer protocol, and external scorer/judge
settings still need to be specified before implementing those scorers.

The current `generate` contract returns text. If choice-likelihood MCQ scoring
is selected, extend the shared inference contract explicitly before using it.

## Data and experiment settings

Keep construction examples under `data/`; store custom split membership in the
records or a small split file. Unchanged published splits can instead be
identified by their dataset revision and split name. Follow the
[proposal's data separation requirement](CS3268_Project_Proposal_Group_6.md#32-dataset-and-metrics).

Experiment configurations belong under `configs/`. They must record the exact
model/tokenizer revisions and precision, dataset revisions and splits, prompt
and decoding settings, seed, intervention checkpoint and strength, and scorer
settings. No executable configuration or default model identifier is supplied
by this skeleton.

Use identical held-out prompts and decoding settings across methods for each
benchmark. Tune method settings on development data, then freeze them before
final testing. Record development versus test use in each run's configuration.

## Saved results

Each run uses `results/<benchmark>/<method>/<run_id>/`. A run ID must be a single
directory name; the runner must reject an existing run directory to avoid
overwriting results.

- `config.json`: resolved experiment settings, source commit, checkpoint identity,
  and dataset/split identity used for this run.
- `predictions.jsonl`: one `Prediction` record per line, containing `example_id`
  and raw `output`. Save predictions before scoring so metrics can be recomputed
  without rerunning generation.
- `metrics.csv`: one `Metric` record per row, with columns `subgroup`, `metric`,
  `value`, and `n`. Use `all` for the overall subgroup; `n` is the number of
  examples contributing to that metric. An undefined value is empty, not zero.

JSON Lines supports appending responses individually and escaping multiline
text. CSV keeps metric summaries easy to compare and plot. External scorers may
save their detailed outputs beside these files; identify them in `config.json`.
An optional generated `results/summary.csv` can add benchmark, method, and run ID
columns when combining runs.

Downloaded data, checkpoints, and generated results are ignored by Git. Commit
small, reviewed configurations and reproducibility metadata deliberately when
needed; do not commit model weights or credentials.

## Contributing

- Use a separate branch for each task and submit changes to `main` through a pull request (PR). Do not push directly to `main`.
- Keep each PR focused on one goal. Use the [PR template](.github/pull_request_template.md) to describe the goal, changes, and checks performed, including any checks not run.
- Approval is optional; authors may merge their own PRs. Teammate review is encouraged for changes to data preparation, steering construction or training, experiment settings, or evaluation.
