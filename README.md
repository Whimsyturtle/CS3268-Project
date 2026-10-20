# CS3268 Project — Group 6

Comparing CAA, ICV, SAE, and ReFT-r1 activation-steering methods on Gemma-3-1b to reduce social bias while preserving general model capability (Track 2: Evaluation).

- [Proposal](CS3268_Project_Proposal_Group_6.md): steering methods, benchmarks, experiment design.
- [Course requirements](docs/CS3268_Project_Document.md): project tracks, requirements, deliverables.
- [Cluster guide](docs/SoC_Cluster_Guide.md): setup and use of the compute cluster; examples use a separate guide repository.
- [Glossary](GLOSSARY.md): preferred terms and simple definitions.
- [AGENTS.md](AGENTS.md): instructions for agents.

## Evaluation

[evaluation.py](evaluation.py) holds the code shared by all steering methods. It uses [uv](https://docs.astral.sh/uv/); run `uv sync` once to install dependencies. Gemma 3 is gated on Hugging Face, so accept its license and run `uv run hf auth login` first.

- `bbq_build_set()` returns the BBQ build set. Construct steering vectors only from these rows.
- `run_benchmarks(model, tokenizer, output_dir, method_name)` scores BBQ and MMLU-Pro with [lm-eval](https://github.com/EleutherAI/lm-evaluation-harness) and saves HolisticBias and AlpacaEval 2.0 responses. Attach your method's hooks to the model before calling it. Scoring of the saved responses is not implemented yet.
- Results change slightly with batch size and GPU type, so load the model with `load_model()`, keep the default `batch_size`, and run every method on the same GPU type.
- `uv run python evaluation.py --output-dir results/baseline` evaluates the unsteered baseline. Add `--limit 2` for a quick test run.
- `uv run pytest` runs the tests.

## Contributing

- Use a separate branch for each task and submit changes to `main` through a pull request (PR). Do not push directly to `main`.
- Keep each PR focused on one goal. Use the [PR template](.github/pull_request_template.md) to describe the goal, changes, and checks performed, including any checks not run.
- Approval is optional; authors may merge their own PRs. Teammate review is encouraged for changes to data preparation, steering construction or training, experiment settings, or evaluation.
