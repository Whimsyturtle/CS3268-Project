# Evaluation

[api.pyi](api.pyi) defines shared records, the benchmark interface, and the runner.
Benchmark implementations belong in `bbq/`, `holistic_bias/`, `mmlu_pro/`, and
`alpaca_eval/` and must provide `load_examples`, `format_prompt`, and `score`
with the shared signatures.

| Function | Responsibility |
| --- | --- |
| `load_examples(config)` | Load unique IDs and native benchmark fields, including references and scoring metadata. |
| `format_prompt(example, config)` | Produce model input without reference answers or scoring-only metadata. |
| `generate(model, tokenizer, prompts, decoding=..., seed=...)` | Return one raw continuation per prompt, in order, excluding input text. |
| `score(examples, predictions, config)` | Join by example ID and return benchmark metric rows. |
| `run(config_path, method=..., benchmark=..., run_id=...)` | Run inference and scoring and return the saved run directory. |

The runner owns model loading, tokenization, formatting settings, decoding, and
seeding. Each steering implementation applies its intervention through the
[shared steering interface](../steering/README.md).

`load_model` requires a full model commit hash and passes it to both pretrained
loaders. After successful loading, it records that hash as `model.config.revision`
for steering compatibility checks; Transformers no longer retains `_commit_hash`.

Preserve metadata required for scoring. Scorers must reject missing, duplicate,
or unexpected prediction IDs instead of silently scoring a subset.
Save [raw predictions](../results/README.md) before scoring so metrics can be
recomputed without rerunning generation.

Use the [proposal](../CS3268_Project_Proposal_Group_6.md) as the source for required
metrics. For multiple-choice benchmarks (BBQ, MMLU-Pro), the model answers with a
letter, which is parsed from its text output; each scorer's docstring explains how
invalid outputs are counted. HolisticBias still needs its external classifier,
so its runs save predictions without metrics. AlpacaEval 2.0 with a custom judge
reads its API key from the environment and saves the judge annotations and
leaderboard with the run.
Its prompt and weighted parser follow the
[upstream configuration](https://github.com/tatsu-lab/alpaca_eval/tree/main/src/alpaca_eval/evaluators_configs/weighted_alpaca_eval_gpt4_turbo),
and it compares against the pinned GPT-4 baseline responses. It reports weighted
win rate. The judge that we are using differs from the official judge, so results are
not comparable with the official AlpacaEval 2.0 leaderboard.

Run a benchmark with
`uv run --locked python -m evaluation.runner --config configs/main.json --method baseline --benchmark bbq --run-id <run_id>`.
For a quick test, copy the config and set a benchmark's `limit`.

All new cluster runs use the [shared runtime requirements](../docs/Cluster_Runtime.md).

Use identical held-out prompts and decoding settings across methods for each
benchmark. Tune settings on development data, then freeze them before final
testing. See [experiment configurations](../configs/README.md).
