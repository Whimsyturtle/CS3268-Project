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

Preserve metadata required for scoring. Scorers must reject missing, duplicate,
or unexpected prediction IDs instead of silently scoring a subset.
Save [raw predictions](../results/README.md) before scoring so metrics can be
recomputed without rerunning generation.

Use the [proposal](../CS3268_Project_Proposal_Group_6.md) as the source for required
metrics. Metric formulas, the MCQ answer protocol, and external scorer/judge
settings still need to be specified before implementing scorers. The current
`generate` contract returns text. Choice-likelihood MCQ scoring would require
an explicit extension to the shared inference contract.

Use identical held-out prompts and decoding settings across methods for each
benchmark. Tune settings on development data, then freeze them before final
testing. See [experiment configurations](../configs/README.md).
