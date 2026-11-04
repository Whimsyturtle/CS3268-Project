# Experiment configurations

Store reviewed experiment configurations here. Each must record:

- Exact model/tokenizer revisions and precision.
- Dataset revisions, splits, and any custom split definitions.
- Prompt formatting, decoding settings, and seed.
- Saved intervention directory, strength, and other method settings.
- Scorer/judge settings and whether the run uses development or test data.

## Strength tuning

[`tuning.json`](tuning.json) configures the common pooled strength tuner. Each
method uses all 65 development records and one `pooled_all` intervention;
candidate strengths remain method-specific. Strength zero is a shared unsteered
baseline rather than a separate artifact for each method.

The generator, judge, and intervention metadata must contain immutable source
revisions. The configured `Qwen/Qwen3-4B` judge runs on the compute cluster in
BF16 with thinking disabled. Local tests use test doubles and do not load either
model. Add another method only after its pooled artifact exists. AlpacaEval's
annotator is a separate evaluation decision and is not configured here.

[main.json](main.json) holds the test-run settings. Prompt formatting is in each benchmark's code
and is identified by the source commit saved with every run. Method strengths stay
`null` until tuned on development data.
