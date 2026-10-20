# Experiment configurations

Store reviewed experiment configurations here. Each must record:

- Exact model/tokenizer revisions and precision.
- Dataset revisions, splits, and any custom split definitions.
- Prompt formatting, decoding settings, and seed.
- Saved intervention directory, strength, and other method settings.
- Scorer/judge settings and whether the run uses development or test data.

[main.json](main.json) holds the test-run settings. Prompt formatting is in each benchmark's code
and is identified by the source commit saved with every run. Method strengths stay
`null` until tuned on development data.
