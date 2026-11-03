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

`decoding` holds the decoding hyperparameters shared by every benchmark and method, and
`do_sample: false` selects greedy decoding. `batch_size` and `use_cache` are fixed
because changing either can change greedy outputs through small numerical differences;
for the same reason, run every method for a benchmark on the same GPU type. Each
benchmark sets only `max_new_tokens`. Other generation settings come from the model's
`generation_config.json` at the pinned revision (which sets the stop tokens `<eos>` and
`<end_of_turn>`) and the transformers version in `uv.lock`; the model's sampling
settings (`top_k`, `top_p`) have no effect under greedy decoding.
