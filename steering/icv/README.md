# ICV

Build the paired-demonstration vector from the shared construction split:

```sh
uv run python -m steering.icv.build
```

Run this from the `ICV` directory after `uv sync` and Hugging Face access setup in
the [project README](../../README.md). The command uses the pinned Gemma 3 model
in `configs/main.json`, reads `biased_response` and `unbiased_response`, and
saves `vector.pt` plus construction metadata in `steering_vectors/icv/`.

The implementation follows the authors' [released code at b187c63](https://github.com/shengliu66/ICV/tree/b187c6387ae41097f5e38182d5a8df7d06ca9ec4):
joint centered PCA on last-token differences from embeddings and decoder layers,
the upstream SVD sign convention, and the first component plus the mean difference.
The embedding segment is discarded after PCA. Steering uses normalized addition
and cosine weighting at each MLP output, restoring its original L2 norm. Use
the shared [evaluation runner](../../evaluation/README.md) after setting the
ICV strength on development examples in `configs/main.json`.

The construction data and output are separate from held-out benchmark data.
Joint PCA replaces the proposal's per-layer variant with user approval.
Gemma 3 uses temporary output hooks instead of upstream's module wrappers;
the output dtype is preserved rather than forced to float16. Only the paired,
single-vector case is supported. Short source comments identify paper/code differences.

Rebuild artifacts produced by the previous implementation, then retune strength.
`load_intervention` rejects the old construction method even when its shape matches.

For the BBQ pooling pilot, extract every construction activation once and build
the agreed pooled, mapped-category, intersection-component, and geographic-transfer
variants with:

```sh
uv run python -m steering.icv.build --bbq-variants
```

Artifacts are written below `steering_vectors/icv/bbq_variants/`. The component
variants use the union of their source examples; they are transfer diagnostics,
not demonstrations of intersectional bias.

Tune the three primary BBQ strategies on the separate development split:

```sh
uv run python -m steering.icv.tune
```

The fixed pilot strengths are `0`, `0.1`, `0.25`, and `0.5`. The tuner writes
response-only likelihood margins to `results/icv_tuning/scores.csv` and shuffles
deterministic generations into `generations.jsonl`. All three strategies use the
same 45 mapped development prompts, and the unsteered generation is produced only
once per prompt. Review that file without opening `generation_key.jsonl`, which
contains the hidden strategy and strength. Rate
stereotype avoidance, relevance, coherence, and refusal/evasion; then select the
smallest strength on the quality/bias Pareto frontier. Teacher-forced likelihood
is a diagnostic, not the sole selection criterion. `config.json` records run
provenance and exact intervention metadata. BBQ remains held out.
