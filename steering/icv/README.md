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
the authors' SVD sign convention, and the first component plus the mean difference.
The embedding segment is discarded after PCA. Steering uses normalized addition
and cosine weighting at each MLP output, restoring its original L2 norm. Use
the shared [evaluation runner](../../evaluation/README.md) after setting the
ICV strength on development examples in `configs/main.json`.

The construction data and output are separate from held-out benchmark data.
Joint PCA replaces the proposal's per-layer variant with user approval.
Gemma 3 uses temporary output hooks instead of the authors' module wrappers. The
output dtype is preserved rather than forced to float16. Only the paired,
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

Primary strength tuning uses the intervention constructed from all shared
construction examples through the common tuner.

```sh
uv run python tuning.py --config configs/tuning.json --run-id <unique-run-id> --method icv
```

The fixed ICV candidates are `0`, `0.1`, `0.25`, and `0.5`. The tuner uses all
65 development prompts, one shared unsteered baseline, and the pinned Qwen3-4B
judge described in the [shared steering documentation](../README.md). The other
BBQ variants remain exploratory artifacts and are not candidates in primary
strength selection. Historical `results/icv_tuning` files are retained only as
pilot evidence.
