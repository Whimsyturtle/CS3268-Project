# CAA

Implements [Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)
for the project's pinned Gemma 3 model and [shared examples](../../shared_examples/README.md),
following the [proposal](../../CS3268_Project_Proposal_Group_6.md).

## Construction and application

CAA uses all 260 construction pairs to compute the raw mean difference
`v = mean(h(unbiased) - h(biased))`. Each activation is the decoder-block output
at the last assistant response content token; prompt tokens, padding, special
tokens, and closing chat markers are excluded. Positive strength adds
`strength * v` toward the supplied unbiased responses. There is no normalization,
PCA, or base-model training.

[`configs/caa.json`](../../configs/caa.json) holds the construction-only layer
setting, following the separate method-config pattern used by
[SAE](../sae/README.md#feature-selection-and-application). CAA does not load an
SAE checkpoint. SAE derives its layer from the pinned checkpoint path and output
hook; CAA specifies the extraction layer directly. Its zero-based block index 13
is an initial middle-layer choice
for the 26-layer model, fixed before strength tuning; it is not an optimal-layer
claim or a choice based on held-out benchmarks.

The [shared steering interface](../api.pyi) loads and validates a saved vector,
then applies it through a context manager. Nonzero strength changes only the last
nonpadding prompt position during the initial pass and newly processed positions
during cached continuation. Earlier prompt positions are unchanged. This scope
supports the shared greedy runner with caching. Strength zero registers no hooks;
normal and exceptional exits remove the method's hooks.

The artifact contains `vector.pt` (a one-dimensional float32 tensor) and
`config.json`. The pinned model uses flat `model` (name), `revision`, and `dtype`
fields so it works with the existing tuner, like ICV. Metadata also records the
tokenizer, chat-template hash,
layer and extraction/application conventions, construction IDs and data hash,
seed, source state, and vector hash. Loading checks model compatibility and vector
integrity; it never constructs an intervention. Follow the
[saved-intervention policy](../../steering_vectors/README.md) for retained artifacts.

## Build and check

These commands are prepared for manual execution; no cluster run is recorded
here. Complete the [project setup](../../README.md#setup) and
[shared cluster runtime](../../docs/Cluster_Runtime.md) first. The latter is the
source for the pinned Python/lockfile environment, `h100-47` allocation, actual
MIG-device verification, and runtime evidence. Run the following at the repository
root in that same verified compute shell:

```bash
uv run --locked python shared_examples/validate.py --check-evaluation-overlap
uv run --locked python -m steering.caa.build
CHECK_ID="caa-check-$(date +%Y%m%d-%H%M%S)"
uv run --locked python -m steering.caa.check --run-id "$CHECK_ID"
```

The builder refuses to overwrite an intervention. For a separate candidate, use
`--output-dir`, pass the same directory to the checker with `--intervention-dir`,
and update `methods.caa.intervention_dir` in a copied tuning config.

Like the [SAE application check](../sae/README.md#build-and-check), the CAA checker
uses development prompts to test exact zero-strength identity, finite changed
logits at nonzero strength, and restoration after normal and exceptional exits.
It also checks hook cleanup and parameter version/storage indicators. It writes
settings, provenance, raw outputs, and checks to
`results/caa_checks/<run-id>/checks.json`, including failures; new cluster checks
also retain `cluster_runtime.json`. Its default strength 1 tests mechanics, not
bias reduction or a tuned strength. A check with the actual pretrained model
remains pending.

## Tune and pilot

Follow the [shared tuning workflow](../README.md#strength-hyperparameter-tuning),
including its SoClaaS judge setup. CAA's candidate strengths and intervention path
are in [`configs/tuning.json`](../../configs/tuning.json).

The configured grid is an initial Gemma development grid. The authors'
[experiment scripts](https://github.com/nrimsky/CAA/blob/main/script.sh#L12-L21)
use `[-1, -0.5, 0, 0.5, 1]` for fixed-layer multiple-choice sweeps and
`[-2, -1.5, -1, 0, 1, 1.5, 2]` for open-ended sweeps. Their
[normalization](https://github.com/nrimsky/CAA/blob/main/normalize_vectors.py#L9-L26)
rescales vector norms across behaviors at each model/layer. With our different
model, construction examples, and raw vectors, these coefficients do not directly
calibrate Gemma's strength. Select using development results.

Build and check the artifact first, then tune on all 65 development prompts:

```bash
TUNE_ID="caa-dev-$(date +%Y%m%d-%H%M%S)"
uv run --locked python tuning.py --config configs/tuning.json --method caa --run-id "$TUNE_ID"
```

Use a fresh run ID for a new experiment; the shared workflow describes resuming an
interruption. Inspect the completed tuning summary before using its selected
strength. Development generation retains the shared batch size 4 and 2048-token
limit; the separate main multiple-choice settings remain batch size 16 and 8 tokens.

Start with paired BBQ and MMLU-Pro pilots before full evaluation. Copy
[`configs/main.json`](../../configs/main.json) to an ignored results directory,
set only CAA's strength to the development-selected value and each pilot's
benchmark `limit` to 32, then use the [evaluation runner](../../evaluation/README.md)
for both `baseline` and `caa`. Preserve the shared decoding settings, pinned data,
and ordered example IDs. Inspect paired predictions and metrics; a 32-example
prefix is a mechanics check rather than a representative benchmark sample.

Retain the constructed artifact, checker evidence, tuning evidence, pilot config,
and benchmark outputs, including each run's runtime evidence. The
[results documentation](../../results/README.md) defines tuning and benchmark
files. Copy them back from the cluster using the connection route in the
[cluster guide](../../docs/SoC_Cluster_Guide.md#2-connect-to-the-cluster); metadata
alone cannot restore the ignored vector or raw outputs.

Construction contrasts and application checks do not establish reduced bias in
generated answers. All [proposal benchmarks](../../CS3268_Project_Proposal_Group_6.md#32-dataset-and-metrics)
remain required; no CAA tuning or benchmark conclusion is recorded here.
