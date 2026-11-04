# SAE steering

Implements [issue #9](https://github.com/Whimsyturtle/CS3268-Project/issues/9) and the
[shared interface](../api.pyi) using one selected pretrained SAE feature, as required
by the [proposal](../../CS3268_Project_Proposal_Group_6.md).

## Feature selection and application

[configs/sae.json](../../configs/sae.json) pins a middle-layer residual-stream SAE
from Google's Gemma Scope 2 suite. Its
[checkpoint configuration](https://huggingface.co/google/gemma-scope-2-1b-it/blob/b0fa29457c3601df0a70c48a15534c738d7c10e0/resid_post/layer_13_width_16k_l0_medium/config.json)
identifies the model and activation location. The base model, precision and seed
come from [the experiment configuration](../../configs/main.json).

Automatic selection follows the
SAE-A activation-ranking approach in [AxBench](https://github.com/stanfordnlp/axbench/blob/main/axbench/models/sae.py):

1. Process each shared construction prompt with its biased and unbiased responses,
   using the model's chat template and the frozen, unsteered model.
2. Encode residual-stream activations with the pretrained SAE. For each response,
   take each feature's largest activation over response content tokens. Exclude
   the prompt, padding, special tokens, and the closing chat marker/newline.
3. Score each feature by AUROC, treating unbiased responses as positive examples.
   Select the highest score, with the lowest feature ID breaking ties. Fail if no
   feature scores above 0.5.
4. Save the selected decoder direction unchanged. Define its activation scale as
   its largest pooled activation across both classes of construction responses.

The response-only mask adapts AxBench's ranking to this project's paired data.
Every pair has equal weight. Construction AUROC measures separation on this data;
it does not establish bias reduction or exclude response-style/length effects.
No development or final evaluation examples determine the feature or its scale.

At inference, add `strength * activation_scale * decoder_direction` to the chosen
layer's output at every token, during both prompt processing and subsequent
generation. This changes the residual stream directly and preserves the SAE's
reconstruction error; it does not replace the residual stream with a reconstruction.
Strength zero installs no hook. The context manager removes its own hook on normal
and exceptional exit, preserving other hooks and model parameters.

The loader uses SAELens' JumpReLU implementation with the same settings as its
[published Gemma 3 loader](https://github.com/decoderesearch/SAELens/blob/v6.54.14/sae_lens/loading/pretrained_sae_loaders.py).
It downloads config and weights at the pinned revision because the library's
convenience loader does not expose revision pinning. It uses the native model
activations without normalization or decoder-bias subtraction, and keeps the
original decoder scale. Pretrained SAE weights stay in the external Hugging Face
cache. Google's SAE metadata names the base model but does not give its training
revision; the saved intervention records the exact base revision actually used.

## Build and check

Use Python 3.12 or newer and accept the Gemma model license on Hugging Face. On the
SoC cluster, follow the [cluster guide](../../docs/SoC_Cluster_Guide.md) to obtain an
appropriate compute node; install dependencies and run these commands there.
Configure `HF_HOME` outside this repository if the default cache is unsuitable.

```bash
uv sync --locked
uv run hf auth login
uv run python shared_examples/validate.py --check-evaluation-overlap
uv run python -m steering.sae.build
uv run python -m steering.sae.check --run-id sae-application-001
```

Commit the implementation before producing reportable artifacts. Feature
construction uses one unpadded response per forward pass and does not truncate
examples. The application check uses two development prompts at a fixed test
strength and the shared decoding settings. It checks zero-strength equivalence,
finite and changed steered logits, and restoration after normal/exceptional exit.
Its outputs are implementation checks, not tuning or benchmark results.

The build writes the [saved intervention metadata](../../steering_vectors/README.md)
plus these generated, Git-ignored files under `steering_vectors/sae/`:

- `direction.pt`: the selected pretrained decoder direction.
- `activations.pt`: response-pooled scores for every feature, in construction ID order.
- `selection.json`: every feature's construction AUROC.

`config.json` records the checkpoint, feature, scale, model, example IDs and data
hash, seed, and source state. Loading checks the base-model name, revision and
precision, residual-stream location, and dimensions. Builds refuse to overwrite
existing artifacts; use `--output-dir` to create a separate candidate. The
application check accepts `--intervention-dir` and writes a new
`results/sae_checks/<run_id>/checks.json`, including environment, settings, checks
and raw outputs. Retain and link these files when updating #9.

Run focused tests with:

```bash
uv run pytest tests/test_sae.py tests/test_sae_gemma.py
```

These tests use small synthetic models and data. The pretrained application check
above remains necessary before completing #9. Strength in `configs/main.json`
stays `null` until separate development tuning is complete.
