# CAA

This implementation adapts [Rimsky et al. (2024), Steering Llama 2 via Contrastive Activation Addition](https://aclanthology.org/2024.acl-long.828/)
to the project's pinned text-only Gemma 3 model and [shared examples](../../shared_examples/README.md).
The [proposal](../../CS3268_Project_Proposal_Group_6.md) remains the experiment design.

## Construction and application

For construction pair `i`, let `h_i(unbiased)` and `h_i(biased)` be decoder-block
outputs at the last assistant response content token. The raw steering vector is
`v = mean_i(h_i(unbiased) - h_i(biased))`; inference adds `strength * v` at the same
block output. Positive strength points toward the unbiased responses. There is no
normalization, PCA, finetuning, or change to the base model's parameters.

Construction uses all 260 construction pairs. The model's chat template identifies
the assistant response boundary; prompt tokens, padding, closing chat markers, and
special tokens are excluded from response content. `configs/caa.json` fixes the
zero-based block index to 13, an initial middle-layer choice for the 26-layer
Gemma-3-1b model. This is not a claim that layer 13 is optimal; it stays fixed
before strength tuning. No layer is selected using held-out benchmarks.

Application follows the [shared steering contract](../api.pyi):
`load_intervention(path, model=model)` validates the saved vector and binds it to
that model; `apply(model, intervention, strength=...)` is a context manager.
The loader checks the pinned model name, `model.config.revision`, precision, vector
hash, shape, finite values, and construction/application conventions. A nonzero
strength adds the vector only at the last nonpadding position in the initial
prompt pass and at newly processed positions in cached continuation passes. The
attention mask identifies the initial position; cache length distinguishes later
passes. Thus the first generated token can change without steering earlier prompt
positions. This implementation supports the current greedy, `use_cache=true`
runner. Strength zero registers no hooks. Every context exit removes its hooks,
including exceptions.

The artifact consists of `vector.pt` (a one-dimensional float32 tensor) and
`config.json` (method-native metadata). Metadata records the nested pinned `model`,
tokenizer revision and chat-template hash, layer/output location, extraction rule,
sign, normalization, generation scope, source example IDs, construction-file hash,
seed, source commit/working changes, and vector-file hash. Loading does not build
or download an intervention. CAA needs no dependency beyond the locked project
environment.

## NUS cluster commands

These are commands to run yourself; no cluster job has been submitted by this
change. Follow the [cluster guide](../../docs/SoC_Cluster_Guide.md) for access and
GPU allocation policy. Use your own SoC username and a plain SSH terminal. Clone
or update this repository on the login node, then obtain a Slurm compute shell
with a CUDA GPU suitable for BF16. Replace the GPU request below if the cluster's
current partition policy requires it.

```bash
ssh -J <soc_username>@stujump.comp.nus.edu.sg <soc_username>@xlogin.comp.nus.edu.sg
cd "$HOME/CS3268-Project"
salloc --gres=gpu:1 -t 120
srun --pty bash
```

Run all remaining setup and computation inside that allocated compute shell.
Do not install dependencies or run models on `xlogin`. Install `uv` on the compute
node if it is unavailable, using the project's [setup instructions](../../README.md#setup).
Accept Gemma's Hugging Face license and authenticate with your own account when
prompted; never put credentials in scripts or committed files.

```bash
cd "$HOME/CS3268-Project"
uv sync --locked
uv run --locked hf auth login
uv run --locked python - <<'PY'
import socket, torch
assert not socket.gethostname().split('.')[0].startswith('xlogin'), 'Use a compute node'
assert torch.cuda.is_available(), 'A CUDA GPU is required for these runs'
assert torch.cuda.is_bf16_supported(), 'A BF16-capable CUDA GPU is required'
print(torch.cuda.get_device_name(), torch.__version__, torch.version.cuda)
PY
uv run --locked pytest tests/test_caa_build.py tests/test_caa_apply.py tests/test_caa_check.py tests/test_caa_integration.py
uv run --locked python shared_examples/validate.py --check-evaluation-overlap
uv run --locked python -m steering.caa.build
CHECK_ID="caa-check-$(date +%Y%m%d-%H%M%S)"
uv run --locked python -m steering.caa.check --run-id "$CHECK_ID"
```

The builder refuses to overwrite an existing intervention. For an intentional
rebuild, choose a new `--output-dir`, then point the checker and a copied tuning
config at that same directory. The checker writes
`results/caa_checks/<run-id>/checks.json`, including failed checks before raising
an error. It compares exact logits, generated token IDs, and text at strength
zero; checks finite changed logits at nonzero strength; and checks restoration
after normal and exceptional exits. It also checks hook cleanup and cheap
parameter version/storage indicators without cloning the full model; focused
tiny-model tests compare every parameter exactly. Raw generated outputs, example
IDs, settings, artifact/data/source provenance, GPU, and runtime versions are
saved. Its default strength 1 is only a mechanics check, not a tuned strength or
evidence of reduced bias. The checker uses development prompts only.

## Tune, then pilot

[`configs/tuning_caa.json`](../../configs/tuning_caa.json) uses the common tuner's
unchanged development, generator, and judge settings with the proposed CAA grid
`[0, 0.5, 1, 2]`. Zero is the shared unsteered baseline. Build the artifact before
starting the tuner. Tune on all 65 development prompts, never on BBQ or another
held-out benchmark. Tuning retains the shared development batch size 4 and
256-token generation limit; the later benchmark pilots retain the separate main
decoding settings. The stages use the same run ID so they share saved evidence:

```bash
TUNE_ID="caa-dev-$(date +%Y%m%d-%H%M%S)"
uv run --locked python tuning.py --config configs/tuning_caa.json --run-id "$TUNE_ID" --stage generate
uv run --locked python tuning.py --config configs/tuning_caa.json --run-id "$TUNE_ID" --stage judge
uv run --locked python tuning.py --config configs/tuning_caa.json --run-id "$TUNE_ID" --stage select
```

Judge calibration must pass and the summary must report completion before using
its selected strength. `configs/main.json` stays untuned (`strength: null`) until
reviewed evidence exists. For a first BBQ and MMLU-Pro pilot, the following copies
the main settings to an ignored file under `results/`, substitutes only the
development-selected CAA strength, and limits each pilot to 32 examples. The
baseline and CAA runs use the same pinned data, ordered example IDs, batch size
16, greedy decoding, caching, and 8-token multiple-choice limit. Run each paired
comparison on the same GPU type. These pilots do not replace full evaluation.

```bash
export TUNE_ID
export PILOT_ID="caa-pilot-$(date +%Y%m%d-%H%M%S)"
uv run --locked python - <<'PY'
import json, os
from pathlib import Path
summary = json.loads((Path('results/tuning') / os.environ['TUNE_ID'] / 'summary.json').read_text())
assert summary['status'] == 'complete'
config = json.loads(Path('configs/main.json').read_text())
config['methods']['caa']['strength'] = summary['selections']['caa']
for benchmark in ('bbq', 'mmlu_pro'):
    config['benchmarks'][benchmark]['limit'] = 32
assert config['decoding'] == {'batch_size': 16, 'do_sample': False, 'use_cache': True}
assert all(config['benchmarks'][b]['max_new_tokens'] == 8 for b in ('bbq', 'mmlu_pro'))
folder = Path('results/caa_pilots') / os.environ['PILOT_ID']
folder.mkdir(parents=True)  # A fresh pilot ID is required.
(folder / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
PY
for benchmark in bbq mmlu_pro; do
  for method in baseline caa; do
    uv run --locked python -m evaluation.runner \
      --config "results/caa_pilots/$PILOT_ID/config.json" \
      --method "$method" --benchmark "$benchmark" --run-id "$PILOT_ID"
  done
done
```

The 32-example prefix is a quick check, not a representative benchmark sample.
Inspect paired saved predictions and matching `example_id` order before comparing
metrics. Retain every run's resolved config and raw predictions. Use the same GPU
type and runtime versions across all methods in each reported comparison. The full
[proposal benchmarks and metrics](../../CS3268_Project_Proposal_Group_6.md#32-dataset-and-metrics)
remain required; HolisticBias and AlpacaEval scorer dependencies tracked in issues
#11 and #12 remain pending. This implementation supplies no final bias or general
capability conclusion.

Bundle artifacts and evidence from the compute shell, then retrieve them from a
local terminal using your own username. Keep the same compute shell so the run-ID
variables remain available:

```bash
BUNDLE="caa-$PILOT_ID.tar.gz"
tar -czf "$BUNDLE" uv.lock configs/main.json configs/caa.json configs/tuning_caa.json steering_vectors/caa \
  "results/caa_checks/$CHECK_ID" "results/tuning/$TUNE_ID" \
  "results/caa_pilots/$PILOT_ID" \
  "results/bbq/baseline/$PILOT_ID" "results/bbq/caa/$PILOT_ID" \
  "results/mmlu_pro/baseline/$PILOT_ID" "results/mmlu_pro/caa/$PILOT_ID"
sha256sum "$BUNDLE"
# On your local computer, outside the compute/login shell:
# Substitute the printed bundle filename below.
scp -J <soc_username>@stujump.comp.nus.edu.sg \
  <soc_username>@xlogin.comp.nus.edu.sg:CS3268-Project/BUNDLE_FILENAME.tar.gz .
```

Supplied unbiased/biased responses define a construction contrast; separation on
those responses does not establish reduced bias in generated answers. The shared
overlap command checks exact normalized text against formatted benchmark prompts;
it does not prove absence of semantic or source overlap. Inspect pair quality and
data provenance before reportable experiments.
