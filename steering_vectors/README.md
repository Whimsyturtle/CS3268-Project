# Saved interventions

`caa/`, `icv/`, `sae/`, and `reft_r1/` hold vectors and any additional parameters
needed for inference.

Each folder's `config.json` records how its intervention was produced: model
revision, layers, source example IDs, seed, and method settings (including SAE
checkpoint/features or ReFT-r1 training settings where applicable).
Populate it when generating the intervention; the current `{}` files are
placeholders, not runnable configs.

Configs and this README are tracked by Git; generated intervention files are ignored.
