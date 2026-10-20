# Saved interventions

`caa/`, `icv/`, `sae/`, and `reft_r1/` hold vectors and any additional parameters
needed for inference.

Each folder's `config.json` records how its intervention was produced: model
revision, layers, source example IDs, seed, and method settings (including SAE
checkpoint/features or ReFT-r1 training settings where applicable).
Populate it when generating the intervention.

Configs and this README are tracked by Git. Generated intervention files are ignored.
