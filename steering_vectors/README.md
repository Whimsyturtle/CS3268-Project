# Saved interventions

`caa/`, `icv/`, `sae/`, and `reft_r1/` hold vectors and any additional parameters
needed for inference. Named subdirectories hold separately constructed candidates.

Each folder's `config.json` records how its intervention was produced: model
revision, layers, source example IDs, seed, and method settings (including SAE
checkpoint/features or ReFT-r1 training settings where applicable).
Populate it when generating the intervention.

`reft_r1_random_control/` holds a random vector used only in
[ReFT-r1 side checks](../results/reft_r1_checks/summary.md).
`reft_r1/positive-only/` holds the earlier ReFT-r1 vector, trained without
[neutral examples](../GLOSSARY.md#neutral-example), kept for comparison.

Commit the `config.json` for each retained intervention, including named
subdirectories. Generated intervention files remain Git-ignored and must be
restored from the retained artifacts before use; the
[SAE verification record](../steering/sae/verification.md#cu126-rebuild-and-verification)
identifies its bundle. Git preserves metadata with LF line endings so its saved
hashes remain valid across platforms.
