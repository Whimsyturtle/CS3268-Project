# SAE verification

[Issue #9](https://github.com/Whimsyturtle/CS3268-Project/issues/9) was verified on
2026-11-05 using source commit `f022357e9c1c71ee8d52da2beaede09fc222afd8`.
Construction selected feature 292 with construction AUROC 0.8428 from 260 pairs;
the complete provenance is in the [saved metadata](../../steering_vectors/sae/config.json).

## Checks

- The build used a clean checkout. The application check's only uncommitted
  change was the generated intervention metadata, matching the saved file.
- Construction IDs and the data hash match the committed shared examples.
  Both saved activation tensors have shape `[260, 16384]` and finite,
  nonnegative float32 values. Recomputing selection reproduces all 16,384 saved
  AUROCs and the activation scale exactly; an independent rank-sum calculation
  selects the same feature.
- The saved direction matches the selected float32 decoder row from the pinned
  pretrained SAE checkpoint byte for byte. Its dimension is 1152, matching Gemma.
- Run `sae-application-20261105-153340` passed all six checks in
  [the application checker](check.py): zero-strength logits and generations
  match baseline, steered logits are finite and changed, and normal and
  exceptional exit restore the baseline. It ran on an NVIDIA H100 NVL MIG
  3g.47gb in BF16, using two development prompts, strength 1, and eight output
  tokens. Package versions match the
  [lockfile at the verified source commit](https://github.com/Whimsyturtle/CS3268-Project/blob/f022357e9c1c71ee8d52da2beaede09fc222afd8/uv.lock).
- The saved overlap log reports no exact normalized shared-text overlap with
  58,398 BBQ, 15,713 HolisticBias, 12,032 MMLU-Pro, and 805 AlpacaEval prompts.
- The implementation's full test suite passed: 40 tests at the source commit
  above. These tests include native Gemma and SAELens integration with small
  synthetic weights.

Strength 1 is an application check only. Development tuning and final benchmark
evaluation remain pending; these checks do not establish bias reduction or
preserved capability.

## Check the saved intervention with cu126

The run above predates the [cu126 setup](../../README.md#setup). Its evidence
remains tied to the original source and environment. On an allocated GPU compute
node, sync the new lockfile, run the setup's CUDA check, then reuse the saved
intervention for a new application check:

```bash
uv sync --locked
uv run --locked python -m steering.sae.check --run-id sae-application-cu126-001
```

This check remains pending. It does not repeat feature selection or establish
that construction scores are identical across CUDA builds. To reproduce
construction under cu126, build into a separate output directory using the
[build instructions](README.md#build-and-check) and compare the saved selection
and activation scale before replacing the original artifacts.

## Retained artifacts

The original bundle is
`bennett@xlogin.comp.nus.edu.sg:~/sae-verification.tar.gz`. Its SHA-256 is
`fd37fdaacc2739485734da1880b825a7e380773bf3b867b4a64af90982244ba7`.
An identical copy was downloaded locally, checked, and extracted into the
repository paths below. Raw artifacts remain Git-ignored under the
[saved-intervention](../../steering_vectors/README.md) and
[results](../../results/README.md) conventions; retain the bundle for transfer.

| Path | SHA-256 |
| --- | --- |
| `steering_vectors/sae/config.json` | `3a9c65109639bdc8be4c93224d6ba27eb76eb91aa4f8f0d3d43292019ae6db76` |
| `steering_vectors/sae/direction.pt` | `6287a6c3cb9d5544815a68a03dbd497a526f9c2d41209facb54ef8737cd94d4f` |
| `steering_vectors/sae/activations.pt` | `7b3f58f60c6b5763a26c34068efcaf4ef81c577f42241d77f50bfa56df838565` |
| `steering_vectors/sae/selection.json` | `8c79e0ab5f95c215b5adb199b291da65cfe90f319b2310697ae4715d0ddded7c` |
| `results/sae_checks/sae-application-20261105-153340/checks.json` | `20fde4d835a22f655bf42d5ff28b3f8b491b663d58515745a18b0ef0ed7e153a` |
| `results/sae_checks/sae-application-20261105-153340/overlap.log` | `2d0458ea030634f5e1bdd7c7640114fe954987c086041bff2eea0b9c3c686ad4` |

The direction's raw float32 values have SHA-256
`ebc70bf2ba80468b9c817aefdd52b9b957331aea83e7a3a2204b284b3723a844`,
matching decoder row 292 read directly from the pinned checkpoint. This differs
from the hash of its serialized `.pt` file above.
