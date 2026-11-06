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

Strength 1 is an application check only; these checks do not establish bias
reduction or preserved capability. See [development tuning](#development-strength-tuning)
for the subsequent strength selection.

## cu126 rebuild and verification

Run `sae-cu126-20261105-202310` repeated construction and checked the new
intervention on 2026-11-05, using clean source commit
`7f1f84c6260a6ec6e255ac678971ffbd8e74be8c`. The retained environment log records
Python 3.14.9, PyTorch `2.14.1+cu126`, CUDA build 12.6, and an NVIDIA H100 NVL MIG
3g.47gb. The archived lockfile matches that source commit byte for byte.

The base model, SAE checkpoint, BF16 precision, seed, and 260 construction pairs
match the original run. Feature 292 remains the highest-scoring feature by
construction AUROC:

| Measurement | Original run | cu126 rebuild |
| --- | ---: | ---: |
| Selected feature | 292 | 292 |
| Construction AUROC | 0.8428032544378699 | 0.8428180473372782 |
| Activation scale | 606.2564697265625 | 606.4310913085938 |

The decoder direction is byte-identical to the previously verified checkpoint
row. The scale increased by 0.0288%; numerical outputs are not identical across
these runs. The cu126 runner-up is feature 10115, with construction AUROC
0.810258875739645. This comparison establishes the same construction winner for
these two runs, not the best feature for bias reduction on held-out generations.

Local inspection of the downloaded bundle confirmed:

- Construction IDs and the data hash match the committed examples. Both saved
  activation tensors are finite, nonnegative float32 arrays of shape
  `[260, 16384]`.
- An independent rank-sum calculation reproduces all 16,384 saved AUROCs within
  `2.3e-16`; the selected feature and activation scale match the saved metadata.
- Build and application-check provenance both report the clean source commit.
  The check used the newly built intervention and the configured model,
  development prompts, seed, and decoding settings.
- All six application checks passed at strength 1 with eight output tokens.
  Saved zero-strength and restored generations equal baseline; steered
  generations differ.

The commands used on the GPU compute node were:

```bash
uv sync --locked
uv run --locked python -m steering.sae.build --output-dir steering_vectors/sae/cu126-20261105-202310
uv run --locked python -m steering.sae.check --intervention-dir steering_vectors/sae/cu126-20261105-202310 --run-id sae-cu126-20261105-202310
```

For another run, follow the [setup](../../README.md#setup) and
[build instructions](README.md#build-and-check), using an unused output directory
and run ID. The original intervention remains at `steering_vectors/sae/`; the
cu126 rebuild is retained separately at the path above.

The bundle is
`bennett@xlogin.comp.nus.edu.sg:~/sae-cu126-20261105-202310.tar.gz`, with SHA-256
`6bd4911beee0aa764c22158ccebfd96050aef0f76469974c52a4599f82e82f4c`.
The downloaded copy matches this hash. Its intervention, check, and log files
were extracted locally into their Git-ignored repository paths:

| Path | Contents |
| --- | --- |
| `steering_vectors/sae/cu126-20261105-202310/` | Metadata, decoder direction, activations, and feature scores |
| `results/sae_checks/sae-cu126-20261105-202310/checks.json` | Application-check settings, provenance, results, and generations |
| `results/sae_checks/sae-cu126-20261105-202310-logs/` | Environment, GPU, source commit, build, check, and comparison logs |

## Development strength tuning

Run `sae-tuning-20261106-202053` used clean source commit
`8d8bbe21f3b50a8f2b35b6786ce0b5daa3ccb4b0` and the cu126 intervention above.
The [saved configuration](../../results/tuning/sae-tuning-20261106-202053/config.json)
records the development-only settings and intervention hashes. The
[summary](../../results/tuning/sae-tuning-20261106-202053/summary.json) selects
strength 0.25 under the [common tuning rule](../README.md#strength-hyperparameter-tuning).
[The final configuration](../../configs/main.json) uses that strength and the
same intervention directory.

Inspection of the downloaded run confirmed:

- All five conditions contain the same 65 development IDs, prompts, and
  categories, with exactly 325 unique generations and matching judgments.
- Recomputing every overall and category score, paired comparison, seeded
  bootstrap interval, and ranking reproduces the summary and selected strength.
- The recorded intervention metadata and all four file hashes match the retained
  cu126 intervention. The archived lockfile matches the source commit exactly.
- Package versions recorded after the run match the lockfile for PyTorch,
  Transformers, SAELens, scikit-learn, requests, and huggingface-hub. The GPU
  snapshot recorded after the run shows an H100 NVL with a 47 GB MIG device.

The selected strength's paired 95% confidence interval touches zero. This is
development selection evidence; final benchmark evaluation and claims about
bias reduction or preserved capability remain pending.

The separate [coherence audit](coherence.md) is prepared for a broader strength
sweep; its generation, judging, and manual review results remain pending.

The complete run is retained at
`bennett@xlogin.comp.nus.edu.sg:/home/b/bennett/CS3268-Project/results/tuning/sae-tuning-20261106-202053/`
and was copied locally to the same repository-relative path. Raw generations,
judgments, logs, and environment snapshots remain Git-ignored under the
[results convention](../../results/README.md#strength-tuning).

## Original retained artifacts

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
