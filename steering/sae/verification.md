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
were extracted locally into the paths below. The retained intervention's
`config.json` is committed; its generated files, checks, and logs remain Git-ignored.

| Path | Contents |
| --- | --- |
| `steering_vectors/sae/cu126-20261105-202310/` | Metadata, decoder direction, activations, and feature scores |
| `results/sae_checks/sae-cu126-20261105-202310/checks.json` | Application-check settings, provenance, results, and generations |
| `results/sae_checks/sae-cu126-20261105-202310-logs/` | Environment, GPU, source commit, build, check, and comparison logs |

## Development strength tuning

### Retained SAE result

On 2026-11-07, the user chose `sae-tuning-mig-20261107-163819` as the sole
retained SAE tuning run and selection basis. It covers all eight configured
strengths on the H100 NVL MIG `3g.47gb` profile. Its
[summary](../../results/tuning/sae-tuning-mig-20261107-163819/summary.json)
selects strength 0.25 under the
[common tuning rule](../README.md#strength-hyperparameter-tuning).
[The final configuration](../../configs/main.json) already records that strength
and the cu126 intervention; neither setting changes.

The tuning run loaded `steering_vectors/sae/cu126-20261105-202310`, constructed
on 2026-11-05; it did not construct a new intervention. Its
[saved configuration](../../results/tuning/sae-tuning-mig-20261107-163819/config.json)
records that directory, embeds its metadata, and hashes all four intervention
files. The [retained intervention metadata](../../steering_vectors/sae/cu126-20261105-202310/config.json)
and local file hashes match that record. The date in `intervention_dir` therefore
identifies construction, while the tuning run ID identifies strength selection.

This run supersedes the two earlier SAE tuning runs, whose directories have
been removed from this checkout. Their committed configurations, summaries, and
[verification history](https://github.com/Whimsyturtle/CS3268-Project/blob/be0d608086ead4d1594546fbdba0b0507864c910/steering/sae/verification.md#development-strength-tuning)
remain in Git history. The [runtime investigation](#baseline-reproduction-check)
below retains the findings that informed the shared cluster requirements.

The shared configuration freeze and held-out benchmark conclusions remain
pending, including verification of matching GPU allocation profiles across
methods under the [comparison settings](../../configs/README.md).

### Eight-condition MIG tuning rerun

Run `sae-tuning-mig-20261107-163819` completed on `xgpi12`, Slurm job `1090406`.
Its [saved configuration](../../results/tuning/sae-tuning-mig-20261107-163819/config.json)
records source `8ba0c958e70af726fc37d4fbc002288731caeda9`, no tracked diff,
and the untracked earlier results directory `results/tuning/sae-tuning-20261107-121116/`;
this was not a clean checkout. The downloaded environment log records H100 NVL
MIG `3g.47gb`, 49,794,777,088 bytes of device memory, 60 SMs, and driver
`580.173.02`, with Python `3.13.18`, PyTorch `2.14.1+cu126` (CUDA `12.6`),
and Transformers `5.20.2`.

Read-only verification of the downloaded files confirmed:

- Exactly 520 unique generations and matching judgments, with the same 65
  development IDs, prompts, and categories in all eight conditions. Review IDs
  and judgment schemas validate; construction and development IDs are disjoint.
- Every overall and category mean, paired count, seeded 95% bootstrap interval,
  ranking, and selected strength reproduces the saved summary.
- Recorded settings match the source configuration, and the cu126 intervention's
  metadata and all four file hashes match the retained intervention.
- The seven checked package versions in the environment log match
  the lockfile: PyTorch, Transformers, tokenizers, huggingface-hub, SAELens,
  scikit-learn, and requests.

The [verified summary](../../results/tuning/sae-tuning-mig-20261107-163819/summary.json)
contains these mean scores under the
original [tuning rubric and selection rule](../README.md#strength-hyperparameter-tuning):

| Strength | Bias avoidance | Response quality | Balanced score |
| --- | ---: | ---: | ---: |
| Baseline | 4.0154 | 4.2308 | 3.8769 |
| 0.01 | 4.0923 | 4.2615 | 3.9385 |
| 0.025 | 4.1385 | 4.3231 | 3.9846 |
| 0.05 | 4.0462 | 4.2769 | 3.9231 |
| 0.1 | 4.1231 | 4.2769 | 3.9385 |
| 0.25 | 4.2154 | 4.3077 | 4.0462 |
| 0.5 | 4.1231 | 4.2462 | 3.9692 |
| 1.0 | 4.1846 | 4.0615 | 3.7692 |

The selected strength is **0.25**, agreeing with the retained setting. Its
balanced-score gain over baseline is +0.1692 (paired 95% interval
[-0.0308, 0.4000]): 10 prompts improve, 49 tie, and 6 worsen. The interval
includes zero; this is development selection evidence, not an established
improvement or a held-out benchmark result.

The downloaded files have these SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `config.json` | `8e17f79a63474567692937bc49ea5d7e189ab3141f1a5d59d67ebd598fe7f77b` |
| `generations.jsonl` | `a2ba3b97aeb56df2c4503bb62a213d6720c7bec21c722969a449bcd5eba8688a` |
| `judgments.jsonl` | `68f09a80042ba0bb64f65e0a3ea69d2b7adfa1b5e078882401bd7cc857fd15ad` |
| `summary.json` | `9fac33f2313090d279bc30a5b1701ac0fa887870d4f02cc334dcb829b6781de4` |
| Environment log | `47f8e6238c3fb7bd7b78910bd7e1b05f31a5f3c49293dedc7f57182f2eb31f92` |

The run remains at `results/tuning/sae-tuning-mig-20261107-163819/` locally and
under `bennett@xlogin.comp.nus.edu.sg:/home/b/bennett/CS3268-Project/`.
The environment log is
`results/sae_checks/sae-tuning-mig-20261107-163819-environment.txt`.
These hashes identify the downloaded files; no separately supplied remote
checksum manifest or raw judge API envelopes were available. Verification
preserved the files and made no new generation or judge calls. The configuration
and summary accompany this record; raw files remain Git-ignored under the
[results convention](../../results/README.md#strength-tuning).

### Baseline reproduction check

The earlier runtime investigation found different generated text across GPU
allocation profiles. The superseded eight-condition run selected baseline;
four-prompt diagnostic checks reproduced its outputs on a full H100 NVL and
the original tuning outputs on H100 NVL MIG `3g.47gb`. Each allocation reproduced
its own outputs across repeats. Both checks reported driver `580.173.02` and
matching Python and package versions, while visible SM counts differed: 132
on the full GPU and 60 on MIG. The GPU profile is the leading explanation;
the exact numerical mechanism and other node-specific effects remain unresolved.

Full-device access was observed on `xgpi17`, and a later `h100-47` preflight on
`xgpi15` also reported full-device access and stopped before generation. These
findings informed the [shared runtime requirements](../../docs/Cluster_Runtime.md).
The complete diagnostic reports were not copied locally; the historical record
relies on user-supplied excerpts and output hashes checked against saved responses.

Before removal of the older runs, comparison by strength and example ID confirmed
that the retained MIG run reproduces all 325 original prompts and responses across
the five shared conditions. Fifteen of those identical responses received different
bias-avoidance and/or response-quality scores under the unchanged rubric. This
explains the small score changes between the MIG runs. The generator, development
data, lockfile, and scoring and selection code matched the original tuning source.
The detailed evidence is in the [verification history](https://github.com/Whimsyturtle/CS3268-Project/blob/be0d608086ead4d1594546fbdba0b0507864c910/steering/sae/verification.md#baseline-reproduction-check).

For a new baseline check, follow the
[shared runtime setup](../../docs/Cluster_Runtime.md), then run from the repository
root in the allocated GPU shell with the retained run's raw generations available:

```bash
uv run --locked python -m diagnostics.check_generation \
  --run-dir results/tuning/sae-tuning-mig-20261107-163819 \
  --output "results/sae_checks/mig-reproduction-$(date +%Y%m%d-%H%M%S).json"
```

[check_generation.py](../../diagnostics/check_generation.py) repeats the saved
run's first baseline batch twice and records runtime, tokenized inputs, output
texts, and exact matches. It makes no judge calls and does not select a strength.
The report checks four prompts, not the complete tuning grid; it does not replace
the retained tuning evidence.

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
