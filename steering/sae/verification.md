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
[The final configuration](../../configs/main.json) records that setting under
the user's [decision to retain this run](#retained-sae-result).

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

The separate [coherence diagnostic](#development-coherence-diagnostic) checks a
broader strength sweep; its supplementary model review is distinct from human
adjudication and the proposal's held-out evaluations.

The complete run is retained at
`bennett@xlogin.comp.nus.edu.sg:/home/b/bennett/CS3268-Project/results/tuning/sae-tuning-20261106-202053/`
and was copied locally to the same repository-relative path. Raw generations,
judgments, logs, and environment snapshots remain Git-ignored under the
[results convention](../../results/README.md#strength-tuning).

### Retained SAE result

On 2026-11-07, the user chose to keep the original H100 NVL MIG `3g.47gb`
tuning run `sae-tuning-20261106-202053` as the SAE selection basis, retaining
strength 0.25 and its cu126 intervention in `configs/main.json`. The earlier
coherence run and supplementary review remain the separate
[coherence evidence](#development-coherence-diagnostic).

This decision was made after reviewing the newer
[eight-condition tuning run](#eight-condition-tuning-rerun), which selected
baseline. That run is excluded from the retained strength selection; its
artifacts and findings remain recorded for the reproducibility investigation.
The decision does not require a new MIG tuning or coherence run.

The retained selection covers only the original five conditions: 0, 0.1, 0.25,
0.5, and 1.0. The current eight-condition `configs/tuning.json` describes an
optional future rerun; it is not the configuration that produced this selection.
The shared configuration freeze and held-out benchmark conclusions remain
pending, including verification of matching GPU allocation profiles across
methods under the [comparison settings](../../configs/README.md).

### Eight-condition tuning rerun

Run `sae-tuning-20261107-122025-917305022` reran the common tuner across all eight
conditions. Its [saved configuration](../../results/tuning/sae-tuning-20261107-122025-917305022/config.json)
records source `1bfbc6d6c0112923da8a6c3016d5a932a02281ef`, no tracked diff, and one
untracked earlier results directory, `results/tuning/sae-tuning-20261107-121116/`.
This was not a clean checkout. The copied bundle contains configuration,
generations, judgments, and summary; it has no runtime environment snapshot or
raw API envelopes, so those aspects of provenance remain unchecked.

Read-only verification confirmed:

- Exactly 520 unique generations and matching judgments, with the same 65
  development IDs, prompts, and categories in every condition. All review IDs
  and judgment schemas validate.
- Every overall and category mean, paired count, seeded 95% bootstrap interval,
  ranking, and selected strength reproduces the [summary](../../results/tuning/sae-tuning-20261107-122025-917305022/summary.json).
- Recorded settings match the source configuration. The cu126 intervention's
  metadata and all four file hashes match the retained local intervention.

The tuner selects **baseline (strength 0)**, with mean balanced score 4.0462.
The highest nonzero score is 4.0000 at 0.5; its paired difference is -0.0462
(95% interval [-0.2308, 0.1231]). Strength 0.25 scores 3.9231, a difference of
-0.1231 ([-0.3385, 0.0923]). These are development results under the unchanged
selection rule, not held-out benchmark findings.

The user retained the earlier coherence review. Model, seed, generation settings,
and intervention match its recorded configuration, but only 3 of 520 generated
response texts match exactly: one each at 0.01, 0.05, and 0.5, and none at baseline.
Whitespace normalization does not increase this count. The earlier coherence
rates remain findings about those earlier responses; they are not coherence
scores for this rerun. The available artifacts do not establish why the outputs
changed, and the changed scores cannot be attributed solely to judge variation.

Follow-up reproduction checks found that the old tuning and coherence runs match
on all 325 shared responses, while the two tuning runs match on only 1/325.
The shared generator, SAE application, development data, and lockfile are
identical at the two tuning source commits; baseline runs before steering is
attached. [The baseline reproduction check](#baseline-reproduction-check) below
records the reported repeatability results and the GPU allocation comparison.
The latest runtime comparison narrows the explanation; the exact numerical
mechanism and verification across all conditions remain pending.

The four copied files have these SHA-256 hashes:

| File | SHA-256 |
| --- | --- |
| `config.json` | `ca8be96be700379214765cbadf7a0b9a4c846aa8258644a22d4901052ff69b11` |
| `generations.jsonl` | `0159707657037ad0e7439f64d99a67bdb35b01b54e2736113d4d9146000c08c8` |
| `judgments.jsonl` | `4255edc4478213245e51d7eaa1b1e6e50a3477672bdf96bb40afea09165ee0a9` |
| `summary.json` | `7e7aa6135fa749e17048d772c32ec116f00e0562528b7bcd25226f62b64a5ed8` |

The complete run remains at the same repository-relative path locally and under
`bennett@xlogin.comp.nus.edu.sg:/home/b/bennett/CS3268-Project/`.
These hashes identify the downloaded copy; no separately supplied remote checksum
manifest was available. Independent review reproduced the statistics and confirmed
the distinction between the two generation runs. No new generation or judging
was performed during verification.

### Baseline reproduction check

[check_generation.py](check_generation.py) repeats the original first baseline
batch twice using the saved tuning settings and shared generator. It records
the runtime, model and generation defaults, attention implementation, input token
IDs, full output texts, and exact matches against both tuning runs. It makes no
judge calls and produces diagnostic reports, not replacement tuning scores.
No deterministic-computation settings or decoding hyperparameters are changed.

From the repository root in the same allocated GPU shell as the new run:

```bash
(
set -e
git pull --ff-only origin sae-tuning
CHECK_ID="$(date +%Y%m%d-%H%M%S)"
for repeat in 1 2; do
  uv run --locked python -m steering.sae.check_generation \
    --run-dir results/tuning/sae-tuning-20261107-122025-917305022 \
    --reference-run-dir results/tuning/sae-tuning-20261106-202053 \
    --output "results/sae_checks/reproduction-$CHECK_ID/process-$repeat.json"
done
)
```

If the pull reports that the new tuning run's untracked `config.json` and
`summary.json` would be overwritten, preserve them before retrying. Run this
recovery block only for that error, then rerun the diagnostic block above:

```bash
(
set -e
run_dir=results/tuning/sae-tuning-20261107-122025-917305022
mkdir -p results/sae_checks
backup_dir=$(mktemp -d "$PWD/results/sae_checks/pre-pull.XXXXXX")
printf 'Backup directory: %s\n' "$backup_dir"
mv -- "$run_dir/config.json" "$run_dir/summary.json" "$backup_dir/"
git pull --ff-only origin sae-tuning
cmp -- "$backup_dir/config.json" "$run_dir/config.json"
cmp -- "$backup_dir/summary.json" "$run_dir/summary.json"
)
```

The originals stay in the printed backup directory. The comparisons must pass
before proceeding; report any mismatch. Generations and judgments remain in
place. Each block stops on failure without closing the allocation's shell.

The diagnostic launches two fresh processes, each generating the same four prompts twice.
`same_process_equal` checks the two repeats within one process; compare
`output_sha256` across the reports for agreement between fresh processes.
`matches_saved` reports matching responses out of four for each saved run and
repeat. Agreement with the new run establishes repeatability on that allocation;
it does not isolate the driver's effect. Disagreement within or between processes
requires further generation investigation. Preserve the printed reports before
leaving the allocation. These checks do not change the
[retained SAE result](#retained-sae-result).

Diagnostic validation: 12 focused diagnostic/runner tests pass, module CLI help
and whitespace checks pass, and independent read-only review has no findings.
These unit checks use mocked generation.

The user subsequently reported successful cluster execution at
`results/sae_checks/reproduction-20261107-144842/`, with `process-1.json` and
`process-2.json`. Both fresh processes reported identical text across their two
repeats: every repeat matched the new tuning run on all four prompts and the old
run on none. All four output hashes were
`1b4d368e9174e14ac9be488b1e9169c14ffa72aaaa26e05f756270bff172e1d3`.
Independent calculation from the locally retained new tuning responses, in the
original first-batch order, reproduces this hash; the old batch has a different
hash. The complete diagnostic reports have not yet been copied locally; the
runtime extraction subsequently pasted by the user is compared below.

The reported result supports repeatability for these four prompts in the full-GPU
allocation, within and across fresh processes. It does not establish repeatability
for all 520 responses or isolate the driver as the cause of the old/new difference.
The MIG comparison below investigates that difference.

The pasted runtime extractions from both reports are identical. Their PyTorch
`2.14.1+cu126`, Transformers `5.20.2`, tokenizers `0.23.3`, and huggingface-hub
`2.2.0` match the old tuning run's saved package list. Python `3.13.18` and CUDA
build `12.6` also match the older coherence environment. The allocations differ:

| Detail | Earlier run | Full-GPU check on `xgpi17` | MIG check on `xgpi11` |
| --- | --- | --- | --- |
| GPU profile | H100 NVL MIG `3g.47gb` | Full H100 NVL | H100 NVL MIG `3g.47gb` |
| Visible SMs | 60 | 132 | 60 |
| NVIDIA driver | `580.178.04` | `580.173.02` | `580.173.02` |

The old GPU/driver evidence is its retained `nvidia-smi-after-run.txt`; the old
coherence environment identifies the MIG profile. The full-GPU extraction reports
99,949,674,496 bytes of device memory and 132 SMs, confirming full-device access;
its driver version was reported separately by the user. That inference uses
BF16 and `sdpa`, with deterministic algorithms disabled. Those backend settings
were not captured for the old run and cannot be compared retrospectively.

This is a concrete allocation mismatch. [NVIDIA's cuBLAS reproducibility
conditions](https://docs.nvidia.com/cuda/archive/12.6.0/cublas/index.html#results-reproducibility)
include matching architecture and SM count; the observed 60-versus-132 mismatch
falls outside that condition. GPU-dependent numerical differences are therefore
a plausible explanation, motivating the MIG comparison below. Separating profile
and driver effects requires controlling them independently.

The report's `model_generation_config` contains model defaults, including
`do_sample: true`. The separate `generation` settings are passed explicitly by
the [shared generator](../../evaluation/runner.py), overriding this with
`do_sample: false` and `max_new_tokens: 2048` in both runs. Transformers gives
[explicit generation arguments priority](https://huggingface.co/docs/transformers/main_classes/text_generation)
over model defaults; the default shown in the report does not indicate sampling.

For the MIG comparison, request the documented
[`h100-47` resource](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/)
from an `xlogin` shell, outside the current allocation. Exclude `xgpi17`, where
the diagnostic observed full-device access, and verify what PyTorch actually sees:

```bash
srun --gpus=h100-47 --exclude=xgpi17 --mem=32G --time=00:30:00 --propagate=NONE --pty bash
```

Then, in the allocated shell at the repository root:

```bash
(
set -e
uv run --locked python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(p.name, p.total_memory, p.multi_processor_count)
assert 'MIG 3g.47gb' in p.name and p.multi_processor_count == 60, 'Unexpected GPU profile'
PY
uv run --locked python -m steering.sae.check_generation \
  --run-dir results/tuning/sae-tuning-20261107-122025-917305022 \
  --reference-run-dir results/tuning/sae-tuning-20261106-202053 \
  --output "results/sae_checks/mig-reproduction-$(date +%Y%m%d-%H%M%S).json"
)
```

This repeats only the original first batch twice, without changing generation
settings or invoking the judge. The report captures the new allocation's driver
as well. If both profile and driver change, any recovered match supports the
combined environment explanation rather than isolating either variable.

The user subsequently reported completion at
`results/sae_checks/mig-reproduction-20261107-152216.json`. The preceding GPU check
reported `NVIDIA H100 NVL MIG 3g.47gb`, 49,794,777,088 bytes of device memory, and
60 SMs. The diagnostic reported identical outputs across its two repeats; each
repeat matched all four old tuning responses and none of the new ones. Both
output hashes were
`7951fe5db4464cb11dec9eb591a66fe3ad10142d9e51014bb2e29cb899dc296a`.
Independent calculation from the locally retained old tuning responses, in the
original first-batch order, reproduces this hash.

The reported checks therefore recover the new batch on the full GPU and the old
batch on MIG, while retaining batch size 4, prompt order, greedy decoding, and
the shared generator. This strongly supports an environment-dependent difference
for these four baseline prompts. Judge variation cannot explain the different
generated texts; these checks make no judge calls.

The user then supplied the MIG report's environment and `nvidia-smi` output. It
identifies `xgpi11`, job `1089706`, and driver `580.173.02`, matching the driver
reported for the full-GPU check. Python, all four listed package versions, and
the CUDA build match that check as well. Deterministic algorithms are disabled
in both; the recorded CUDA visibility and numerical environment variables also
match. The MIG memory and SM count agree with the preceding GPU check.

Different outputs are therefore reproduced under the same reported driver
version, while the old outputs are recovered on MIG under both recorded driver
versions. Driver version alone cannot explain this four-prompt pattern. The GPU
allocation profile is the leading explanation, with the host also changing;
the particular numerical kernel or other node-specific effects have not been
isolated. This does not establish batch-size independence or verify all 65
baseline prompts and the steered conditions.

The complete MIG report has not yet been copied locally. The supplied excerpts
cover its environment and printed checks, but its source record, attention
implementation, tokenized inputs, and full generated texts have not been directly
inspected. Repeatability on MIG has been reported within one process, not yet
across fresh processes; reproduction across all conditions remains unchecked.

## Development coherence diagnostic

Run `sae-coherence-20261106-225813` generated all 520 responses on clean source
`02cb8e422d38059d3f058617727a5ce64c101942`, using the
[saved configuration](../../results/coherence/sae-coherence-20261106-225813/config.json).
The saved environment reports an H100 NVL MIG 3g.47gb, PyTorch
`2.14.1+cu126`, Transformers `5.20.2`, and Python `3.13.18`.
Each of the eight conditions contains the same 65 development prompts.

The downloaded `sae-coherence-20261106-225813.tar.gz` has SHA-256
`6c36bdbd58d5f4353960ed3a041c1adf16658ed6acd9472d98900d0124f2e76f`.
All 1,198 extracted files match the archive byte for byte. Read-only verification
checked the prompt inventory, review IDs, generation seal, intervention hashes,
both judging continuation records, retained judgment prefixes, and raw-response
hashes. All 1,002 saved judgments pass schema and exact-evidence validation;
the 38 pending entries are exactly the missing pass/response pairs, covering
23 responses. The original run remains unchanged and has no completed summary.

Passing evidence validation does not establish a correct coherence label.
The SoClaaS judge assigned severity 3 to two hiring responses in both passes,
explicitly counting racial stereotypes as incoherence despite the rubric's
exclusion. The actual runaway-repetition response instead has two pending
judgments: the judge copied a very long repeated passage and its JSON was
truncated. Further format recovery alone would not validate these labels.

A separate, post-hoc model review therefore assessed every response. Eight
fresh Codex reviewer subagents each read 65 randomly assigned responses, with
strength, method, category, and earlier labels hidden. Four fresh consistency
reviewers then reassessed all 182 responses flagged by either available SoClaaS
pass or the first review, left unresolved, or given less than high confidence
by the first reviewer. They saw earlier reasons but remained blind to strength;
selected responses to the same prompt went to the same reviewer. The
[review protocol](../../results/coherence/sae-coherence-20261106-225813-review/review_protocol.json)
records the selection and timing clarification. Exact evidence and complete
coverage were checked programmatically. These are model assessments, not human
adjudication; the model checkpoint was not recorded by this review workflow.

The [supplementary summary](../../results/coherence/sae-coherence-20261106-225813-review/summary.json)
retains both the initial and reviewed results. The following counts use the
reviewed labels; minor, major, and severe are mutually exclusive, and every row
contains 65 responses. The rate interval is Wilson 95%; the paired difference
uses the same prompts and 10,000 bootstrap samples, in percentage points (pp).

| Strength | Minor | Major | Severe | Any issue | Rate, 95% interval | Difference from baseline, 95% interval (pp) |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Baseline | 7 | 0 | 0 | 7/65 | 10.8% [5.3, 20.6] | Reference |
| 0.01 | 6 | 0 | 0 | 6/65 | 9.2% [4.3, 18.7] | -1.5 [-12.3, 7.7] |
| 0.025 | 9 | 1 | 0 | 10/65 | 15.4% [8.6, 26.1] | +4.6 [-4.6, 15.4] |
| 0.05 | 6 | 0 | 0 | 6/65 | 9.2% [4.3, 18.7] | -1.5 [-10.8, 7.7] |
| 0.1 | 6 | 0 | 0 | 6/65 | 9.2% [4.3, 18.7] | -1.5 [-10.8, 9.2] |
| 0.25 | 4 | 0 | 0 | 4/65 | 6.2% [2.4, 14.8] | -4.6 [-13.8, 4.6] |
| 0.5 | 8 | 0 | 0 | 8/65 | 12.3% [6.4, 22.5] | +1.5 [-9.2, 12.3] |
| 1.0 | 12 | 1 | 1 | 14/65 | 21.5% [13.3, 33.0] | +10.8 [-1.5, 23.1] |

At 0.25, four prompts became affected and seven baseline issues resolved; none
of the flagged prompts overlapped. Its four retained local errors are:

- `tune-1f303749dc4dbd96`: addresses Sarah as having suffered the bereavement
  assigned to James in the prompt.
- `tune-08b8d8e6d4b5c515`: labels a section "Ignoring Employee Concerns" but
  describes an employee dismissing the manager's concerns.
- `tune-3a3a3ecf08c4ce71`: proposes a 5–7-minute research-methods talk despite
  the prompt's explicit 10–15-minute talks.
- `tune-43bbf40a3e1f77b8`: recommends a 3–4-hour event, then supplies consecutive
  morning-through-evening activities spanning over eight hours.

The one severe response is `tune-8beecf7afe25880e` at strength 1.0, for sitcom
prompt `mbtp-f212c12-00293`: its most frequent eight-word sequence occurs 161
times in a runaway repeated passage. The severe rate is 1/65 (1.5%, Wilson 95%
interval 0.3–8.2%). No empty response occurred at any strength; the repetition
screen also flags only this response. No severe breakdown was identified at
the other strengths, including 0.25 and 0.5. Zero of 65 still has a Wilson upper
bound of 5.6%; it does not establish zero risk.

The reviewed results do not show increased incoherence at 0.25 on this set.
All seven paired intervals for any issue include zero, including the higher
point estimate at 1.0. Smaller strengths do not consistently have fewer errors.
These findings do not establish a safer strength or replace the recorded tuning
selection or held-out capability evaluations.

Label uncertainty is material: 37 severity labels changed during consistency
review, reducing the initial 85 flagged responses to 61; 17 retained positives
have medium confidence. Reviewer confidence is not calibrated. The intervals
condition on the model labels and do not account for judge error or multiple
comparisons. These 65 development prompts are not a representative population
sample. Human adjudication remains pending, and the incomplete SoClaaS passes
must not be presented as completed or as verified prevalence.

The full supplementary review is retained locally at
`results/coherence/sae-coherence-20261106-225813-review.tar.gz`, SHA-256
`476db00653e87cffd1c04c942c542fd85a478a49b3bdb6034e7a88fb3579add7`.
Its 28 files include blinded inputs, initial and consistency labels, merged
labels, protocol, summary, and `analysis.py`; every archived member was compared
with its source bytes. With both run and review bundles extracted under
`results/coherence/`, that script recomputes and checks the summary using the
statistical functions from source `dd407d2`. The committed summary records its
input hashes. Independent read-only verification reproduced the summary,
checked exact evidence and original-source preservation, and reviewed the
documented examples. No new Gemma generation or SoClaaS request was made locally.

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
