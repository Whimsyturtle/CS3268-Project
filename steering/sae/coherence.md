# SAE development coherence audit

[audit_coherence.py](audit_coherence.py) measures coherence separately from
the common tuner's combined response-quality score. This is a development-only
diagnostic for [issue #13](https://github.com/Whimsyturtle/CS3268-Project/issues/13),
requested after inspection found contradictions and a severe repeated passage.
It does not select a strength or change [the final configuration](../../configs/main.json).
The proposal's held-out capability evaluations remain necessary.

See the [verification record](verification.md#development-coherence-diagnostic)
for the downloaded run and its separate supplementary model review.

## Protocol

[The audit configuration](../../configs/sae_coherence.json) fixes the strength
grid, model, intervention, decoding, and judge settings before the run. Generate
every condition afresh on the same GPU type and allocation profile, including
baseline; do not combine new low-strength generations with the earlier tuning
run. All conditions use the same 65 development prompts. No construction or final
evaluation prompts are added.

The judge receives only a prompt and its generated response. It does not receive
the strength, method, baseline identity, category, or another judgment. The exact
rubric and output schema are defined in `audit_coherence.py` and retained with the
run. The severity scale is:

| Severity | Meaning |
| --- | --- |
| 0 | No coherence problem detected |
| 1 | Minor local contradiction or confused reference; the answer remains understandable |
| 2 | Major inconsistency or confusion undermining a substantial part of the answer |
| 3 | Severe breakdown, including sustained nonsense, runaway repetition, or an empty response |

A severe passage can warrant severity 3 even when the rest of the response is
useful. Conflicts with explicit prompt facts count, such as charging admission
for an explicitly free event. Factual errors requiring outside knowledge,
stereotypes, safety refusals, omissions, and stylistic differences alone do not
count as incoherence. An abrupt ending at the output length limit alone does not
count either. Complementary advice and clearly explained alternatives are not
contradictions. Positive judgments must quote supporting text.

Each response is judged twice in separately shuffled orders, using the same
SoClaaS model and rubric. These repeated calls measure consistency; they are not
independent judges, and agreement does not establish correctness. The configured
run generates 520 responses and requests 1,040 valid judgments, plus any retries
for malformed judge output. The service model ID is not an immutable checkpoint.

For each pass and strength, the summary reports severity counts and the rates of
any issue (severity at least 1), major-or-worse issues (at least 2), and severe
issues (3). Wilson 95% intervals accompany rates, including when no failures are
observed. Paired comparisons use the same prompt at baseline and each strength,
report newly affected and resolved cases, and bootstrap the difference in rates.
Positive differences mean more coherence failures under steering. Intervals are
pointwise, exploratory estimates; they do not correct for multiple comparisons
or judge errors. These development prompts are not a representative sample of
all future use.

The two passes remain separate. Their disagreements, all flagged responses,
all responses with repaired evidence quotes, and a seeded 20% sample of the
remaining responses rated zero by both passes form a blinded manual
review queue. Review its prompt/response pairs before consulting the judge's
labels. Automated results remain unadjudicated until that review is completed;
do not describe them as verified incoherence prevalence. Empty-output and
repetition screens are supplementary checks, not replacements for judgment.

## Full rerun of tuning and coherence

The [latest MIG tuning rerun](verification.md#eight-condition-mig-tuning-rerun)
has verified exact matches to every retained coherence response. The original
coherence review remains retained; the commands below are optional instructions
for a further fresh run.

The [common tuning configuration](../../configs/tuning.json) and
[coherence configuration](../../configs/sae_coherence.json) use the same eight
SAE conditions, including baseline, and identical generation settings. The original
tuning run covered five conditions; the linked MIG rerun has scores for all eight.

Use these two separate commands for a fresh full rerun. Each generates all 520
responses afresh on the same GPU type and allocation profile: 65 development
prompts per condition.
The tuner applies its original bias-avoidance and response-quality rubric to
every response and selects a strength; coherence requests two judgments per
response and retains its separate review queue. Neither run fills gaps in an old
run. Coherence judgments can still require retries and manual review as described
in [the protocol](#protocol).

Follow [cluster setup](#run-on-the-cluster) below first. On the allocated GPU node,
use a clean checkout containing this update, with the cu126 intervention available.
From the repository root:

```bash
uv sync --locked
read -rsp "SoClaaS API key: " SOCLAAS_API_KEY
echo
export SOCLAAS_API_KEY
RERUN_ID="$(date +%Y%m%d-%H%M%S)"
echo "Tuning: sae-tuning-$RERUN_ID"
echo "Coherence: sae-coherence-$RERUN_ID"
uv run --locked python tuning.py --config configs/tuning.json --run-id "sae-tuning-$RERUN_ID" --method sae
uv run --locked python -m steering.sae.audit_coherence --config configs/sae_coherence.json --run-id "sae-coherence-$RERUN_ID" --stage all
unset SOCLAAS_API_KEY
```

Save both printed IDs. Use new IDs for a fresh rerun; reuse the corresponding ID
and unchanged source/configuration to resume an interrupted command. Both runs
retain all conditions, judgments, and summaries in their own directories under
`results/tuning/` and `results/coherence/`. Check both summaries before using the
new results; the recorded final SAE setting is not changed automatically.

## Run on the cluster

Follow the [shared cluster runtime](../../docs/Cluster_Runtime.md)
to obtain and verify a compute allocation.
Start from a clean checkout with the existing cu126 intervention available at
the configured path; Git does not transfer its ignored tensor files. Its
[verification record](verification.md#cu126-rebuild-and-verification) gives the
retained bundle location. Avoid rebuilding the intervention for this audit.

Ordinary resumes require the recorded source commit. For an audit already
started with the root-level script at commit `02cb8e4`, keep that checkout and
the original commands unless using the explicit [judging recovery](#recover-from-quote-formatting-failures)
below.

For a new audit, from the repository root on the allocated GPU node:

```bash
git switch sae-tuning
git pull --ff-only origin sae-tuning
uv sync --locked
uv run --locked python -m cluster_runtime
AUDIT_RUN_ID="sae-coherence-$(date +%Y%m%d-%H%M%S)"
echo "$AUDIT_RUN_ID"
uv run --locked python -m steering.sae.audit_coherence --config configs/sae_coherence.json --run-id "$AUDIT_RUN_ID" --stage generate
```

Keep the printed run ID. Generation requires the same pinned model access as
the original tuning run. Completed strength conditions are retained if the
process stops; rerun the command with that same ID and source/configuration to
resume. Use the same GPU type and software environment for generation resumes.

Judging does not load Gemma or need a GPU, but its shared imports still load
PyTorch. Use a compute allocation with network access, not the resource-limited
login node. Activate the same checkout/environment, set `AUDIT_RUN_ID` to the saved
ID if using a new shell, and enter the API key without putting it in command
history or a repository file:

```bash
read -rsp "SoClaaS API key: " SOCLAAS_API_KEY
echo
export SOCLAAS_API_KEY
uv run --locked python -m steering.sae.audit_coherence --config configs/sae_coherence.json --run-id "$AUDIT_RUN_ID" --stage judge
uv run --locked python -m steering.sae.audit_coherence --config configs/sae_coherence.json --run-id "$AUDIT_RUN_ID" --stage summarize
unset SOCLAAS_API_KEY
```

Repeat the judging command after an interruption; saved valid judgments are not
requested again. An invalid reply gets up to two retries per invocation. An item
that still fails is left unresolved while the script processes other items.
After the batch, `pending_judgments.jsonl` lists unresolved items and judging
exits with an error if any remain. They are never scored as zero or omitted from
a completed summary; summarization rejects missing judgments. Inspect their raw
replies before retrying. Earlier replies are preserved. Summarization needs no
API key. `--stage all` runs
generation, judging, and summarization in sequence when the key is already set.

### Recover from quote-formatting failures

The judge can omit Markdown markers when quoting a response, such as quoting
`Visual Aids:` when the source contains `**Visual Aids:**`. Wording reminders
did not resolve the observed failure. The script now tries a narrow deterministic
repair after exact validation fails: restore paired bold markers around one
complete `**Label:**` at the start of a single-line list item. All other characters
must match, and the source span must be unique. Partial labels, paraphrases,
whitespace or punctuation changes and ambiguous matches are rejected. Recovery
is conservatively disabled for any source containing backticks or tilde fences,
including multiline code spans.

The repaired evidence is an exact slice of the original source and passes the
unchanged validator. The raw reply, severity, and reason remain unchanged.
`evidence_recovery` on the derived judgment records the original and exact quotes,
source offsets, and raw-file identity/hash. Saved failed replies are tried before
new API calls. Every repaired response joins the blinded manual review queue,
and the summary reports recovery counts per pass/strength. This is a format
repair, not confirmation that the judge's coherence claim is correct.

For a stopped run created before this fix, use the following on a compute node.
Keep its original run ID, enter the API key as above, and ensure no other process
is writing to the run directory:

```bash
git pull --ff-only origin sae-tuning
uv run --locked python -m steering.sae.audit_coherence --config configs/sae_coherence.json --run-id "$AUDIT_RUN_ID" --stage judge --resume-judging &&
uv run --locked python -m steering.sae.audit_coherence --config configs/sae_coherence.json --run-id "$AUDIT_RUN_ID" --stage summarize
unset SOCLAAS_API_KEY
```

The explicit continuation requires a clean checkout and all 520 sealed
generations. Configuration, rubric, development data, intervention, and response
hashes must match. `config.json` retains the original generation source;
`judging_continuation.json` records the new judging source and hashes of the
retained judgment prefix and raw replies. If a later fix is needed, the same
explicit command adds `judging_continuation-2.json`, then numbered successors,
each recording the previous file's hash. Earlier records remain unchanged.
Later resumes verify every record and its retained evidence and require the
latest continuation source. The summary binds the complete record history.
Only `judge` and `summarize` stages are
allowed after continuation; no generation is repeated. Completed judgments and
valid saved replies are reused. If interrupted, repeat the same recovery command.

New raw files include the actual `request_prompt`; older files lack this field.
Retain that distinction when inspecting retry behavior. The continuation mixes
retry/format-recovery implementations, which may affect accepted judgments even
with an unchanged rubric; completed judgments are not retroactively replaced.
Manual review remains necessary: an exact quote alone does not establish a
contradiction or validate the judge's severity label.

## Retain and review results

Retain the entire `results/coherence/<run_id>/` directory for verification,
including raw API responses and the manual review queue. The script rejects
changed configuration, data, interventions, or generation contents on resume;
source changes require the explicit judging continuation above. No real-model
generation or live judging is established merely by the script's unit tests passing.

Use `summary.json` for the per-pass rates and paired comparisons,
`judgments.jsonl` for labels and quoted evidence, and `review_queue.jsonl` for the
blinded manual check. `raw_judgments/` retains the API responses, including
malformed replies; `config.json` and `generation_environment.json` record the run
settings and generation environment.
