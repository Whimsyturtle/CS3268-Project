# Results

Save runs under `<benchmark>/<method>/<run_id>/`. Use a single directory name
for `run_id` and never overwrite an existing run.

- `config.json`: resolved settings, source commit, checkpoint, and dataset/split identity.
- `predictions.jsonl`: `example_id` and raw `output`, one record per line. Save before scoring.
- `metrics.csv`: `subgroup`, `metric`, `value`, and `n`, one metric per row.

For metrics, use `all` for the overall subgroup, `n` for the contributing example
count, and an empty value for undefined results. See [record types](../evaluation/api.pyi).

Raw generated outputs are ignored by Git. `.gitkeep` files preserve empty folders.

## Strength tuning

The common tuner writes one directory per run:

```text
results/tuning/<run_id>/config.json
results/tuning/<run_id>/generations.jsonl
results/tuning/<run_id>/judgments.jsonl
results/tuning/<run_id>/summary.json
```

- `config.json` records the resolved configuration and source state.
- `generations.jsonl` stores generated responses.
- `judgments.jsonl` stores the judge scores.
- `summary.json` stores aggregates, rankings, and selected strengths.

Interrupted judging resumes from the saved judgments. Completed summaries are
not overwritten. Commit `config.json` and `summary.json` for selected tuning
runs, and keep `generations.jsonl` and `judgments.jsonl` outside Git.

## Coherence audit

The [development coherence audit](../steering/sae/coherence.md) retains its run
under `results/coherence/<run_id>/`; commit checked `config.json`, `summary.json`,
and all `judging_continuation*.json` records when present explicitly with `git add -f`.
Retain the remaining raw files outside Git. The whole directory stays ignored
during execution so stage transitions and resumes preserve a clean source checkout.
