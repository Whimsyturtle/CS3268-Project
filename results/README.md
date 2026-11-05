# Results

Save runs under `<benchmark>/<method>/<run_id>/`. Use a single directory name
for `run_id` and never overwrite an existing run.

- `config.json`: resolved settings, source commit, checkpoint, and dataset/split identity.
- `predictions.jsonl`: `example_id` and raw `output`, one record per line. Save before scoring.
- `metrics.csv`: `subgroup`, `metric`, `value`, and `n`, one metric per row.

For metrics, use `all` for the overall subgroup, `n` for the contributing example
count, and an empty value for undefined results. See [record types](../evaluation/api.pyi).

Generated outputs are ignored by Git. `.gitkeep` files preserve empty folders.

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
not overwritten.
