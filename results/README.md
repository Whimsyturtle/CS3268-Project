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

`config.json` records the exact source state, all development IDs and category
counts, model and judge revisions, and intervention metadata. Generations and
judgments are linked by an opaque `review_id`. Condition metadata is never sent
to the judge. Judgment batches are saved as atomic complete snapshots, so a
restarted `judge` stage evaluates only missing IDs. `summary.json` records judge
calibration, overall and per-category aggregates, rankings, selected strengths,
and condition-anonymous qualitative summaries. A completed summary is never
overwritten.
