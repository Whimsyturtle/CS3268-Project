# Results

Save runs under `<benchmark>/<method>/<run_id>/`. Use a single directory name
for `run_id` and never overwrite an existing run.

- `config.json`: resolved settings, source commit, checkpoint, and dataset/split identity.
- `predictions.jsonl`: `example_id` and raw `output`, one record per line. Save before scoring.
- `metrics.csv`: `subgroup`, `metric`, `value`, and `n`, one metric per row.
- `cluster_runtime.json`: evidence for new cluster runs, defined by the
  [shared runtime requirements](../docs/Cluster_Runtime.md); retain with raw artifacts, including tuning runs.

Runs of AlpacaEval 2.0 with a custom judge also contain `annotations.json` and
`leaderboard.csv` from its pairwise judge. `alpaca_annotations_cache.json`
preserves completed judge calls across rate-limit retries.

HolisticBias runs also contain `classifier_inputs.jsonl`,
`style_probabilities.jsonl`, and `classifier_manifest.json`; see the
[benchmark documentation](../evaluation/holistic_bias/README.md).

For metrics, use `all` for the overall subgroup, `n` for the contributing example
count, and an empty value for undefined results. See [record types](../evaluation/api.pyi).

For full benchmark runs (no `limit`), commit the whole run directory, so teammates
can run [paired comparisons](../GLOSSARY.md#bootstrap-confidence-interval) from its
raw outputs. Don't commit runs with a `limit`. Each run records `git status`, so
uncommitted runs appear in later runs' `config.json`.

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
not overwritten. Commit the whole run directory for selected tuning runs.
