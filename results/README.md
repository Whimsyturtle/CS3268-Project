# Results

Runs belong under `<benchmark>/<method>/<run_id>/`.
A run ID must be a single directory name. Reject existing run directories to
avoid overwriting results.

- `config.json`: resolved experiment settings, source commit, checkpoint
  identity, and dataset/split identity.
- `predictions.jsonl`: one `Prediction` record per line with `example_id` and
  raw `output`. Save before scoring so metrics can be recomputed independently.
- `metrics.csv`: one `Metric` record per row with columns `subgroup`, `metric`,
  `value`, and `n`. Use `all` for the overall subgroup; `n` counts examples
  contributing to the metric. Leave undefined values empty instead of zero.

Record types are defined in [evaluation/api.pyi](../evaluation/api.pyi).
JSON Lines supports appending individual responses and escaping multiline text;
CSV keeps metric summaries easy to compare and plot. External scorer details
may be saved beside these files and identified in `config.json`.
An optional generated `results/summary.csv` can add benchmark, method, and run ID
columns when combining runs.

Generated outputs are ignored by Git. Empty method folders contain `.gitkeep`
placeholders only; no evaluation results have been generated. Commit small,
reviewed reproducibility metadata deliberately when needed.
