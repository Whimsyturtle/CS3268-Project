# Results

Save runs under `<benchmark>/<method>/<run_id>/`. Use a single directory name
for `run_id` and never overwrite an existing run.

- `config.json`: resolved settings, source commit, checkpoint, and dataset/split identity.
- `predictions.jsonl`: `example_id` and raw `output`, one record per line. Save before scoring.
- `metrics.csv`: `subgroup`, `metric`, `value`, and `n`, one metric per row.

For metrics, use `all` for the overall subgroup, `n` for the contributing example
count, and an empty value for undefined results. See [record types](../evaluation/api.pyi).

Generated outputs are ignored by Git. `.gitkeep` files preserve empty folders.
Side checks that are not main results may be force-added, with a summary in their
own folder, such as [`reft_r1_checks/`](reft_r1_checks/summary.md).
