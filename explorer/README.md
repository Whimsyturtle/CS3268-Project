# Intervention Explorer

The explorer is a local, read-only interface for completed evaluation, tuning,
control, and activation-capture files. It runs on CPU and never loads Gemma or
modifies result files.

## Run

From the repository root:

```bash
uv run --locked python -m explorer.app
```

Open `http://127.0.0.1:8000`. The default configuration reads `results/` and
shows unavailable states for optional data that has not been configured.

Pass a local JSON configuration when using captures, controls, named experiment
groups, or a different port:

```bash
uv run --locked python -m explorer.app --config explorer.local.json
```

Keep `explorer.local.json` untracked when it contains machine-specific paths.
Relative paths are resolved from the configuration file.

```json
{
  "results_root": "results",
  "capture_root": "captures",
  "capture_bundles": ["gemma-development"],
  "control_files": ["controls/icv-bbq.json"],
  "host": "127.0.0.1",
  "port": 8000,
  "experiment_groups": [
    {
      "name": "main-test",
      "split_role": "test",
      "points": [
        {
          "method": "baseline",
          "strength": null,
          "bias": {
            "run": "bbq/baseline/<run-id>",
            "metric": "bias_score_ambiguous",
            "subgroup": "all"
          },
          "capability": {
            "run": "mmlu_pro/baseline/<run-id>",
            "metric": "accuracy",
            "subgroup": "all"
          }
        }
      ]
    }
  ]
}
```

Add one point per method and strength. A group becomes unavailable if its points
do not share the required model, source, axis, and decoding identities.

## Optional records

A control file records `family`, non-empty `matching` criteria, a `metric`
identity, one observed `{run_id, method, effect}` object, and a list of control
`{run_id, effect}` objects. The explorer validates method-specific control
families and preserves every run ID.

A capture bundle contains:

```text
capture_config.json
capture_index.jsonl
activations.npy
```

`capture_config.json` identifies the model, development source split, and
baseline run. Each index row identifies the run, example, method, strength,
state, category, layer, hook, token selection, and source split. The finite
two-dimensional activation array has one row per index record. PCA is fitted
only on the declared development baseline rows.

The explorer specification is maintained outside this repository at
`../docs/intervention-explorer-spec.md`.

