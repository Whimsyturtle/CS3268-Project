# Shared examples

Each JSONL record contains `id`, `prompt`, `biased_response`, and `unbiased_response`.

- `construction.jsonl`: examples used to build or train interventions.
- `development.jsonl`: separate examples used to tune settings such as strength.

Use the same splits across methods. Keep related variants in the same split
and keep construction and development data separate from final evaluation datasets.
